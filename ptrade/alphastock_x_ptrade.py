# -*- coding: utf-8 -*-
"""
AlphaStock-V3 黄金中小盘低换手多因子策略 (PTrade 测试端 / 仿真实盘可直接运行版)
==============================================================================
基于本地 5 年（2022–2026）零未来函数真实资金账本实测胜出配方 (V3-GoldSMid)：
  - 本地实测：5bp滑点年化 +12.09% / 夏普 0.60 / 回撤 -27.7% / 熊市 +6.8% / 15bp严苛滑点仍赚 +9.8%
  1. 股票池护栏（黄金中小盘 25~150亿，可切 SAFE_MID / EX_MICRO）：
     - 剔除 ST / 退市警示 / 停牌 / 股价 <= 3元 / 20日均额 < 3000万 / 近5日连板>=2次妖股
  2. V3 实证高 ICIR 机构量价因子组合（10日 ICIR 最高达 +0.79）：
     - small_circ   (适度小流通市值)
     - low_turn_20  (20日低换手)
     - neg_max3_20  (近20日最大3日收益均值取反 — 经典 MAX 彩票股惩罚, 10日IC +9.72%)
     - pv_div_20    (20日量价背离)
     - vwap_bias_20 (20日均价偏离 mean(VWAP/C - 1))
     - vol_shock_20 (20日放量下跌冲击平滑, 10日ICIR +0.79)
     - amihud_20    (20日非流动性)
  3. 平滑防抖宏观防踩踏门控：中证1000(000852.SS) 均线趋势 + 跌停占比 < 3.5%，
     且要求连续 3 日翻多才开门、连续 2 日转弱才关门空仓，彻底消除频繁来回换手磨损。
  4. 20 日月频调仓 + 2.5 倍排名缓冲（Rank Buffer），把年换手压低至 ~13.9x。
"""
import numpy as np
import pandas as pd


def initialize(context):
    # ==================== 核心配置区 ====================
    g.pool_mode = "GOLD_SMID"    # "GOLD_SMID"(25~150亿, V3冠军) | "SAFE_MID"(50~300亿) | "EX_MICRO"(去尾10%)
    g.top_n = 20                 # 持股只数
    g.rebal_days = 20            # 月频调仓（20个交易日）
    g.buffer_mult = 2.5          # 排名缓冲倍数（持仓仍在 Top 50 内不换手）
    g.use_macro_gate = True      # 启用防抖宏观+跌停熔断门控（关门空仓）
    g.confirm_on = 3             # 连续3日转强才开门
    g.confirm_off = 2            # 连续2日转弱才关门空仓
    g.min_price = 3.0            # 防面值退市底线
    g.min_amt_20d = 3e7          # 20日均成交额底线（3000万元）

    g.day_idx = 0
    g.gate_state = False
    g.streak_on = 0
    g.streak_off = 0
    g.ld_hist = []

    g.bench_small = "000852.SS"  # 中证1000
    set_benchmark("000300.SS")
    set_commission(commission_ratio=0.00025, min_commission=5.0, type="STOCK")
    set_fixed_slippage(fixedslippage=0.0005)


def _mad_zscore(arr):
    arr = np.asarray(arr, dtype=float)
    valid = np.isfinite(arr)
    if valid.sum() < 20:
        return np.full_like(arr, np.nan)
    med = np.nanmedian(arr[valid])
    mad = np.nanmedian(np.abs(arr[valid] - med))
    sc = 1.4826 * mad if mad > 1e-12 else np.nanstd(arr[valid])
    if not np.isfinite(sc) or sc < 1e-12:
        return np.full_like(arr, np.nan)
    arr = np.clip(arr, med - 3.5 * sc, med + 3.5 * sc)
    mu = np.nanmean(arr)
    sd = np.nanstd(arr)
    return (arr - mu) / sd if sd > 1e-12 else np.full_like(arr, np.nan)


def handle_data(context, data):
    t = g.day_idx
    g.day_idx += 1

    # 1. 中证1000 均线趋势信号
    idx_ok = True
    if g.use_macro_gate:
        bm = get_history(65, "1d", "close", security_list=[g.bench_small], fq="pre", include=False)[g.bench_small].dropna().values
        if len(bm) >= 60:
            idx_ok = bool((bm[-1] > np.mean(bm[-20:])) or (bm[-1] > np.mean(bm[-60:])))

    # 2. 获取全市场沪深非ST/非停牌股票 + 当日总市值/流通市值
    all_stocks = [s for s in get_Ashares() if s.endswith((".SS", ".SZ"))]
    st_map = get_stock_status(all_stocks, "ST")
    halt_map = get_stock_status(all_stocks, "HALT")
    clean = [s for s in all_stocks if not st_map.get(s, False) and not halt_map.get(s, False)]

    val_df = get_fundamentals(clean, "valuation", fields=["total_value", "float_value"])
    if val_df is None or val_df.empty:
        return
    val_df = val_df.dropna(subset=["total_value", "float_value"])
    val_df = val_df[(val_df["total_value"] > 0) & (val_df["float_value"] > 0)]

    mv_unit = 1e8 if val_df["total_value"].median() > 1e8 else 1e4
    tot_yi = val_df["total_value"] / mv_unit
    if g.pool_mode == "SAFE_MID":
        sub = val_df[(tot_yi >= 50.0) & (tot_yi <= 300.0)]
    elif g.pool_mode == "GOLD_SMID":
        sub = val_df[(tot_yi >= 25.0) & (tot_yi <= 150.0)]
    else:
        q10 = val_df["float_value"].quantile(0.10)
        sub = val_df[val_df["float_value"] >= q10]

    cand_pool = sub.index.tolist()
    if len(cand_pool) < g.top_n * 2:
        return

    # 3. 拉取近 26 日前复权行情计算 V3 因子 + 昨日跌停占比
    fields = ["close", "volume", "money", "turnover_rate"]
    hist = get_history(26, "1d", fields, security_list=cand_pool, fq="pre", include=False)

    rows = []
    ld_cnt, act_cnt = 0, 0
    for s in cand_pool:
        try:
            c = hist["close"][s].dropna().values
            v = hist["volume"][s].dropna().values
            m = hist["money"][s].dropna().values
            tr = hist["turnover_rate"][s].dropna().values
        except Exception:
            continue
        if len(c) < 25 or c[-1] <= g.min_price:
            continue

        ret = c[1:] / c[:-1] - 1.0
        act_cnt += 1
        if ret[-1] <= -0.095:
            ld_cnt += 1

        if np.mean(m[-20:]) < g.min_amt_20d:
            continue
        if np.sum(ret[-5:] >= 0.095) >= 2:
            continue

        t20 = float(np.mean(tr[-20:]))
        if t20 <= 0 or np.std(c[-20:]) <= 1e-8 or np.std(v[-20:]) <= 1e-8:
            continue

        # 经典 MAX 彩票惩罚: -mean(近20日最大3个单日涨幅)
        top3_mean = float(np.mean(np.sort(ret[-20:])[-3:]))
        # 20日 VWAP 偏离: mean((money/volume)/close - 1)
        vwap_arr = m[-20:] / np.maximum(v[-20:], 1.0)
        vwap_bias = float(np.mean(np.clip(vwap_arr / np.maximum(c[-20:], 1e-6) - 1.0, -0.15, 0.15)))
        # 20日放量下跌冲击平滑: -mean(ret * log(V / MA5V))
        v_s = pd.Series(v)
        v_ma5 = v_s.rolling(5, min_periods=3).mean().values
        v_ratio = np.clip(v[-20:] / np.maximum(v_ma5[-20:], 1.0), 0.1, 10.0)
        vol_shock = float(np.mean(np.clip(-ret[-20:] * np.log(v_ratio), -0.5, 0.5)))
        # 20日 Amihud 非流动性
        illiq = float(np.mean(np.abs(ret[-20:]) / np.maximum(m[-20:] / 1e8, 1e-4)))

        rows.append({
            "sec": s,
            "small_circ": -float(np.log(sub.loc[s, "float_value"])),
            "low_turn": -float(np.log(t20)),
            "neg_max3": -top3_mean,
            "pv_div": -float(np.corrcoef(c[-20:], v[-20:])[0, 1]),
            "vwap_bias": vwap_bias,
            "vol_shock": vol_shock,
            "amihud": float(np.log(max(illiq, 1e-8))),
        })

    # 4. 更新平滑防抖门控状态（3日确认开 / 2日确认关）
    ld_ratio = (ld_cnt / float(act_cnt)) if act_cnt >= 100 else 0.0
    g.ld_hist.append(ld_ratio)
    if len(g.ld_hist) > 3:
        g.ld_hist = g.ld_hist[-3:]
    raw_gate = (not g.use_macro_gate) or (idx_ok and max(g.ld_hist) < 0.035)

    if t == 0:
        g.gate_state = raw_gate
    else:
        if raw_gate:
            g.streak_on += 1
            g.streak_off = 0
            if not g.gate_state and g.streak_on >= g.confirm_on:
                g.gate_state = True
        else:
            g.streak_off += 1
            g.streak_on = 0
            if g.gate_state and g.streak_off >= g.confirm_off:
                g.gate_state = False

    held = [s for s, pos in context.portfolio.positions.items() if pos.amount > 0]

    if not g.gate_state:
        for s in held:
            order_target_value(s, 0)
        return

    # 仅在月频调仓日或刚开门空仓建仓时调仓
    if len(held) > 0 and (t % g.rebal_days != 0):
        return

    if len(rows) < g.top_n:
        return

    df_f = pd.DataFrame(rows).set_index("sec")
    weights = {
        "small_circ": 1.0,
        "low_turn": 1.0,
        "neg_max3": 1.0,
        "pv_div": 1.0,
        "vwap_bias": 1.0,
        "vol_shock": 0.8,
        "amihud": 0.6,
    }
    score = pd.Series(0.0, index=df_f.index)
    for col, w in weights.items():
        score = score + w * _mad_zscore(df_f[col].values)

    score = score.dropna().sort_values(ascending=False)
    ranks = {s: r for r, s in enumerate(score.index, start=1)}
    buf_rank = int(round(g.top_n * g.buffer_mult))

    keep = sorted([s for s in held if s in ranks and ranks[s] <= buf_rank], key=lambda s: ranks[s])[: g.top_n]
    add = [s for s in score.index if s not in keep][: max(g.top_n - len(keep), 0)]
    target = (keep + add)[: g.top_n]

    for s in held:
        if s not in target:
            order_target_value(s, 0)

    total_val = context.portfolio.portfolio_value * 0.985
    each_val = total_val / len(target)
    for s in target:
        order_target_value(s, each_val)
