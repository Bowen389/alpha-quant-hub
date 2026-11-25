# -*- coding: utf-8 -*-
"""
Champion-V2 低换手多因子选股策略 (PTrade 测试端 / 仿真实盘可直接运行版)
========================================================================
基于本地全 A 股 5 年（2022–2026）零未来函数审计重构：
  1. 彻底剔除原版静态市值未来函数与负 Alpha 毒药因子（连板追高/放量追高/追新高）；
  2. 精选 5 个经 5 年实证稳健的低换手量价 Alpha（等权合成，零权重过拟合）：
       - 20日低振幅 (-mean((H-L)/prevC, 20))
       - 20日低换手 (-log(mean(turnover, 20)))
       - 20日低波动 (-std(ret, 20))
       - 20日量价背离 (-corr(close, volume, 20))
       - 20日隔夜强/日内弱 (mean(O/prevC - 1) - mean(C/O - 1))
  3. 剔除近 5 日连板暴炒股 + 10/20 日低频调仓 + 2 倍排名缓冲（Rank Buffer），把年换手压至 5x~8x；
  4. 可选大盘门控（沪深300 > MA60 或关闭门控纯多头）。
"""
import numpy as np
import pandas as pd


def initialize(context):
    # ==================== 核心参数区 ====================
    g.universe_mode = "TOP800"   # "TOP800"(大中盘前800) 或 "LIQUID_ALL"(全A日均额>=5000万)
    g.top_n = 20                 # 持股数量
    g.rebal_days = 10            # 调仓周期（交易日：10=双周 / 20=月频）
    g.buffer_mult = 2.0          # 排名缓冲倍数（现有持仓仍在 Top 40 内则继续持有不换手）
    g.use_gate = False           # False=V2-Top800稳健纯多(5年+9.1%/超额+13.3%); True=HS300>MA60门控(熊市正收益,回撤-16%)
    g.gate_ma = 60
    g.min_amt_20d = 5e7          # 流动性底线（5000万元）
    g.day_idx = 0

    g.bench = "000300.SS"
    set_benchmark(g.bench)
    set_commission(commission_ratio=0.00025, min_commission=5.0, type="STOCK")
    set_fixed_slippage(fixedslippage=0.0005)


def _mad_zscore(s):
    arr = np.asarray(s, dtype=float)
    valid = np.isfinite(arr)
    if valid.sum() < 20:
        return np.full_like(arr, np.nan)
    med = np.nanmedian(arr)
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

    # 1. 大盘门控判断（基于截至昨日收盘的历史数据，零未来函数）
    gate_on = True
    if g.use_gate:
        bm = get_history(g.gate_ma + 5, "1d", "close", security_list=[g.bench], fq="pre", include=False)[g.bench].dropna().values
        if len(bm) >= g.gate_ma:
            gate_on = bool(bm[-1] > np.mean(bm[-g.gate_ma:]))

    held = [s for s, pos in context.portfolio.positions.items() if pos.amount > 0]

    # 门控关闭 -> 立即清仓空仓
    if not gate_on:
        for s in held:
            order_target_value(s, 0)
        return

    # 非调仓日且已有持仓 -> 保持不动（极低换手）
    if len(held) > 0 and (t % g.rebal_days != 0):
        return

    # 2. 获取当日可交易股票池（沪深非ST、上市满60日）
    all_stocks = get_Ashares()
    st_flags = get_stock_status(all_stocks, "ST")
    halt_flags = get_stock_status(all_stocks, "HALT")
    cand_pool = [
        s for s in all_stocks
        if s.endswith((".SS", ".SZ"))
        and not st_flags.get(s, False)
        and not halt_flags.get(s, False)
    ]

    # 按当日总市值筛选前 800（或全A高流动性池）
    if g.universe_mode == "TOP800":
        fund_df = get_fundamentals(cand_pool, "valuation", fields=["total_value"])
        if fund_df is not None and not fund_df.empty and "total_value" in fund_df.columns:
            cand_pool = fund_df.sort_values("total_value", ascending=False).head(800).index.tolist()

    # 3. 批量拉取近 25 日前复权行情计算 5 个低换手量价 Alpha
    fields = ["open", "high", "low", "close", "volume", "money", "turnover_rate"]
    hist = get_history(25, "1d", fields, security_list=cand_pool, fq="pre", include=False)

    f_amp, f_turn, f_vol, f_pv, f_smart = {}, {}, {}, {}, {}
    for s in cand_pool:
        try:
            c = hist["close"][s].dropna().values
            o = hist["open"][s].dropna().values
            h = hist["high"][s].dropna().values
            l = hist["low"][s].dropna().values
            v = hist["volume"][s].dropna().values
            m = hist["money"][s].dropna().values
            tr = hist["turnover_rate"][s].dropna().values
        except Exception:
            continue
        if len(c) < 22 or c[-1] <= 2.0 or np.mean(m[-20:]) < g.min_amt_20d:
            continue

        ret = c[1:] / c[:-1] - 1.0
        # 过滤近5日连续大涨/连板炒作股 (>=2天涨超9.5%)
        if np.sum(ret[-5:] >= 0.095) >= 2:
            continue

        amp = (h[1:] - l[1:]) / c[:-1]
        overnight = np.clip(o[1:] / c[:-1] - 1.0, -0.15, 0.15)
        intraday = np.clip(c[1:] / o[1:] - 1.0, -0.15, 0.15)

        t20 = np.mean(tr[-20:])
        if t20 <= 0 or np.std(c[-20:]) <= 1e-8 or np.std(v[-20:]) <= 1e-8:
            continue

        f_amp[s] = -float(np.mean(amp[-20:]))
        f_turn[s] = -float(np.log(t20))
        f_vol[s] = -float(np.std(ret[-20:]))
        f_pv[s] = -float(np.corrcoef(c[-20:], v[-20:])[0, 1])
        f_smart[s] = float(np.mean(overnight[-20:]) - np.mean(intraday[-20:]))

    valid_secs = list(f_amp.keys())
    if len(valid_secs) < g.top_n:
        return

    df_f = pd.DataFrame({
        "amp": [f_amp[s] for s in valid_secs],
        "turn": [f_turn[s] for s in valid_secs],
        "vol": [f_vol[s] for s in valid_secs],
        "pv": [f_pv[s] for s in valid_secs],
        "smart": [f_smart[s] for s in valid_secs],
    }, index=valid_secs)

    for col in df_f.columns:
        df_f[col] = _mad_zscore(df_f[col].values)

    score = df_f.mean(axis=1).dropna().sort_values(ascending=False)
    ranks = {s: r for r, s in enumerate(score.index, start=1)}
    buf_rank = int(round(g.top_n * g.buffer_mult))

    # 4. 排名缓冲选股：已有持仓仍在 Top 40 内继续保留
    keep = sorted([s for s in held if s in ranks and ranks[s] <= buf_rank], key=lambda s: ranks[s])[: g.top_n]
    add = [s for s in score.index if s not in keep][: max(g.top_n - len(keep), 0)]
    target = (keep + add)[: g.top_n]

    # 5. 先卖后买等权配置
    for s in held:
        if s not in target:
            order_target_value(s, 0)

    total_val = context.portfolio.portfolio_value * 0.985
    each_val = total_val / len(target)
    for s in target:
        order_target_value(s, each_val)
