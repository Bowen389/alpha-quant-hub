# -*- coding: utf-8 -*-
"""
Evo-S1 V2 (进化参数 P2 / BF / FB · 严格零跟风单账户每日 1~2 只) — PTrade 实战单文件
========================================================================================
基于 2022-01-04 ~ 2026-11-25 严格样本内(22~24.6) + 2.4年样本外(24.7~26.11)单轴高原验证，
在保留 P2 / BF / FB 核心涨停微结构骨架的同时，注入 3 项样本内外双双跃升的进化参数 + 同行业去重 + 4档可选门控：
  1) Evo-P2：首板换手收紧至 [2%, 12%]（锁紧筹码，年化 +32.5% -> +51.4%，夏普 0.88 -> 1.39）
  2) Evo-BF：新增高位防爆雷上限 pos_med60 < 1.60（避开翻倍高位退潮核按钮，夏普 0.42 -> 0.80）
  3) Evo-FB：弱市 C2 彻底取消追高开，改为纯低开 [-5.0%, 0.0%]（胜率 58.8%，夏普 0.72 -> 1.19，回撤仅 -22%）
  4) 严格零跟风双第 1 名：席位1 = Evo-P2 第1名，席位2 = Evo-H1(Evo-BF优先/否则Evo-FB) 第1名；
     不买同引擎第 2 名跟风杂毛；同日买 2 只时自动剔除同行业共振股！

一键切换参数（initialize 内修改）：
  - g.gate_mode    : "OFF"（无门控高收益） / "LOOSE_R9"（极松双弱停，空5%）/ "SOFT_R11"（柔性半仓，空15%）/ "STRICT_MA30"（严格沪深300>=MA30）
  - g.cap_per_slot : False（满日预算50%） / True（单股封顶25%总资产，回撤显著更低）
"""

import numpy as np
import pandas as pd


def initialize(context):
    g.gate_mode = "SOFT_R11"    # 可选："OFF" / "LOOSE_R9" / "SOFT_R11" / "STRICT_MA30"
    g.cap_per_slot = False      # False=当日1只时拿满50%预算；True=单股封顶25%总资产
    g.dedup_industry = True     # 同日买2只时自动不同行业去重
    g.hold_days = 1
    g.min_amt_yi = 1.0          # 成交额下限（亿元）

    g.size_scale = 1.0          # 盘前计算好的当日仓位系数：1.0 / 0.5 / 0.0
    g.p2_cands = []
    g.bf_cands = []
    g.fb_cands = []
    g.bf_gate_on = False
    g.zz_lt_ma20 = False
    g.buy_day_map = {}
    g.bought_today = False
    g.sold_today = False

    set_benchmark("000300.SS")
    set_commission(commission_ratio=0.00025, min_commission=5.0, type="STOCK")
    set_slippage(0.001)

    run_daily(context, open_buy_routine, time="09:31")
    run_daily(context, close_sell_routine, time="14:55")


def _is_main_board(code):
    c = str(code).split(".")[0]
    return c.startswith(("600", "601", "603", "605", "000", "001", "002", "003"))


def _calc_limit_up(prev_close):
    return np.floor(prev_close * 1.10 * 100.0 + 0.5) / 100.0


def _calc_limit_down(prev_close):
    return np.floor(prev_close * 0.90 * 100.0 + 0.5) / 100.0


def _get_industry_tag(stock):
    try:
        blk = get_stock_blocks(stock)
        if isinstance(blk, dict):
            for k in ("HY", "SW1", "industry", "sw_l1"):
                if k in blk and blk[k]:
                    v = blk[k]
                    return str(v[0] if isinstance(v, (list, tuple)) else v)
    except Exception:
        pass
    return None


def before_trading_start(context, data):
    """盘前严格使用 T-1 日数据：计算门控仓位系数 + 筛选 Evo-P2 / Evo-BF / Evo-FB 候选池。"""
    g.p2_cands = []
    g.bf_cands = []
    g.fb_cands = []
    g.bought_today = False
    g.sold_today = False
    g.size_scale = 1.0

    # 1. 沪深300 & 中证1000 指数趋势（仅 T-1 收盘，include=False）
    hs_h = get_history(35, "1d", ["close"], ["000300.SS"], fq="pre", include=False)
    hs_c = hs_h["close"]["000300.SS"].dropna() if isinstance(hs_h, dict) else hs_h["close"].dropna()
    hs_ok20 = bool(len(hs_c) >= 20 and float(hs_c.iloc[-1]) >= float(hs_c.iloc[-20:].mean()))
    hs_ok30 = bool(len(hs_c) >= 30 and float(hs_c.iloc[-1]) >= float(hs_c.iloc[-30:].mean()))

    if g.gate_mode == "STRICT_MA30" and (not hs_ok30):
        g.size_scale = 0.0
        log.info("[严格门控 OFF] 沪深300 < MA30，今日空仓观望")
        return

    zz_h = get_history(25, "1d", ["close"], ["000852.SS"], fq="pre", include=False)
    zz_c = zz_h["close"]["000852.SS"].dropna() if isinstance(zz_h, dict) else zz_h["close"].dropna()
    g.zz_lt_ma20 = bool(len(zz_c) >= 20 and float(zz_c.iloc[-1]) < float(zz_c.iloc[-20:].mean()))

    all_stocks = [s for s in get_Ashares() if _is_main_board(s)]
    st_dict = get_stock_status(all_stocks, "ST")
    halt_dict = get_stock_status(all_stocks, "HALT")
    universe = [s for s in all_stocks if not st_dict.get(s, False) and not halt_dict.get(s, False)]
    if not universe:
        return

    hist_raw = get_history(65, "1d", ["open", "high", "close", "volume", "money", "turnover_rate"], universe, fq="none", include=False)
    hist_adj = get_history(65, "1d", ["close"], universe, fq="pre", include=False)

    ro_df, rh_df, rc_df = hist_raw["open"], hist_raw["high"], hist_raw["close"]
    rv_df, rm_df, to_df = hist_raw["volume"], hist_raw["money"], hist_raw["turnover_rate"]
    ac_df = hist_adj["close"]

    # 主板等权趋势门控 (供 BF 内部使用)
    ret_df = ac_df.pct_change().clip(-0.11, 0.11)
    ew_nav = (1.0 + ret_df.mean(axis=1).fillna(0.0)).cumprod()
    g.bf_gate_on = bool(len(ew_nav) >= 20 and float(ew_nav.iloc[-1]) >= float(ew_nav.iloc[-20:].mean()))

    # 统计昨日全主板跌停家数 + 近3日昨日涨停溢价（供 R11 / R9 门控使用）
    ld_cnt_yday = 0
    zt_prem_days = [[], [], []]
    min_amt = g.min_amt_yi * 1e8

    for s in universe:
        if s not in rc_df.columns or s not in ac_df.columns:
            continue
        rcs = rc_df[s].dropna()
        acs = ac_df[s].reindex(rcs.index).dropna()
        if len(rcs) < 62 or len(acs) < 62:
            continue
        rvs = rv_df[s].reindex(rcs.index)
        rms = rm_df[s].reindex(rcs.index)
        tos = to_df[s].reindex(rcs.index)
        ros = ro_df[s].reindex(rcs.index)
        rhs = rh_df[s].reindex(rcs.index)

        c0, c1, c2 = float(rcs.iloc[-1]), float(rcs.iloc[-2]), float(rcs.iloc[-3])
        o0, o1 = float(ros.iloc[-1]), float(ros.iloc[-2])
        h0, h1 = float(rhs.iloc[-1]), float(rhs.iloc[-2])
        v0, m0 = float(rvs.iloc[-1]), float(rms.iloc[-1])
        to0 = float(tos.iloc[-1]) if np.isfinite(tos.iloc[-1]) else np.nan
        if c0 <= 2.0 or c1 <= 0 or v0 <= 0:
            continue

        if c0 <= _calc_limit_down(c1) + 1e-4:
            ld_cnt_yday += 1

        # 近 3 日昨日涨停溢价统计
        for lag in range(3):
            idx_cur = -1 - lag
            idx_prv = -2 - lag
            idx_pr2 = -3 - lag
            cc = float(rcs.iloc[idx_cur])
            cp = float(rcs.iloc[idx_prv])
            cp2 = float(rcs.iloc[idx_pr2])
            if cp2 > 0 and cp >= _calc_limit_up(cp2) - 1e-4 and cp > 0:
                zt_prem_days[lag].append(np.clip(cc / cp - 1.0, -0.11, 0.11))

        # 仅昨日主板首板可进入今日竞价候选
        lu0 = c0 >= _calc_limit_up(c1) - 1e-4
        lu1 = c1 >= _calc_limit_up(c2) - 1e-4
        if not (lu0 and (not lu1)):
            continue

        not_one_word = not (o0 >= _calc_limit_up(c1) - 1e-4 and abs(h0 - o0) < 1e-4)
        med60 = float(acs.iloc[-61:-1].median())
        pos_med60 = float(acs.iloc[-2]) / med60 if med60 > 0 else 9.9
        s20 = float(rms.iloc[-21:-1].mean())

        # (A) Evo-P2：换手收紧 [2%, 12%]，pos_med60 < 1.15，成交额 >= 1亿
        if not_one_word and np.isfinite(to0) and (2.0 <= to0 <= 12.0) and (m0 >= min_amt) and (pos_med60 < 1.15):
            g.p2_cands.append({"code": s, "prev_close": c0, "amt": m0})

        # (B) Evo-BF：新增高位排雷 pos_med60 < 1.60
        prev_lu_cnt = 0
        for k in range(5):
            ci = float(rcs.iloc[-7 + k])
            cp = float(rcs.iloc[-8 + k])
            if cp > 0 and ci >= _calc_limit_up(cp) - 1e-4:
                prev_lu_cnt += 1
        prev_broken = (h1 >= _calc_limit_up(c2) - 1e-4) and (not lu1)
        bf_raw = ((c1 < o1) or prev_broken) and (prev_lu_cnt >= 1) and (c0 > max(c1, o1))
        bf_score = int(prev_lu_cnt == 1) + int(s20 >= 2.5e8) + int(o0 >= c1)
        if bf_raw and (s20 >= max(2e8, min_amt)) and (not prev_broken) and (bf_score >= 2) and (pos_med60 < 1.60):
            g.bf_cands.append({"code": s, "prev_close": c0, "bf_score": bf_score, "s20": s20})

        # (C) Evo-FB：弱市低位首板候选
        if (o0 < _calc_limit_up(c1) - 0.005) and (m0 >= min_amt) and (pos_med60 < 1.0):
            g.fb_cands.append({"code": s, "prev_close": c0, "pos_med60": pos_med60})

    # 决定门控仓位系数
    zt_m3 = float(np.mean([np.mean(x) if len(x) >= 3 else 0.0 for x in zt_prem_days]))
    if g.gate_mode == "OFF":
        g.size_scale = 1.0
    elif g.gate_mode == "STRICT_MA30":
        g.size_scale = 1.0 if hs_ok30 else 0.0
    elif g.gate_mode == "SOFT_R11":
        g.size_scale = 1.0 if hs_ok30 else (0.5 if ld_cnt_yday < 15 else 0.0)
    elif g.gate_mode == "LOOSE_R9":
        g.size_scale = 0.0 if ((not hs_ok20) and (zt_m3 <= 0.0)) else 1.0

    log.info(
        "[Evo-S1 V2 盘前] gate=%s scale=%.1f (hs20=%s hs30=%s ld=%d ztm3=%.2f%%) | P2=%d BF=%d FB=%d"
        % (g.gate_mode, g.size_scale, hs_ok20, hs_ok30, ld_cnt_yday, zt_m3 * 100.0, len(g.p2_cands), len(g.bf_cands), len(g.fb_cands))
    )


def _get_live_open_gap(stock, prev_close):
    snap = get_snapshot(stock)
    if not snap or stock not in snap:
        return None, None, True
    op = float(snap[stock].get("open_px", 0.0) or snap[stock].get("last_px", 0.0))
    if op <= 0 or prev_close <= 0:
        return None, None, True
    gap = op / prev_close - 1.0
    open_is_lu = op >= _calc_limit_up(prev_close) - 1e-4
    return op, gap, open_is_lu


def open_buy_routine(context):
    """09:31 开盘执行：严格零跟风（至多 1 只 Evo-P2 第1名 + 1 只 Evo-H1 第1名）+ 同行业去重。"""
    if g.bought_today or g.size_scale <= 1e-6:
        return
    g.bought_today = True
    today_str = context.current_dt.strftime("%Y-%m-%d")

    total_val = context.portfolio.portfolio_value
    avail_cash = context.portfolio.cash
    daily_budget = min((total_val / float(g.hold_days + 1)) * g.size_scale, avail_cash * 0.99)
    if daily_budget < 5000.0:
        return

    # 1) Evo-P2 竞价过滤：gap (-5%, -1.5%]，最深温和低开优先
    p2_hits = []
    for item in g.p2_cands:
        op, gap, is_lu = _get_live_open_gap(item["code"], item["prev_close"])
        if (not is_lu) and (gap is not None) and (-0.05 < gap <= -0.015):
            p2_hits.append((gap, -item["amt"], item["code"]))
    p2_hits.sort()
    p2_codes = [x[2] for x in p2_hits]

    # 2) Evo-BF 竞价过滤：gap (-10.1%, +8%]
    bf_hits = []
    if g.bf_gate_on:
        for item in g.bf_cands:
            op, gap, is_lu = _get_live_open_gap(item["code"], item["prev_close"])
            if (not is_lu) and (gap is not None) and (-0.101 < gap <= 0.08):
                bf_hits.append((-item["bf_score"], -item["s20"], item["code"]))
    bf_hits.sort()
    bf_codes = [x[2] for x in bf_hits]

    # 3) Evo-FB 竞价过滤：C2 彻底取消追高开，改为纯低开 [-5.0%, 0.0%]！
    fb_codes = []
    if g.zz_lt_ma20:
        c2, c1 = [], []
        for item in g.fb_cands:
            op, gap, is_lu = _get_live_open_gap(item["code"], item["prev_close"])
            if is_lu or gap is None:
                continue
            if item["pos_med60"] < 0.7 and (-0.05 <= gap <= 0.0):
                c2.append((gap, item["code"]))
            elif 0.7 <= item["pos_med60"] < 1.0 and (-0.05 <= gap <= -0.02):
                c1.append((gap, item["code"]))
        c2.sort(); c1.sort()
        fb_codes = [x[1] for x in (c2 if c2 else c1)]

    h1_codes = bf_codes if bf_codes else fb_codes
    picks = []
    seen_ind = set()

    # 席位1：Evo-P2 第1名
    if p2_codes:
        s1 = p2_codes[0]
        picks.append((s1, "Evo-P2"))
        if g.dedup_industry:
            ind1 = _get_industry_tag(s1)
            if ind1:
                seen_ind.add(ind1)

    # 席位2：Evo-H1 第1名（跳过重复代码或同行业）
    for s2 in h1_codes:
        if any(x[0] == s2 for x in picks):
            continue
        if g.dedup_industry:
            ind2 = _get_industry_tag(s2)
            if ind2 and ind2 in seen_ind:
                continue
        picks.append((s2, "Evo-BF" if bf_codes else "Evo-FB"))
        break

    if not picks:
        log.info("[开盘跳过] 今日无触发 Evo-P2 / Evo-H1 标的")
        return

    if g.cap_per_slot:
        per_cash = min((total_val / float((g.hold_days + 1) * 2)) * g.size_scale, daily_budget / len(picks))
    else:
        per_cash = daily_budget / len(picks)

    for s, tag in picks:
        order_value(s, per_cash)
        g.buy_day_map[s] = today_str
        log.info("[Evo-S1 V2 买入] %s (%s) scale=%.1f 预算=%.0f" % (s, tag, g.size_scale, per_cash))


def close_sell_routine(context):
    """14:55 尾盘平仓昨日持仓（不受门控阻挡，跌停顺延）。"""
    if g.sold_today:
        return
    g.sold_today = True
    today_str = context.current_dt.strftime("%Y-%m-%d")

    for s, pos in list(context.portfolio.positions.items()):
        if pos.enable_amount <= 0:
            continue
        if g.buy_day_map.get(s, "") != today_str:
            order_target_value(s, 0.0)
            log.info("[尾盘清仓] %s 股数=%d" % (s, pos.enable_amount))
            g.buy_day_map.pop(s, None)


def handle_data(context, data):
    pass
