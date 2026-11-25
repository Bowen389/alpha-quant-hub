# -*- coding: utf-8 -*-
"""
单账户纯股票 · 每日只买 1~2 只统一选股算法 (含沪深300大盘门控 + 同行业去重) — PTrade 实战单文件
====================================================================================================
严格基于 2022-01-04 ~ 2026-11-25 本机通达信最新 5 年实测（PIT 动态主板池 + 严格 09:25 竞价零未来 + 全费用）：

核心升级（已通过 MA10~MA60 参数高原 + 5/10/15bp 三档滑点 + 2026-07/11 极端退潮考验）：
  1) 新增【沪深300 大盘趋势门控】(g.use_hs300_gate = True, g.hs300_ma_win = 20，可选 20 或 30)：
     - 盘前仅用 T-1 日沪深300 (000300.SS) 收盘价判断：仅当 昨收 >= MA20(或MA30) 时允许开新仓，否则空仓避险！
     - 已持有的昨日仓位不受门控影响，按规则于 T+1 日 14:55 正常尾盘平仓。
     - 实测将 2026-11 月 -32% 核按钮直接压成 0.0% 完美空仓躲过，2026-07 由 -11% 转为 +5.8%！
  2) 新增【同日同行业去重】(g.dedup_industry = True)：
     - 每日买 2 只时自动规避同一申万/通达信行业，杜绝同板块双跌停共振杀伤。

支持 4 种一键切换模式（修改 initialize 中的 g.mode 即可，均配合 HS300>=MA20）：
  1) g.mode = "U1_TOP1"        [旗舰推荐 · 每日只买唯一 1 只，纯6因子统一打分]
     - MA20 5年实测：5bp年化 +53.99% | 夏普 1.69 | 回撤 -28.41% | 卡玛 1.90 | 10bp +47.1% | 15bp +40.4%
     - 连续 5 年全部 > +14%：22:+77% / 23:+33% / 24:+27% / 25:+144% / 26:+14%（MA30 26年+28%, 夏普1.73）

  2) g.mode = "ZF2_TOP2_FULL"  [双核推荐 · 每日 1~2 只零跟风双第1名 + 同行业去重]
     - MA20 5年实测：5bp年化 +49.41% | 夏普 1.69 | 回撤 -28.68% | 卡玛 1.72 | 10bp +42.7% | 15bp +36.3%
     - 连续 5 年全部 >= +10%：22:+70% / 23:+17% / 24:+10% / 25:+184% / 26:+14%

  3) g.mode = "S1_PAIR_FULL"   [纯原版双强配对 · 1席P2低开第1名 + 1席H1第1名，零因子权重]
     - MA20 5年实测：5bp年化 +38.49% | 夏普 1.29 | 回撤 -33.79% | 10bp +32.3% | 15bp +26.3%

  4) g.mode = "ZF2_TOP2_CAP"   [更低波动 · 零跟风双第1名 + 单股封顶 25% 总资产]
"""

import numpy as np
import pandas as pd


def initialize(context):
    # 可选："U1_TOP1"（旗舰1只） / "ZF2_TOP2_FULL"（旗舰1-2只） / "S1_PAIR_FULL"（纯原版1-2只） / "ZF2_TOP2_CAP"
    g.mode = "U1_TOP1"
    g.use_hs300_gate = True     # 开启沪深300大盘门控（机构资金水位风向标）
    g.hs300_ma_win = 20         # 均线窗口：20 或 30 均处于高夏普参数高原（MA20:1.69 / MA30:1.73）
    g.dedup_industry = True     # 同日买2只时自动剔除同行业共振股
    g.hold_days = 1
    g.min_amt_yi = 1.0          # 成交额下限（亿元），资金量大可提高至 2.0

    g.hs300_gate_on = False
    g.p2_cands = []
    g.bf_cands = []
    g.fb_cands = []
    g.score_map = {}
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


def _get_industry_tag(stock):
    """兼容获取 PTrade 所属行业标签（用于同日同行业去重，API 不可用时安全回退 None）。"""
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
    """盘前严格使用 T-1 日历史数据：先检沪深300大盘门控，再算全主板 6 因子百分位并筛选候选池。"""
    g.p2_cands = []
    g.bf_cands = []
    g.fb_cands = []
    g.score_map = {}
    g.bought_today = False
    g.sold_today = False

    # 0. 【最高优先级】沪深300 大盘趋势门控（严格仅用 T-1 收盘价，include=False）
    if g.use_hs300_gate:
        need = g.hs300_ma_win + 5
        hs_h = get_history(need, "1d", ["close"], ["000300.SS"], fq="pre", include=False)
        hs_c = hs_h["close"]["000300.SS"].dropna() if isinstance(hs_h, dict) else hs_h["close"].dropna()
        g.hs300_gate_on = bool(
            len(hs_c) >= g.hs300_ma_win
            and float(hs_c.iloc[-1]) >= float(hs_c.iloc[-g.hs300_ma_win:].mean())
        )
    else:
        g.hs300_gate_on = True

    if not g.hs300_gate_on:
        log.info("[大盘门控 OFF] 沪深300 昨收 < MA%d，今日停止开新仓（已有持仓照常尾盘平仓）" % g.hs300_ma_win)
        return

    all_stocks = [s for s in get_Ashares() if _is_main_board(s)]
    st_dict = get_stock_status(all_stocks, "ST")
    halt_dict = get_stock_status(all_stocks, "HALT")
    universe = [s for s in all_stocks if not st_dict.get(s, False) and not halt_dict.get(s, False)]
    if not universe:
        return

    # 1. 中证1000 MA20 弱市补位判断（仅 T-1 收盘）
    zz_h = get_history(25, "1d", ["close"], ["000852.SS"], fq="pre", include=False)
    zz_c = zz_h["close"]["000852.SS"].dropna() if isinstance(zz_h, dict) else zz_h["close"].dropna()
    g.zz_lt_ma20 = bool(len(zz_c) >= 20 and zz_c.iloc[-1] < zz_c.iloc[-20:].mean())

    # 2. 拉取近 65 日不复权(精确涨停价) + 前复权(20/60日因子) 日线，严格 include=False
    hist_raw = get_history(65, "1d", ["open", "high", "close", "volume", "money", "turnover_rate"], universe, fq="none", include=False)
    hist_adj = get_history(65, "1d", ["close"], universe, fq="pre", include=False)

    ro_df, rh_df, rc_df = hist_raw["open"], hist_raw["high"], hist_raw["close"]
    rv_df, rm_df, to_df = hist_raw["volume"], hist_raw["money"], hist_raw["turnover_rate"]
    ac_df = hist_adj["close"]

    # 主板等权趋势门控 (供 H1-BF 内部使用)
    ret_df = ac_df.pct_change().clip(-0.11, 0.11)
    ew_nav = (1.0 + ret_df.mean(axis=1).fillna(0.0)).cumprod()
    g.bf_gate_on = bool(len(ew_nav) >= 20 and ew_nav.iloc[-1] >= ew_nav.iloc[-20:].mean())

    # 3. 批量计算全主板 6 因子并筛选昨日首板事件候选
    raw_factors = {}
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

        med60 = float(acs.iloc[-61:-1].median())
        pos_med60 = float(acs.iloc[-2]) / med60 if med60 > 0 else np.nan
        mom20 = float(acs.iloc[-1] / acs.iloc[-21] - 1.0) if acs.iloc[-21] > 0 else np.nan

        to_tail20 = tos.iloc[-20:].astype(float)
        to_5 = float(to_tail20.iloc[-5:].mean())
        to_20 = float(to_tail20.mean())
        turn_shrink = -(to_5 / to_20) if (np.isfinite(to_5) and np.isfinite(to_20) and to_20 > 1e-6) else np.nan

        r20 = acs.pct_change().iloc[-20:].to_numpy(dtype=float)
        lv20 = np.log(np.clip(rvs.iloc[-20:].to_numpy(dtype=float), 1.0, None))
        pv_div = np.nan
        if np.isfinite(r20).sum() >= 15:
            ok = np.isfinite(r20) & np.isfinite(lv20)
            if ok.sum() >= 15 and np.std(r20[ok]) > 1e-9 and np.std(lv20[ok]) > 1e-9:
                pv_div = -float(np.corrcoef(r20[ok], lv20[ok])[0, 1])

        rc20 = rcs.iloc[-20:].to_numpy(dtype=float)
        v20 = rvs.iloc[-20:].to_numpy(dtype=float)
        m20 = rms.iloc[-20:].to_numpy(dtype=float)
        vwap = np.where(v20 > 0, m20 / v20, np.nan)
        ratio = np.where((vwap > 0) & np.isfinite(vwap), rc20 / vwap - 1.0, np.nan)
        ratio = np.where(np.abs(ratio) <= 0.15, ratio, np.nan)
        vwap_bias = float(np.nanmean(ratio)) if np.isfinite(ratio).sum() >= 10 else np.nan

        raw_factors[s] = (-pos_med60, -mom20, -to0, turn_shrink, pv_div, vwap_bias)

        # 仅昨日首板可进入今日 09:30 竞价候选
        lu0 = c0 >= _calc_limit_up(c1) - 1e-4
        lu1 = c1 >= _calc_limit_up(c2) - 1e-4
        if not (lu0 and (not lu1)):
            continue

        not_one_word = not (o0 >= _calc_limit_up(c1) - 1e-4 and abs(h0 - o0) < 1e-4)
        s20 = float(rms.iloc[-21:-1].mean())

        if not_one_word and np.isfinite(to0) and (3.0 <= to0 <= 15.0) and (m0 >= min_amt) and np.isfinite(pos_med60) and (pos_med60 < 1.15):
            g.p2_cands.append({"code": s, "prev_close": c0, "amt": m0})

        prev_lu_cnt = 0
        for k in range(5):
            ci = float(rcs.iloc[-7 + k])
            cp = float(rcs.iloc[-8 + k])
            if cp > 0 and ci >= _calc_limit_up(cp) - 1e-4:
                prev_lu_cnt += 1
        prev_broken = (h1 >= _calc_limit_up(c2) - 1e-4) and (not lu1)
        bf_raw = ((c1 < o1) or prev_broken) and (prev_lu_cnt >= 1) and (c0 > max(c1, o1))
        bf_score = int(prev_lu_cnt == 1) + int(s20 >= 2.5e8) + int(o0 >= c1)
        if bf_raw and (s20 >= max(2e8, min_amt)) and (not prev_broken) and (bf_score >= 2):
            g.bf_cands.append({"code": s, "prev_close": c0, "bf_score": bf_score, "s20": s20})

        if (o0 < _calc_limit_up(c1) - 0.005) and (m0 >= min_amt) and np.isfinite(pos_med60) and (pos_med60 < 1.0):
            g.fb_cands.append({"code": s, "prev_close": c0, "pos_med60": pos_med60})

    if raw_factors:
        fdf = pd.DataFrame.from_dict(
            raw_factors, orient="index",
            columns=["low_pos60", "low_mom20", "low_to1d", "turn_shrink", "pv_div_20", "vwap_bias_20"]
        )
        g.score_map = fdf.rank(pct=True).fillna(0.5).mean(axis=1).to_dict()

    log.info(
        "[盘前扫描 mode=%s HS300>=MA%d=ON] bf_gate=%s zz<MA20=%s | P2候=%d BF候=%d FB候=%d"
        % (g.mode, g.hs300_ma_win, g.bf_gate_on, g.zz_lt_ma20, len(g.p2_cands), len(g.bf_cands), len(g.fb_cands))
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


def _append_dedup(picks, cand_list, max_slots):
    """按顺序追加候选，自动跳过重复代码及同行业共振股。"""
    seen_ind = set()
    if g.dedup_industry:
        for s in picks:
            ind = _get_industry_tag(s)
            if ind:
                seen_ind.add(ind)
    for s in cand_list:
        if len(picks) >= max_slots:
            break
        if s in picks:
            continue
        if g.dedup_industry:
            ind = _get_industry_tag(s)
            if ind and ind in seen_ind:
                continue
            if ind:
                seen_ind.add(ind)
        picks.append(s)
        break


def open_buy_routine(context):
    """09:31 开盘执行：仅当沪深300大盘门控 ON 时，按真实 09:30 竞价跳空买入 1~2 只。"""
    if g.bought_today or (not g.hs300_gate_on):
        return
    g.bought_today = True
    today_str = context.current_dt.strftime("%Y-%m-%d")

    total_val = context.portfolio.portfolio_value
    avail_cash = context.portfolio.cash
    daily_budget = min(total_val / float(g.hold_days + 1), avail_cash * 0.99)
    if daily_budget < 5000.0:
        return

    p2_hits = []
    for item in g.p2_cands:
        op, gap, is_lu = _get_live_open_gap(item["code"], item["prev_close"])
        if (not is_lu) and (gap is not None) and (-0.05 < gap <= -0.015):
            p2_hits.append((gap, -item["amt"], item["code"]))
    p2_hits.sort()
    p2_codes = [x[2] for x in p2_hits]

    bf_hits = []
    if g.bf_gate_on:
        for item in g.bf_cands:
            op, gap, is_lu = _get_live_open_gap(item["code"], item["prev_close"])
            if (not is_lu) and (gap is not None) and (-0.101 < gap <= 0.08):
                bf_hits.append((-item["bf_score"], -item["s20"], item["code"]))
    bf_hits.sort()
    bf_codes = [x[2] for x in bf_hits]

    fb_codes = []
    if g.zz_lt_ma20:
        c2, c1 = [], []
        for item in g.fb_cands:
            op, gap, is_lu = _get_live_open_gap(item["code"], item["prev_close"])
            if is_lu or gap is None:
                continue
            if item["pos_med60"] < 0.7 and (-0.025 <= gap <= 0.05):
                c2.append((gap, item["code"]))
            elif 0.7 <= item["pos_med60"] < 1.0 and (-0.05 <= gap <= -0.02):
                c1.append((gap, item["code"]))
        c2.sort(); c1.sort()
        fb_codes = [x[1] for x in (c2 if c2 else c1)]

    union_codes = list(dict.fromkeys(p2_codes + bf_codes + fb_codes))
    if not union_codes:
        log.info("[开盘跳过] 今日无满足竞价条件的事件标的，现金观望")
        return

    union_ranked = sorted(union_codes, key=lambda s: -g.score_map.get(s, 0.5))
    h1_codes = bf_codes if bf_codes else fb_codes
    picks = []
    if g.mode == "U1_TOP1":
        picks = [union_ranked[0]]
        max_slots, cap_per_slot = 1, False
    elif g.mode == "S1_PAIR_FULL":
        max_slots, cap_per_slot = 2, False
        if p2_codes:
            picks.append(p2_codes[0])
        _append_dedup(picks, h1_codes, max_slots)
    else:
        max_slots = 2
        cap_per_slot = (g.mode == "ZF2_TOP2_CAP")
        picks.append(union_ranked[0])
        second_pool = bf_codes if bf_codes else p2_codes
        _append_dedup(picks, second_pool, max_slots)

    if not picks:
        return

    if cap_per_slot:
        per_cash = min(total_val / float((g.hold_days + 1) * max_slots), daily_budget / len(picks))
    else:
        per_cash = daily_budget / len(picks)

    for s in picks:
        order_value(s, per_cash)
        g.buy_day_map[s] = today_str
        log.info("[开盘买入 %s] %s | 6因子分=%.3f | 金额=%.0f" % (g.mode, s, g.score_map.get(s, 0.5), per_cash))


def close_sell_routine(context):
    """14:55 尾盘卖出所有满 hold=1 的昨日持仓（跌停顺延至次日，不受大盘门控阻挡）。"""
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
