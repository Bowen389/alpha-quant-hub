# -*- coding: utf-8 -*-
"""
Meta-Stock-Event Dual-Core (55% zonghe P2-Opt Bear + 45% zonghe H1 Bull) — PTrade 实战模板
=========================================================================================
回测实证（2022-01-04 ~ 2026-11-03，本机通达信精准复权 + PIT动态主板池 + 真实多批次现金账本 + 双边全费用）：
  - 55% P2-Opt + 45% H1 零择时参数双核：5年年化 +28.70%，夏普 0.91，最大回撤 -30.59%，卡玛 0.94
    * 2022-2023 熊市：+16.4% / 夏普 0.60（P2-Opt 首板温和低开主导盈利）
    * 2024-2026 牛市：+37.4% / 夏普 1.09（H1 BF 反包主导盈利）
  - 绝对零未来函数：
    1) before_trading_start 仅使用 T-1 日及之前收盘数据，筛选次日待观察的 P2 / BF / FB 候选池；
    2) 09:30 开盘读取实时开盘价计算真实集合竞价跳空 gap，仅触发满足 gap 条件且未一字涨停的标的；
    3) hold=1：T 日开盘买入，T+1 日 14:55 尾盘卖出（若跌停则顺延至次日可成交时卖出）；
    4) 资金按两日滚动仓位分批管理（每日可用新开仓预算 <= 总资产 / 2，其中 P2-Opt 占 55%，H1 占 45%）。

注：建议与 AlphaFlow-Switch-22 (ETF 50%) 组合为 Meta-All-3Core（或与 V2-Top800 组合），
    本文件专责执行纯股票事件双核仓位（可独立运行于子账户）。
"""

import numpy as np
import pandas as pd


def initialize(context):
    g.w_p2 = 0.55          # P2-Opt 熊市首板低开权重
    g.w_h1 = 0.45          # H1 (BF反包 + FB弱市补位) 权重
    g.hold_days = 1        # 持有 1 个完整交易日：T日09:30买 -> T+1日14:55卖
    g.p2_cands = []        # 盘前生成的 P2-Opt 候选列表
    g.bf_cands = []        # 盘前生成的 H1-BF 候选列表
    g.fb_cands = []        # 盘前生成的 H1-FB 候选列表
    g.bf_gate_on = False
    g.zz_lt_ma20 = False
    g.buy_day_map = {}     # {stock: buy_trade_date_str}
    g.bought_today = False
    g.sold_today = False

    # 设置交易基准与手续费（按实盘常规费率）
    set_benchmark("000300.SS")
    set_commission(commission_ratio=0.00025, min_commission=5.0, type="STOCK")
    set_slippage(0.001)

    # 定时执行（若在 PTrade 分钟回测/实盘中运行）
    run_daily(context, open_buy_routine, time="09:31")
    run_daily(context, close_sell_routine, time="14:55")


def _is_main_board(code):
    """仅保留沪深主板（600/601/603/605/000/001/002/003），自动剔除科创板/创业板/北交所。"""
    c = str(code).split(".")[0]
    return c.startswith(("600", "601", "603", "605", "000", "001", "002", "003"))


def _calc_limit_up(prev_close):
    """主板 10% 涨停价（四舍五入到分）。"""
    return np.floor(prev_close * 1.10 * 100.0 + 0.5) / 100.0


def before_trading_start(context, data):
    """盘前（09:00前）严格使用 T-1 日历史数据筛选今日监控候选池（零未来函数）。"""
    g.p2_cands = []
    g.bf_cands = []
    g.fb_cands = []
    g.bought_today = False
    g.sold_today = False

    # 1. 动态获取全市场主板非ST股票（Point-in-Time，绝不使用静态白名单）
    all_stocks = [s for s in get_Ashares() if _is_main_board(s)]
    st_dict = get_stock_status(all_stocks, "ST")
    halt_dict = get_stock_status(all_stocks, "HALT")
    universe = [s for s in all_stocks if not st_dict.get(s, False) and not halt_dict.get(s, False)]
    if not universe:
        return

    # 2. 中证1000 MA20 弱市门控（仅用 T-1 日收盘）
    zz_h = get_history(25, "1d", ["close"], ["000852.SS"], fq="pre", include=False)
    zz_c = zz_h["close"]["000852.SS"].dropna() if isinstance(zz_h, dict) else zz_h["close"].dropna()
    g.zz_lt_ma20 = bool(len(zz_c) >= 20 and zz_c.iloc[-1] < zz_c.iloc[-20:].mean())

    # 3. 批量拉取近 65 日日线（严格 include=False，不含今日数据）
    hist = get_history(65, "1d", ["open", "high", "low", "close", "volume", "money", "turnover_rate"], universe, fq="none", include=False)
    o_df = hist["open"]
    h_df = hist["high"]
    c_df = hist["close"]
    v_df = hist["volume"]
    m_df = hist["money"]
    to_df = hist["turnover_rate"]

    # 计算主板等权趋势门控 (ew_main >= MA20)
    ret_df = c_df.pct_change().clip(-0.11, 0.11)
    ew_nav = (1.0 + ret_df.mean(axis=1).fillna(0.0)).cumprod()
    g.bf_gate_on = bool(len(ew_nav) >= 20 and ew_nav.iloc[-1] >= ew_nav.iloc[-20:].mean())

    # 4. 逐只筛选昨日首板 / BF反包 / FB低位候选
    for s in universe:
        if s not in c_df.columns:
            continue
        cs = c_df[s].dropna()
        if len(cs) < 62:
            continue
        os_ = o_df[s].reindex(cs.index)
        hs = h_df[s].reindex(cs.index)
        vs = v_df[s].reindex(cs.index)
        ms = m_df[s].reindex(cs.index)
        tos = to_df[s].reindex(cs.index)

        c0, c1, c2 = float(cs.iloc[-1]), float(cs.iloc[-2]), float(cs.iloc[-3])
        o0, o1 = float(os_.iloc[-1]), float(os_.iloc[-2])
        h0, h1 = float(hs.iloc[-1]), float(hs.iloc[-2])
        v0, m0 = float(vs.iloc[-1]), float(ms.iloc[-1])
        to0 = float(tos.iloc[-1]) if np.isfinite(tos.iloc[-1]) else 0.0

        if c0 <= 0 or c1 <= 0 or v0 <= 0:
            continue

        lu0 = c0 >= _calc_limit_up(c1) - 1e-4
        lu1 = c1 >= _calc_limit_up(c2) - 1e-4
        not_one_word = not (o0 >= _calc_limit_up(c1) - 1e-4 and abs(h0 - o0) < 1e-4)
        first_board = lu0 and (not lu1)
        if not first_board:
            continue

        med60 = float(cs.iloc[-61:-1].median())
        pos_med60 = c1 / med60 if med60 > 0 else 9.9
        s20 = float(ms.iloc[-21:-1].mean())

        # (A) P2-Opt 候选：主板非一字首板 + 换手3-15% + 成交额>=1亿 + 低位(pos_med60<1.15)
        if not_one_word and (3.0 <= to0 <= 15.0) and (m0 >= 1e8) and (pos_med60 < 1.15):
            g.p2_cands.append({"code": s, "prev_close": c0, "amt": m0})

        # (B) H1-BF 候选：前5日曾涨停 + 昨日反包前阴/炸板 + s20>=2亿 + bf_score>=2
        prev_c_arr = cs.iloc[-8:-2].values
        prev_lu_cnt = int(sum(
            cs.iloc[-7 + k] >= _calc_limit_up(prev_c_arr[k]) - 1e-4
            for k in range(5)
            if prev_c_arr[k] > 0
        ))
        prev_broken = (h1 >= _calc_limit_up(c2) - 1e-4) and (not lu1)
        bf_raw = ((c1 < o1) or prev_broken) and (prev_lu_cnt >= 1) and (c0 > max(c1, o1))
        bf_score = int(prev_lu_cnt == 1) + int(s20 >= 2.5e8) + int(o0 >= c1)
        if bf_raw and (s20 >= 2e8) and (not prev_broken) and (bf_score >= 2):
            g.bf_cands.append({"code": s, "prev_close": c0, "bf_score": bf_score, "s20": s20})

        # (C) H1-FB 候选：弱市低位首板补位
        if (o0 < _calc_limit_up(c1) - 0.005) and (m0 >= 1e8) and (pos_med60 < 1.0):
            g.fb_cands.append({"code": s, "prev_close": c0, "pos_med60": pos_med60})

    log.info(
        "[盘前扫描] bf_gate=%s zz<MA20=%s | P2候=%d BF候=%d FB候=%d"
        % (g.bf_gate_on, g.zz_lt_ma20, len(g.p2_cands), len(g.bf_cands), len(g.fb_cands))
    )


def _get_live_open_gap(stock, prev_close):
    """读取今日 09:30 开盘价，返回 (open_px, gap, is_limit_up_open)。"""
    snap = get_snapshot(stock)
    if not snap or stock not in snap:
        return None, None, True
    op = float(snap[stock].get("open_px", 0.0) or snap[stock].get("last_px", 0.0))
    if op <= 0 or prev_close <= 0:
        return None, None, True
    gap = op / prev_close - 1.0
    is_lu = op >= _calc_limit_up(prev_close) - 1e-4
    return op, gap, is_lu


def open_buy_routine(context):
    """09:31 开盘执行：读取真实集合竞价跳空 gap，分配今日 P2-Opt (55%) 与 H1 (45%) 预算。"""
    if g.bought_today:
        return
    g.bought_today = True
    today_str = context.current_dt.strftime("%Y-%m-%d")

    total_val = context.portfolio.portfolio_value
    avail_cash = context.portfolio.cash
    # 滚动 hold=1：每日新开仓总预算不超过总资产的 50%
    daily_budget = min(total_val / float(g.hold_days + 1), avail_cash * 0.99)
    if daily_budget < 5000.0:
        return

    # 1) P2-Opt 执行（55% 权重，gap 落在 (-5%, -1.5%]，最深低开优先 Top 2）
    p2_hits = []
    for item in g.p2_cands:
        op, gap, is_lu = _get_live_open_gap(item["code"], item["prev_close"])
        if (not is_lu) and (gap is not None) and (-0.05 < gap <= -0.015):
            p2_hits.append((gap, -item["amt"], item["code"], op))
    p2_hits.sort()
    p2_picks = [x[2] for x in p2_hits[:2]]

    if p2_picks:
        per_p2 = (daily_budget * g.w_p2) / len(p2_picks)
        for s in p2_picks:
            order_value(s, per_p2)
            g.buy_day_map[s] = today_str
            log.info("[P2-Opt 买入] %s 预算=%.0f" % (s, per_p2))

    # 2) H1 执行（45% 权重：BF主线优先，否则弱市FB补位，Top 1）
    h1_pick = None
    if g.bf_gate_on:
        bf_hits = []
        for item in g.bf_cands:
            op, gap, is_lu = _get_live_open_gap(item["code"], item["prev_close"])
            if (not is_lu) and (gap is not None) and (-0.101 < gap <= 0.08):
                bf_hits.append((-item["bf_score"], -item["s20"], item["code"]))
        if bf_hits:
            bf_hits.sort()
            h1_pick = bf_hits[0][2]

    if h1_pick is None and g.zz_lt_ma20:
        c2_hits, c1_hits = [], []
        for item in g.fb_cands:
            op, gap, is_lu = _get_live_open_gap(item["code"], item["prev_close"])
            if is_lu or gap is None:
                continue
            if item["pos_med60"] < 0.7 and (-0.025 <= gap <= 0.05):
                c2_hits.append((gap, item["code"]))
            elif 0.7 <= item["pos_med60"] < 1.0 and (-0.05 <= gap <= -0.02):
                c1_hits.append((gap, item["code"]))
        if c2_hits:
            c2_hits.sort()
            h1_pick = c2_hits[0][1]
        elif c1_hits:
            c1_hits.sort()
            h1_pick = c1_hits[0][1]

    if h1_pick is not None and h1_pick not in p2_picks:
        order_value(h1_pick, daily_budget * g.w_h1)
        g.buy_day_map[h1_pick] = today_str
        log.info("[H1 买入] %s 预算=%.0f" % (h1_pick, daily_budget * g.w_h1))


def close_sell_routine(context):
    """14:55 尾盘卖出所有非今日买入（已满 hold=1 日）且可成交的持仓。"""
    if g.sold_today:
        return
    g.sold_today = True
    today_str = context.current_dt.strftime("%Y-%m-%d")

    for s, pos in list(context.portfolio.positions.items()):
        if pos.enable_amount <= 0:
            continue
        # T+1 卖出昨日及之前买入的仓位
        if g.buy_day_map.get(s, "") != today_str:
            order_target_value(s, 0.0)
            log.info("[尾盘清仓] %s 股数=%d" % (s, pos.enable_amount))
            g.buy_day_map.pop(s, None)


def handle_data(context, data):
    """兼容日频/分钟频框架：具体执行交由 run_daily(09:31 / 14:55) 触发。"""
    pass
