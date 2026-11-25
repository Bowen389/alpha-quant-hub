# -*- coding: utf-8 -*-
"""
S9-K5-Evo 全领域 ETF 趋势轮动策略 (PTrade 测试端 / 仿真实盘可直接运行版)
==========================================================================
融合原版 s9-duoyinzi (K5) 核心架构 + 本地通达信消融验证通过的正向增强：
  1. 广度 + 沪深300 MA60 OR 门控（市场转熊自动切入防御层）
  2. 40日指数衰减加权对数回归 R^2 动量 × 波动率惩罚 (score / vol^0.5)
  3. 放量排雷 (成交额 > 前5日均额 × 2.0)、趋势破位 (MA20/MA60)、-10%收盘止损
  4. 最短持仓5日 + 冷静期5日 + 15%换仓迟滞（低换手）
  5. 防御层升级：门控关闭时在国债(511260)/黄金(518880)间按动量择强，双弱切短融(511360)
  6. 目标波动率仓位缩放（默认 15% 稳健档，可改 0.22 为进取甜点档；闲置现金买银华日利 511880）
"""
import numpy as np


def initialize(context):
    # ==================== 参数区（切换档位只需改这里） ====================
    g.top_n = 2                   # 进攻持仓只数 (K5=2)
    g.target_vol = 0.15           # 目标年化波动率: 0.15=K5稳健档 / 0.22=进取甜点档 / 0.0=关闭缩放
    g.stop_loss = 0.10            # 收盘止损 -10%
    g.mom_lookback = 40           # WLS 动量窗口
    g.mom_decay = 0.05            # 指数衰减系数
    g.vol_penalty = 0.5           # 波动率惩罚指数 (score / vol^0.5; 设0退回原版K5)
    g.vol_spike_mult = 2.0        # 放量排雷倍数
    g.min_hold = 5                # 最短持仓交易日
    g.cooldown = 5                # 卖出后冷静期交易日
    g.switch_margin = 0.15        # 换仓得分优势门槛
    g.regime_thr = 0.20           # 广度多头占比阈值 (OR 门控)
    g.def_mode = "bond_gold_top1" # "bond_gold_top1"(择强+短融) 或 "bond_gold"(原版50/50)

    # 基准与防御/现金标的
    g.bench = "000300.SS"
    g.bond_etf = "511260.SS"      # 十年国债ETF
    g.gold_etf = "518880.SS"      # 黄金ETF
    g.short_bond = "511360.SS"    # 短融ETF (双弱避险)
    g.cash_etf = "511880.SS"      # 银华日利 (目标波动缩放后的闲置现金计息)

    # 已按主题去重的高流动性全领域代表 ETF 池（覆盖宽基/行业/科技/周期/红利/跨境/商品，PTrade 回测极快）
    g.etf_pool = [
        # 宽基
        "510300.SS", "510050.SS", "510500.SS", "512100.SS",
        "159915.SZ", "588000.SS", "159901.SZ", "510880.SS",
        # 科技/制造
        "512480.SS", "515880.SS", "515000.SS", "515030.SS",
        "515790.SS", "512660.SS", "512720.SS", "159819.SZ",
        "515070.SS", "512980.SS", "516160.SS", "562500.SS",
        # 金融/消费/医药
        "512880.SS", "512800.SS", "512070.SS", "159928.SZ",
        "512690.SS", "512010.SS", "512170.SS", "159938.SZ",
        "512200.SS", "159996.SZ",
        # 周期/资源/公用
        "512400.SS", "515220.SS", "515210.SS", "516020.SS",
        "159611.SZ", "516150.SS", "515230.SS", "512900.SS",
        # 跨境/商品/转债
        "513100.SS", "513500.SS", "513180.SS", "513050.SS",
        "513090.SS", "513520.SS", "513030.SS", "518880.SS",
        "159985.SZ", "159980.SZ", "511380.SS", "511260.SS",
    ]

    # 状态记录
    g.day_idx = 0
    g.entry_day = {}
    g.entry_price = {}
    g.last_exit_day = {}
    g.breadth_hist = []

    set_benchmark(g.bench)
    set_commission(commission_ratio=0.0001, min_commission=5.0, type="ETF")
    set_fixed_slippage(fixedslippage=0.0002)


def _calc_wls_score(close_arr, L=40, decay=0.05, vol_penalty=0.5):
    arr = np.asarray(close_arr, dtype=float)
    if len(arr) < L or np.any(arr <= 0) or np.any(~np.isfinite(arr)):
        return np.nan
    y = np.log(arr[-L:])
    j = np.arange(L, dtype=float)
    w = np.exp(decay * (j - (L - 1)))
    w /= w.sum()
    x = j - (w * j).sum()
    a2 = (w * x * x).sum()
    y_bar = (w * y).sum()
    dy = y - y_bar
    sxy = (w * x * dy).sum()
    sst = (w * dy * dy).sum()
    if a2 <= 1e-14 or sst <= 1e-14:
        return 0.0
    slope = sxy / a2
    r2 = np.clip((slope * slope) * a2 / sst, 0.0, 1.0)
    mom = np.exp(np.clip(slope * 252.0, -5.0, 5.0)) - 1.0
    score = np.clip(mom, -1.0, 20.0) * r2
    if vol_penalty > 0 and len(arr) >= 21:
        lr = np.diff(np.log(arr[-21:]))
        vol20 = np.std(lr) * np.sqrt(252.0)
        if np.isfinite(vol20) and vol20 > 0.03:
            score = score / (vol20 ** vol_penalty)
    return float(score)


def handle_data(context, data):
    t = g.day_idx
    g.day_idx += 1

    # 1. 拉取截至昨日收盘的 65 日前复权收盘价与成交额（无未来函数）
    all_secs = list(dict.fromkeys(g.etf_pool + [g.bond_etf, g.gold_etf, g.short_bond, g.cash_etf]))
    hist_c = get_history(65, "1d", "close", security_list=all_secs, fq="pre", include=False)
    hist_a = get_history(10, "1d", "money", security_list=all_secs, fq="pre", include=False)
    bm_c = get_history(65, "1d", "close", security_list=[g.bench], fq="pre", include=False)[g.bench].dropna().values

    # 2. 计算各标的特征（趋势、得分、放量）与市场广度
    scores = {}
    trend_ok = {}
    vol_spike = {}
    valid_cnt = 0
    trend_cnt = 0

    for sec in all_secs:
        try:
            c = hist_c[sec].dropna().values
            a = hist_a[sec].dropna().values
        except Exception:
            continue
        if len(c) < 60 or c[-1] <= 0:
            continue
        ma20 = np.mean(c[-20:])
        ma60 = np.mean(c[-60:])
        t_ok = bool(c[-1] > ma20 and ma20 > ma60)
        trend_ok[sec] = t_ok
        scores[sec] = _calc_wls_score(c, g.mom_lookback, g.mom_decay, g.vol_penalty)
        if len(a) >= 6:
            base_a = np.mean(a[-6:-1])
            vol_spike[sec] = bool(base_a > 0 and a[-1] > g.vol_spike_mult * base_a)
        else:
            vol_spike[sec] = False

        if sec in g.etf_pool:
            valid_cnt += 1
            if t_ok:
                trend_cnt += 1

    # 3. 广度 + 沪深300 MA60 OR 门控
    breadth = trend_cnt / max(valid_cnt, 1)
    g.breadth_hist.append(breadth)
    if len(g.breadth_hist) > 5:
        g.breadth_hist = g.breadth_hist[-5:]
    br_ma5 = float(np.mean(g.breadth_hist))
    idx_on = len(bm_c) >= 60 and bm_c[-1] > np.mean(bm_c[-60:])
    gate_on = bool(idx_on or (br_ma5 >= g.regime_thr))

    # 4. 当前非现金持仓检查
    held = [
        sec for sec, pos in context.portfolio.positions.items()
        if pos.amount > 0 and sec != g.cash_etf
    ]

    if not gate_on:
        # 防御层
        if g.def_mode == "bond_gold_top1":
            cand = [(s, scores.get(s, -1.0)) for s in (g.bond_etf, g.gold_etf)]
            cand = [(s, sc) for s, sc in cand if np.isfinite(sc) and sc > 0]
            if cand:
                cand.sort(key=lambda x: x[1], reverse=True)
                target = [cand[0][0]]
            else:
                target = [g.short_bond]
        else:
            target = [g.bond_etf, g.gold_etf]
    else:
        # 进攻层：逐只持仓过滤
        keep = []
        for sec in held:
            if sec not in g.etf_pool:
                continue
            c_now = hist_c[sec].dropna().values[-1] if sec in hist_c else np.nan
            ep = g.entry_price.get(sec, np.nan)
            if g.stop_loss > 0 and np.isfinite(ep) and np.isfinite(c_now) and (c_now / ep - 1.0 <= -g.stop_loss):
                g.last_exit_day[sec] = t
                continue
            if vol_spike.get(sec, False):
                g.last_exit_day[sec] = t
                continue
            if t - g.entry_day.get(sec, t) < g.min_hold:
                keep.append(sec)
                continue
            if not trend_ok.get(sec, False):
                g.last_exit_day[sec] = t
                continue
            keep.append(sec)

        # 候选排序（趋势多头 & 无放量 & 得分有限）
        ranked = [
            sec for sec in g.etf_pool
            if trend_ok.get(sec, False)
            and not vol_spike.get(sec, False)
            and np.isfinite(scores.get(sec, np.nan))
        ]
        ranked.sort(key=lambda s: scores[s], reverse=True)

        base_sc = min([scores.get(s, 0.0) for s in keep]) if keep else -1e9
        add = []
        for sec in ranked:
            if sec in keep or sec in add:
                continue
            if len(keep) > 0 and scores[sec] < base_sc * (1.0 + g.switch_margin):
                continue
            if t - g.last_exit_day.get(sec, -9999) <= g.cooldown:
                continue
            add.append(sec)
            if len(keep) + len(add) >= g.top_n:
                break
        target = (sorted(keep, key=lambda s: -scores.get(s, 0.0)) + add)[: g.top_n]

    # 5. 仅成员变化时触发调仓（保留原版 S9 低换手语义）
    if set(target) == set(held):
        return

    # 6. 目标波动率仓位缩放
    scaler = 1.0
    if g.target_vol > 0 and len(target) > 0:
        rets = []
        for sec in target:
            c = hist_c[sec].dropna().values
            if len(c) >= 21:
                rets.append(np.diff(np.log(c[-21:])))
        if rets:
            vol = float(np.std(np.concatenate(rets)) * np.sqrt(252.0))
            if np.isfinite(vol) and vol > 1e-6:
                scaler = float(np.clip(g.target_vol / vol, 0.0, 1.0))

    # 7. 执行调仓：先卖后买，闲置现金配置银华日利
    for sec in list(context.portfolio.positions.keys()):
        if sec not in target and sec != g.cash_etf and context.portfolio.positions[sec].amount > 0:
            order_target_value(sec, 0)
            g.entry_day.pop(sec, None)
            g.entry_price.pop(sec, None)

    total_val = context.portfolio.portfolio_value * 0.99
    if len(target) > 0:
        w_each = scaler / len(target)
        cash_w = max(0.0, 1.0 - scaler)
        order_target_value(g.cash_etf, total_val * cash_w)
        for sec in target:
            order_target_value(sec, total_val * w_each)
            if sec not in g.entry_day:
                g.entry_day[sec] = t
                g.entry_price[sec] = float(hist_c[sec].dropna().values[-1])
    else:
        order_target_value(g.cash_etf, total_val)
