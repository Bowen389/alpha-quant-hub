# -*- coding: utf-8 -*-
"""
AlphaFlow-Dual 纯权益 ETF 轮动策略 (PTrade 测试端 / 仿真实盘可直接运行版)
==========================================================================
严格约束：0 债基、0 黄金、0 商品，100% 纯权益 ETF 轮动 + 允许空仓。
支持两种经本地 1,678 只全市场 ETF 消融与双周期验证的新架构（顶部 MODE 一键切换）：
  1) MODE = "SWITCH"  (三态零占坑引擎):
       - 状态1 (国内门控开): 100% 资金投入国内纯权益强趋势龙头 (海外替补席立刻让位, 绝不占坑!)
       - 状态2 (国内门控关): 仅在 4 只高流动性核心海外纯权益宽基 (纳指/标普/日经/德国) 中择强
                            (须满足 close > MA20 > MA60 且 R^2 >= 0.40)
       - 状态3 (国内外全熊): 100% 空仓留现金!
  2) MODE = "CORESAT" (双轨正交组合):
       - 65% 资金独立跑国内纯权益趋势引擎 (关门即空仓)
       - 35% 资金独立跑海外核心纯权益宽基趋势 (走弱即空仓)
       - 两轨天然低相关、零占坑，无需债基黄金即可大幅压低最大回撤并穿越牛熊。
"""
import numpy as np


def initialize(context):
    # ==================== 核心配置区 ====================
    g.mode = "SWITCH"             # "SWITCH"(三态零占坑) 或 "CORESAT"(65/35双轨正交)
    g.dom_weight = 0.65           # 仅 CORESAT 模式生效：国内轨资金占比（其余为海外轨）
    g.top_n = 2                   # 国内轨最大持仓只数
    g.ov_top_n = 1                # 海外替补/海外轨最大持仓只数
    g.target_vol = 0.22           # 国内目标年化波动率: 0.15(稳健) / 0.22(进取甜点) / 0.0(全仓)
    g.ov_target_vol = 0.16        # 海外宽基目标波动率 (0.0=关闭缩放)

    # 策略核心因子与风控参数
    g.mom_lookback = 40
    g.mom_decay = 0.05
    g.vol_penalty = 0.5           # 风险调整 R^2 动量 (score / vol^0.5)
    g.ov_min_r2 = 0.40            # 海外宽基趋势质量底线 (不达标即空仓)
    g.max_pair_corr = 0.75        # 20日收益率相关性去重上限 (真分散)
    g.vol_spike_mult = 2.0        # 放量排雷倍数 (A股ETF情绪见顶保护)
    g.stop_loss = 0.10            # 收盘止损 -10%
    g.min_hold = 5                # 最短持仓交易日
    g.cooldown = 5                # 卖出冷静期交易日
    g.switch_margin = 0.15        # 换仓迟滞门槛
    g.regime_thr = 0.20           # 广度多头占比阈值 (OR 门控)

    g.bench = "000300.SS"

    # 核心海外纯权益宽基（仅限高流动性大盘宽基，杜绝小众高溢价妖基）
    g.overseas_core = [
        "513100.SS",  # 纳指ETF
        "513500.SS",  # 标普500ETF
        "513520.SS",  # 日经ETF
        "513030.SS",  # 德国ETF
    ]

    # 国内/港股高流动性纯权益 ETF 池（已彻底剔除债基/黄金/商品/货基，并按主题去重）
    g.domestic_pool = [
        # 宽基与风格
        "510300.SS", "510050.SS", "510500.SS", "512100.SS",
        "159915.SZ", "588000.SS", "159901.SZ", "510880.SS",
        # 科技与高端制造
        "512480.SS", "515880.SS", "515000.SS", "515030.SS",
        "515790.SS", "512660.SS", "512720.SS", "159819.SZ",
        "515070.SS", "512980.SS", "516160.SS", "562500.SS",
        # 金融消费医药
        "512880.SS", "512800.SS", "512070.SS", "159928.SZ",
        "512690.SS", "512010.SS", "512170.SS", "159938.SZ",
        "512200.SS", "159996.SZ",
        # 周期与公用
        "512400.SS", "515220.SS", "515210.SS", "516020.SS",
        "159611.SZ", "516150.SS", "515230.SS", "512900.SS",
        # 港股纯权益
        "513180.SS", "513050.SS", "513090.SS", "159792.SZ",
    ]

    # 运行状态
    g.day_idx = 0
    g.entry_day = {}
    g.entry_price = {}
    g.last_exit_day = {}
    g.in_backup_mode = set()
    g.breadth_hist = []

    set_benchmark(g.bench)
    set_commission(commission_ratio=0.0001, min_commission=5.0, type="ETF")
    set_fixed_slippage(fixedslippage=0.0002)


def _calc_wls_score_and_r2(close_arr, L=40, decay=0.05, vol_penalty=0.5):
    arr = np.asarray(close_arr, dtype=float)
    if len(arr) < L or np.any(arr <= 0) or np.any(~np.isfinite(arr)):
        return np.nan, np.nan
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
        return 0.0, 0.0
    slope = sxy / a2
    r2 = float(np.clip((slope * slope) * a2 / sst, 0.0, 1.0))
    mom = float(np.exp(np.clip(slope * 252.0, -5.0, 5.0)) - 1.0)
    sc = float(np.clip(mom, -1.0, 20.0) * r2)
    if vol_penalty > 0 and len(arr) >= 21:
        lr = np.diff(np.log(arr[-21:]))
        vol20 = float(np.std(lr) * np.sqrt(252.0))
        if np.isfinite(vol20) and vol20 > 0.03:
            sc = sc / (vol20 ** vol_penalty)
    return sc, r2


def _vol_scaler(hist_c, secs, target_vol):
    if target_vol <= 0 or not secs:
        return 1.0
    rets = []
    for s in secs:
        try:
            c = hist_c[s].dropna().values
            if len(c) >= 21:
                rets.append(np.diff(np.log(c[-21:])))
        except Exception:
            continue
    if not rets:
        return 1.0
    v = float(np.std(np.concatenate(rets)) * np.sqrt(252.0))
    if not np.isfinite(v) or v <= 1e-6:
        return 1.0
    return float(np.clip(target_vol / v, 0.0, 1.0))


def _corr_dedupe(ranked, hist_c, max_corr, limit_n):
    if max_corr <= 0 or len(ranked) <= 1:
        return ranked[:limit_n]
    sel = []
    ret_cache = {}
    for s in ranked:
        try:
            c = hist_c[s].dropna().values
            if len(c) < 21:
                continue
            r = np.diff(np.log(c[-21:]))
        except Exception:
            continue
        if np.std(r) < 1e-8:
            continue
        too_close = False
        for k in sel:
            rk = ret_cache[k]
            if float(np.corrcoef(r, rk)[0, 1]) >= max_corr:
                too_close = True
                break
        if not too_close:
            sel.append(s)
            ret_cache[s] = r
        if len(sel) >= limit_n:
            break
    return sel


def handle_data(context, data):
    t = g.day_idx
    g.day_idx += 1

    all_secs = list(dict.fromkeys(g.domestic_pool + g.overseas_core))
    hist_c = get_history(65, "1d", "close", security_list=all_secs, fq="pre", include=False)
    hist_a = get_history(10, "1d", "money", security_list=all_secs, fq="pre", include=False)
    bm_c = get_history(65, "1d", "close", security_list=[g.bench], fq="pre", include=False)[g.bench].dropna().values

    scores, r2_map, trend_ok, vol_spike = {}, {}, {}, {}
    dom_valid, dom_trend = 0, 0

    for sec in all_secs:
        try:
            c = hist_c[sec].dropna().values
            a = hist_a[sec].dropna().values
        except Exception:
            continue
        if len(c) < 60 or c[-1] <= 0:
            continue
        ma20 = float(np.mean(c[-20:]))
        ma60 = float(np.mean(c[-60:]))
        t_ok = bool(c[-1] > ma20 and ma20 > ma60)
        trend_ok[sec] = t_ok
        sc, r2 = _calc_wls_score_and_r2(c, g.mom_lookback, g.mom_decay, g.vol_penalty)
        scores[sec] = sc
        r2_map[sec] = r2
        if len(a) >= 6:
            base_a = float(np.mean(a[-6:-1]))
            vol_spike[sec] = bool(base_a > 0 and a[-1] > g.vol_spike_mult * base_a)
        else:
            vol_spike[sec] = False

        if sec in g.domestic_pool:
            dom_valid += 1
            if t_ok:
                dom_trend += 1

    # 国内门控 (沪深300 > MA60 OR 国内纯权益ETF广度5日均 >= 20%)
    breadth = dom_trend / max(dom_valid, 1)
    g.breadth_hist.append(breadth)
    if len(g.breadth_hist) > 5:
        g.breadth_hist = g.breadth_hist[-5:]
    idx_on = len(bm_c) >= 60 and bm_c[-1] > float(np.mean(bm_c[-60:]))
    dom_gate = bool(idx_on or (float(np.mean(g.breadth_hist)) >= g.regime_thr))

    # 选出合格海外核心宽基
    ov_cands = [
        s for s in g.overseas_core
        if trend_ok.get(s, False)
        and not vol_spike.get(s, False)
        and np.isfinite(scores.get(s, np.nan)) and scores[s] > 0
        and r2_map.get(s, 0.0) >= g.ov_min_r2
    ]
    ov_cands.sort(key=lambda s: scores[s], reverse=True)
    ov_picks = ov_cands[: g.ov_top_n]

    held = [s for s, pos in context.portfolio.positions.items() if pos.amount > 0]

    # ---------- 辅助函数：筛选国内槽位 ----------
    def select_domestic(held_dom):
        if not dom_gate:
            return []
        keep = []
        for s in held_dom:
            if s not in g.domestic_pool:
                continue
            c_now = float(hist_c[s].dropna().values[-1]) if s in hist_c else np.nan
            ep = g.entry_price.get(s, np.nan)
            if g.stop_loss > 0 and np.isfinite(ep) and np.isfinite(c_now) and (c_now / ep - 1.0 <= -g.stop_loss):
                g.last_exit_day[s] = t
                continue
            if vol_spike.get(s, False):
                g.last_exit_day[s] = t
                continue
            if t - g.entry_day.get(s, t) < g.min_hold:
                keep.append(s)
                continue
            if not trend_ok.get(s, False):
                g.last_exit_day[s] = t
                continue
            keep.append(s)

        raw_ranked = [
            s for s in g.domestic_pool
            if trend_ok.get(s, False)
            and not vol_spike.get(s, False)
            and np.isfinite(scores.get(s, np.nan)) and scores[s] > 0
        ]
        raw_ranked.sort(key=lambda s: scores[s], reverse=True)
        ranked = _corr_dedupe(raw_ranked, hist_c, g.max_pair_corr, g.top_n * 3)

        base_sc = min([scores.get(s, 0.0) for s in keep]) if keep else -1e9
        add = []
        for s in ranked:
            if s in keep or s in add:
                continue
            if len(keep) > 0 and scores[s] < base_sc * (1.0 + g.switch_margin):
                continue
            if t - g.last_exit_day.get(s, -9999) <= g.cooldown:
                continue
            add.append(s)
            if len(keep) + len(add) >= g.top_n:
                break
        return (sorted(keep, key=lambda s: -scores.get(s, 0.0)) + add)[: g.top_n]

    target_weights = {}

    if g.mode == "SWITCH":
        # 三态零占坑切换：国内门控开时，强制清空海外替补席 (绝不占坑!)
        if dom_gate:
            active_dom_held = [s for s in held if s not in g.in_backup_mode]
            dom_picks = select_domestic(active_dom_held)
            if set(dom_picks) == set(held):
                return
            sc = _vol_scaler(hist_c, dom_picks, g.target_vol)
            for s in dom_picks:
                target_weights[s] = sc / len(dom_picks)
            new_backup = set()
        else:
            if set(ov_picks) == set(held):
                return
            sc = _vol_scaler(hist_c, ov_picks, g.ov_target_vol)
            for s in ov_picks:
                target_weights[s] = sc / len(ov_picks)
            new_backup = set(ov_picks)
    else:
        # 双轨正交组合：65% 国内纯权益 + 35% 海外核心宽基，各自独立，弱势即空仓
        held_dom = [s for s in held if s in g.domestic_pool]
        held_ov = [s for s in held if s in g.overseas_core]
        dom_picks = select_domestic(held_dom)
        # 海外轨仅保留仍趋势多头或未满5日的持仓
        ov_keep = [s for s in held_ov if (t - g.entry_day.get(s, t) < g.min_hold) or trend_ok.get(s, False)]
        ov_add = [s for s in ov_picks if s not in ov_keep][: max(g.ov_top_n - len(ov_keep), 0)]
        ov_final = (ov_keep + ov_add)[: g.ov_top_n]

        if set(dom_picks + ov_final) == set(held):
            return
        if dom_picks:
            sc_d = _vol_scaler(hist_c, dom_picks, g.target_vol)
            for s in dom_picks:
                target_weights[s] = target_weights.get(s, 0.0) + g.dom_weight * sc_d / len(dom_picks)
        if ov_final:
            sc_o = _vol_scaler(hist_c, ov_final, g.ov_target_vol)
            for s in ov_final:
                target_weights[s] = target_weights.get(s, 0.0) + (1.0 - g.dom_weight) * sc_o / len(ov_final)
        new_backup = set()

    # 先卖后买（其余仓位自然保持纯现金空仓，0债基、0黄金）
    for s in list(context.portfolio.positions.keys()):
        if s not in target_weights and context.portfolio.positions[s].amount > 0:
            order_target_value(s, 0)
            g.entry_day.pop(s, None)
            g.entry_price.pop(s, None)

    total_val = context.portfolio.portfolio_value * 0.99
    for s, w in target_weights.items():
        order_target_value(s, total_val * w)
        if s not in g.entry_day:
            g.entry_day[s] = t
            g.entry_price[s] = float(hist_c[s].dropna().values[-1])

    g.in_backup_mode = new_backup
