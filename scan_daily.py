"""Unified Post-Close Daily Candidate Scanner for All S-Tier Strategies (Alpha Quant Hub).

Generates daily actionable next-open candidate reports:
  - reports/YYYY-MM-DD.md
  - reports/latest.md
  - reports/latest.json

Covers:
  0. Macro & Short-Term Sentiment Gate Dashboard (STRICT_MA30 / SOFT_R11 / LOOSE_R9)
  1. #1/#3 Evo-S1 V2 & V2-Cap (Evo-P2 / Evo-BF / Evo-FB next-day 09:25 trigger price windows)
  2. #5/#6 U1-Pure6Fac-Top1 & ZF2 (All candidates ranked by full-market 6-factor composite score)
  3. #4 AlphaFlow-Switch-22 (Pure-Equity Cross-Market ETF Rotation, 0 bonds 0 gold)
  4. #15 Champion-V2 Top800 (Large-Cap 5-Factor Top 15 Watchlist)
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
from time import sleep

import numpy as np
import pandas as pd

from market_data import (
    ALL_TARGET_ETFS,
    DOMESTIC_EQUITY_ETFS,
    OVERSEAS_CORE_ETFS,
    fetch_cloud_dataset,
    load_local_tdx_dataset,
)

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
SH_TZ = timezone(timedelta(hours=8))


def _limit_up(prev_close: float) -> float:
    return float(np.floor(prev_close * 1.10 * 100.0 + 0.5) / 100.0)


def _limit_down(prev_close: float) -> float:
    return float(np.floor(prev_close * 0.90 * 100.0 + 0.5) / 100.0)


def _wls_score_and_r2(close_arr: np.ndarray, L: int = 40, decay: float = 0.05, vol_penalty: float = 0.5) -> tuple[float, float]:
    arr = np.asarray(close_arr, dtype=float)
    if len(arr) < L or np.any(arr <= 0) or np.any(~np.isfinite(arr)):
        return np.nan, np.nan
    y = np.log(arr[-L:])
    j = np.arange(L, dtype=float)
    w = np.exp(decay * (j - (L - 1)))
    w /= w.sum()
    x = j - (w * j).sum()
    a2 = float((w * x * x).sum())
    y_bar = float((w * y).sum())
    dy = y - y_bar
    sxy = float((w * x * dy).sum())
    sst = float((w * dy * dy).sum())
    if a2 <= 1e-14 or sst <= 1e-14:
        return 0.0, 0.0
    slope = sxy / a2
    r2 = float(np.clip((sxy * sxy) / (a2 * sst), 0.0, 1.0))
    ann_slope = (np.exp(slope * 252.0) - 1.0)
    raw = ann_slope * r2
    rets = np.diff(np.log(arr[-21:]))
    ann_vol = float(np.std(rets, ddof=1) * np.sqrt(252.0)) if len(rets) >= 15 else 0.25
    score = raw / (max(ann_vol, 0.08) ** vol_penalty)
    return float(score), float(r2)


def compute_etf_signal(etf_bars: dict[str, pd.DataFrame], hs300: pd.Series) -> dict:
    """Exact AlphaFlow-Switch-22 pure-equity ETF rotation signal (0 bonds, 0 gold)."""
    hs_c = float(hs300.iloc[-1])
    hs_ma60 = float(hs300.iloc[-60:].mean()) if len(hs300) >= 60 else hs_c
    hs_above_ma60 = hs_c >= hs_ma60

    dom_rows = []
    ov_rows = []
    trend_hits = 0
    dom_valid = 0
    ret_map = {}

    for sym, df in etf_bars.items():
        cs = df["close"].dropna()
        vs = df["volume"].reindex(cs.index).fillna(0.0)
        if len(cs) < 62:
            continue
        arr = cs.to_numpy(dtype=float)
        c0 = float(arr[-1])
        ma20 = float(arr[-20:].mean())
        ma60 = float(arr[-60:].mean())
        trend_ok = bool(c0 > ma20 and ma20 > ma60)
        vol_ratio = float(vs.iloc[-1] / max(vs.iloc[-21:-1].mean(), 1.0))
        vol_spike = bool(vol_ratio > 2.0)
        sc, r2 = _wls_score_and_r2(arr, L=40, decay=0.05, vol_penalty=0.5)
        rets20 = np.diff(np.log(arr[-21:]))
        ann_vol = float(np.std(rets20, ddof=1) * np.sqrt(252.0)) if len(rets20) >= 15 else 0.25
        ret_map[sym] = rets20

        item = {
            "symbol": sym,
            "name": ALL_TARGET_ETFS.get(sym, sym),
            "close": round(c0, 3),
            "score": round(sc, 3) if np.isfinite(sc) else None,
            "r2": round(r2, 3) if np.isfinite(r2) else None,
            "ann_vol": round(ann_vol, 3),
            "trend_ok": trend_ok,
            "vol_spike": vol_spike,
        }
        if sym in DOMESTIC_EQUITY_ETFS:
            dom_valid += 1
            if trend_ok:
                trend_hits += 1
            if trend_ok and (not vol_spike) and np.isfinite(sc) and sc > 0:
                dom_rows.append(item)
        elif sym in OVERSEAS_CORE_ETFS:
            item["qualified"] = bool(trend_ok and np.isfinite(sc) and sc > 0 and np.isfinite(r2) and r2 >= 0.40)
            ov_rows.append(item)

    breadth = trend_hits / max(dom_valid, 1)
    dom_gate_on = bool(hs_above_ma60 or breadth >= 0.20)
    dom_rows.sort(key=lambda x: -(x["score"] or -999))
    ov_rows.sort(key=lambda x: -(x["score"] or -999))

    # Pair correlation dedup (< 0.75) for domestic top 2
    dom_picks = []
    for cand in dom_rows:
        if len(dom_picks) >= 2:
            break
        ok = True
        for prev in dom_picks:
            r1, r2_arr = ret_map.get(cand["symbol"]), ret_map.get(prev["symbol"])
            if r1 is not None and r2_arr is not None and len(r1) == len(r2_arr) and len(r1) >= 15:
                corr = float(np.corrcoef(r1, r2_arr)[0, 1])
                if np.isfinite(corr) and corr >= 0.75:
                    ok = False
                    break
        if ok:
            dom_picks.append(cand)

    targets = []
    if dom_gate_on and dom_picks:
        inv_vols = [1.0 / max(x["ann_vol"], 0.08) for x in dom_picks]
        raw_w = [v / sum(inv_vols) for v in inv_vols]
        # Portfolio 20d vol scaling to target 22%
        port_ret = sum(w * ret_map[x["symbol"]] for w, x in zip(raw_w, dom_picks))
        p_vol = float(np.std(port_ret, ddof=1) * np.sqrt(252.0))
        scale = float(np.clip(0.22 / max(p_vol, 1e-6), 0.0, 1.0))
        for x, w in zip(dom_picks, raw_w):
            targets.append({"sleeve": "国内主攻", "symbol": x["symbol"], "name": x["name"], "weight": round(w * scale, 4), "score": x["score"], "r2": x["r2"]})
        state = "状态1: 国内主攻 (海外让位)"
    else:
        ov_qual = [x for x in ov_rows if x["qualified"]]
        if ov_qual:
            top_ov = ov_qual[0]
            w = float(np.clip(0.16 / max(top_ov["ann_vol"], 0.08), 0.0, 1.0))
            targets.append({"sleeve": "海外宽基替补", "symbol": top_ov["symbol"], "name": top_ov["name"], "weight": round(w, 4), "score": top_ov["score"], "r2": top_ov["r2"]})
            state = "状态2: 国内关门 → 海外核心替补"
        else:
            state = "状态3: 国内外全弱 → 100% 纯现金空仓"

    cash_w = round(max(0.0, 1.0 - sum(x["weight"] for x in targets)), 4)
    return {
        "state": state,
        "domestic_gate_on": dom_gate_on,
        "hs300_above_ma60": hs_above_ma60,
        "domestic_breadth": round(breadth, 4),
        "targets": targets,
        "cash_weight": cash_w,
        "top_domestic": dom_rows[:6],
        "overseas_status": ov_rows,
    }


def scan_all_strategies(ds: dict) -> dict:
    """Compute all macro gates, Evo-S1 V2 candidates, U1/ZF2 6-factor rankings, ETF signals, and V2-Top800."""
    asof = ds["asof"]
    hs300 = ds["indices"]["sh000300"]
    zz1000 = ds["indices"]["sh000852"]
    stock_bars = ds["stock_bars"]
    names = ds["names"]
    ind_names = ds["ind_names"]

    # 1. Macro indices
    hs_c = float(hs300.iloc[-1])
    hs_ma20 = float(hs300.iloc[-20:].mean())
    hs_ma30 = float(hs300.iloc[-30:].mean())
    hs_ma60 = float(hs300.iloc[-60:].mean())
    hs_ok20 = hs_c >= hs_ma20
    hs_ok30 = hs_c >= hs_ma30

    zz_c = float(zz1000.iloc[-1])
    zz_ma20 = float(zz1000.iloc[-20:].mean())
    zz_lt_ma20 = zz_c < zz_ma20

    # 2. Equal-weight Main-Board trend + Limit-up/Limit-down + 3-day ZT premium + 6-factor raw values
    all_dates = hs300.index[-25:]
    ret_by_date: dict[str, list[float]] = {d: [] for d in all_dates}
    lu_cnt = 0
    ld_cnt = 0
    zt_prem_days = [[], [], []]

    raw_f6 = {}
    raw_v2 = {}
    evo_p2_cands = []
    orig_p2_cands = []
    evo_bf_cands = []
    orig_bf_cands = []
    evo_fb_cands = []
    orig_fb_cands = []

    for sym, df in stock_bars.items():
        if len(df) < 62 or df.index[-1] != asof:
            continue
        rcs = df["rc"].to_numpy(dtype=float)
        ros = df["ro"].to_numpy(dtype=float)
        rhs = df["rh"].to_numpy(dtype=float)
        rls = df["rl"].to_numpy(dtype=float)
        qcs = df["qc"].to_numpy(dtype=float)
        qhs = df["qh"].to_numpy(dtype=float)
        qls = df["ql"].to_numpy(dtype=float)
        vols = df["volume"].to_numpy(dtype=float)
        amts = df["amount"].to_numpy(dtype=float)
        tos = df["to_pct"].to_numpy(dtype=float)

        c0, c1, c2 = rcs[-1], rcs[-2], rcs[-3]
        o0, o1 = ros[-1], ros[-2]
        h0, h1 = rhs[-1], rhs[-2]
        v0, m0 = vols[-1], amts[-1]
        to0 = tos[-1]
        if c0 <= 2.0 or c1 <= 0 or v0 <= 0 or not np.isfinite(c0):
            continue

        # EW return series over last 25 days
        q_ret = pd.Series(qcs, index=df.index).pct_change().clip(-0.11, 0.11)
        for d in all_dates:
            if d in q_ret.index and np.isfinite(q_ret.loc[d]):
                ret_by_date[d].append(float(q_ret.loc[d]))

        lu0 = c0 >= _limit_up(c1) - 1e-4
        lu1 = c1 >= _limit_up(c2) - 1e-4
        ld0 = c0 <= _limit_down(c1) + 1e-4
        if lu0: lu_cnt += 1
        if ld0: ld_cnt += 1

        # 3-day ZT premium
        for lag in range(3):
            cc, cp, cp2 = rcs[-1 - lag], rcs[-2 - lag], rcs[-3 - lag]
            if cp2 > 0 and cp >= _limit_up(cp2) - 1e-4 and cp > 0:
                zt_prem_days[lag].append(float(np.clip(cc / cp - 1.0, -0.11, 0.11)))

        # 6-factor raw values (for U1 / ZF2 cross-sectional percentile)
        med60 = float(np.nanmedian(qcs[-61:-1]))
        pos_med60 = float(qcs[-2] / med60) if med60 > 0 else np.nan
        mom20 = float(qcs[-1] / qcs[-21] - 1.0) if qcs[-21] > 0 else np.nan
        to_5 = float(np.nanmean(tos[-5:])) if np.isfinite(tos[-5:]).any() else np.nan
        to_20 = float(np.nanmean(tos[-20:])) if np.isfinite(tos[-20:]).any() else np.nan
        turn_shrink = -(to_5 / to_20) if (np.isfinite(to_5) and np.isfinite(to_20) and to_20 > 1e-6) else np.nan

        r20 = np.diff(qcs[-21:]) / np.where(qcs[-21:-1] > 0, qcs[-21:-1], np.nan)
        lv20 = np.log(np.clip(vols[-20:], 1.0, None))
        ok_pv = np.isfinite(r20) & np.isfinite(lv20)
        pv_div = -float(np.corrcoef(r20[ok_pv], lv20[ok_pv])[0, 1]) if (ok_pv.sum() >= 15 and np.std(r20[ok_pv]) > 1e-9 and np.std(lv20[ok_pv]) > 1e-9) else np.nan

        vwap20 = np.where(vols[-20:] > 0, amts[-20:] / vols[-20:], np.nan)
        vrat = np.where((vwap20 > 0) & np.isfinite(vwap20), rcs[-20:] / vwap20 - 1.0, np.nan)
        vrat = np.where(np.abs(vrat) <= 0.15, vrat, np.nan)
        vwap_bias = float(np.nanmean(vrat)) if np.isfinite(vrat).sum() >= 10 else np.nan

        raw_f6[sym] = (-pos_med60, -mom20, -to0, turn_shrink, pv_div, vwap_bias)

        # Champion-V2 Top800 5-factor raw values (exclude recent 5d limit-up)
        s20 = float(np.nanmean(amts[-21:-1]))
        has_recent_zt = any(rcs[-1 - k] >= _limit_up(rcs[-2 - k]) - 1e-4 for k in range(5) if rcs[-2 - k] > 0)
        if not has_recent_zt and s20 >= 1e8 and c0 > 3.0:
            amp20 = float(np.nanmean((qhs[-20:] - qls[-20:]) / np.where(qcs[-21:-1] > 0, qcs[-21:-1], np.nan)))
            vol20 = float(np.nanstd(r20)) if np.isfinite(r20).sum() >= 15 else np.nan
            sacc = float(np.nanmean(np.where(r20 < 0, -lv20, 0.0))) if np.isfinite(r20).sum() >= 15 else np.nan
            raw_v2[sym] = (s20, -amp20, -to_20, -vol20, pv_div, sacc)

        # First-board event candidates!
        if not (lu0 and (not lu1)):
            continue
        not_ow = not (o0 >= _limit_up(c1) - 1e-4 and abs(h0 - o0) < 1e-4)
        nm = names.get(sym, sym)
        ind = ind_names.get(sym, "-")

        # P2 candidates (Evo: to [2,12]%; Orig: to [3,15]%)
        if not_ow and np.isfinite(to0) and (m0 >= 1e8) and np.isfinite(pos_med60) and (pos_med60 < 1.15):
            base_item = {
                "symbol": sym, "name": nm, "ind": ind, "close": round(c0, 2),
                "to_pct": round(to0, 2), "amt_yi": round(m0 / 1e8, 2), "pos60": round(pos_med60, 2),
                "trig_lo": round(c0 * 0.9501, 2), "trig_hi": round(c0 * 0.9850, 2),
                "gap_rule": "(-5.0%, -1.5%]",
            }
            if 2.0 <= to0 <= 12.0:
                evo_p2_cands.append(base_item)
            if 3.0 <= to0 <= 15.0:
                orig_p2_cands.append(base_item)

        # BF candidates (Evo adds pos_med60 < 1.60!)
        prev_lu_cnt = sum(1 for k in range(5) if rcs[-8 + k] > 0 and rcs[-7 + k] >= _limit_up(rcs[-8 + k]) - 1e-4)
        prev_broken = (h1 >= _limit_up(c2) - 1e-4) and (not lu1)
        bf_raw = ((c1 < o1) or prev_broken) and (prev_lu_cnt >= 1) and (c0 > max(c1, o1))
        bf_score = int(prev_lu_cnt == 1) + int(s20 >= 2.5e8) + int(o0 >= c1)
        if bf_raw and (s20 >= 2e8) and (not prev_broken) and (bf_score >= 2):
            bf_item = {
                "symbol": sym, "name": nm, "ind": ind, "close": round(c0, 2),
                "bf_score": bf_score, "s20_yi": round(s20 / 1e8, 2), "to_pct": round(to0, 2) if np.isfinite(to0) else 0.0,
                "pos60": round(pos_med60, 2) if np.isfinite(pos_med60) else 9.9,
                "trig_lo": round(c0 * 0.9001, 2), "trig_hi": round(c0 * 1.0800, 2),
                "gap_rule": "(-10%, +8.0%]",
            }
            orig_bf_cands.append(bf_item)
            if np.isfinite(pos_med60) and pos_med60 < 1.60:
                evo_bf_cands.append(bf_item)

        # FB candidates (Evo C2 strictly <= 0%, Orig C2 <= +5%)
        if (o0 < _limit_up(c1) - 0.005) and (m0 >= 1e8) and np.isfinite(pos_med60) and (pos_med60 < 1.0) and len(df) >= 120:
            is_c2 = pos_med60 < 0.70
            evo_fb_cands.append({
                "symbol": sym, "name": nm, "ind": ind, "close": round(c0, 2),
                "bucket": "C2(超跌<0.7)" if is_c2 else "C1(0.7~1.0)",
                "to_pct": round(to0, 2) if np.isfinite(to0) else 0.0, "amt_yi": round(m0 / 1e8, 2),
                "pos60": round(pos_med60, 2),
                "trig_lo": round(c0 * 0.9500, 2),
                "trig_hi": round(c0 * (1.0000 if is_c2 else 0.9800), 2),
                "gap_rule": "[-5.0%, 0.0%]" if is_c2 else "[-5.0%, -2.0%]",
            })
            orig_fb_cands.append({
                "symbol": sym, "name": nm, "ind": ind, "close": round(c0, 2),
                "bucket": "C2(超跌<0.7)" if is_c2 else "C1(0.7~1.0)",
                "to_pct": round(to0, 2) if np.isfinite(to0) else 0.0, "amt_yi": round(m0 / 1e8, 2),
                "pos60": round(pos_med60, 2),
                "trig_lo": round(c0 * (0.9750 if is_c2 else 0.9500), 2),
                "trig_hi": round(c0 * (1.0500 if is_c2 else 0.9800), 2),
                "gap_rule": "[-2.5%, +5.0%]" if is_c2 else "[-5.0%, -2.0%]",
            })

    # EW Main-Board MA20 gate
    ew_rets = [float(np.mean(ret_by_date[d])) if ret_by_date[d] else 0.0 for d in all_dates]
    ew_nav = pd.Series(np.cumprod(1.0 + np.asarray(ew_rets)), index=all_dates)
    bf_gate_on = bool(len(ew_nav) >= 20 and float(ew_nav.iloc[-1]) >= float(ew_nav.iloc[-20:].mean()))
    zt_m3 = float(np.mean([np.mean(x) if len(x) >= 3 else 0.0 for x in zt_prem_days]))

    # 6-Factor cross-sectional scores across all Main-Board stocks
    f6_df = pd.DataFrame.from_dict(raw_f6, orient="index", columns=["low_pos60", "low_mom20", "low_to1d", "turn_shrink", "pv_div_20", "vwap_bias_20"])
    score6_map = f6_df.rank(pct=True).fillna(0.5).mean(axis=1).to_dict() if not f6_df.empty else {}

    # Sort Evo candidates
    evo_p2_cands.sort(key=lambda x: (-x["amt_yi"], x["to_pct"]))
    evo_bf_cands.sort(key=lambda x: (-x["bf_score"], -x["s20_yi"]))
    evo_fb_cands.sort(key=lambda x: (0 if "C2" in x["bucket"] else 1, x["pos60"]))

    # Build U1-Pure6Fac ranked union list (from original P2/BF/FB pool, matching #5/#6 exact backtest!)
    u1_pool = {}
    for item in orig_p2_cands:
        u1_pool.setdefault(item["symbol"], {**item, "triggers": [], "gap_rules": []})
        u1_pool[item["symbol"]]["triggers"].append("P2")
        u1_pool[item["symbol"]]["gap_rules"].append(f"P2:{item['trig_lo']}~{item['trig_hi']}")
    if bf_gate_on:
        for item in orig_bf_cands:
            u1_pool.setdefault(item["symbol"], {**item, "triggers": [], "gap_rules": []})
            u1_pool[item["symbol"]]["triggers"].append("BF")
            u1_pool[item["symbol"]]["gap_rules"].append(f"BF:{item['trig_lo']}~{item['trig_hi']}")
    if zz_lt_ma20:
        for item in orig_fb_cands:
            u1_pool.setdefault(item["symbol"], {**item, "triggers": [], "gap_rules": []})
            u1_pool[item["symbol"]]["triggers"].append("FB")
            u1_pool[item["symbol"]]["gap_rules"].append(f"FB:{item['trig_lo']}~{item['trig_hi']}")

    u1_ranked = []
    for sym, info in u1_pool.items():
        u1_ranked.append({
            "symbol": sym, "name": info["name"], "ind": info["ind"], "close": info["close"],
            "score6": round(float(score6_map.get(sym, 0.5)), 4),
            "triggers": "+".join(info["triggers"]),
            "pos60": info["pos60"], "to_pct": info["to_pct"],
            "trigger_windows": " / ".join(info["gap_rules"]),
        })
    u1_ranked.sort(key=lambda x: -x["score6"])

    # Champion-V2 Top800 watchlist (Top 15)
    v2_watch = []
    if raw_v2:
        v2_df = pd.DataFrame.from_dict(raw_v2, orient="index", columns=["s20", "low_amp", "low_turn", "low_vol", "pv_div", "smart_acc"])
        v2_top800 = v2_df.sort_values("s20", ascending=False).head(800)
        v2_sc = v2_top800[["low_amp", "low_turn", "low_vol", "pv_div", "smart_acc"]].rank(pct=True).fillna(0.5).mean(axis=1)
        for sym, sc in v2_sc.sort_values(ascending=False).head(15).items():
            c0 = float(stock_bars[sym]["rc"].iloc[-1])
            v2_watch.append({
                "symbol": sym, "name": names.get(sym, sym), "ind": ind_names.get(sym, "-"),
                "close": round(c0, 2), "s20_yi": round(float(v2_top800.loc[sym, "s20"]) / 1e8, 2),
                "score": round(float(sc), 4),
            })

    # AlphaFlow-Switch-22 ETF signal
    etf_sig = compute_etf_signal(ds["etf_bars"], hs300)

    gates = {
        "hs300_close": round(hs_c, 2),
        "hs300_ma20": round(hs_ma20, 2),
        "hs300_ma30": round(hs_ma30, 2),
        "hs300_ma60": round(hs_ma60, 2),
        "hs300_ok20": hs_ok20,
        "hs300_ok30": hs_ok30,
        "zz1000_close": round(zz_c, 2),
        "zz1000_ma20": round(zz_ma20, 2),
        "zz1000_lt_ma20": zz_lt_ma20,
        "ew_main_ok20": bf_gate_on,
        "lu_cnt": lu_cnt,
        "ld_cnt": ld_cnt,
        "zt_prem_ma3_pct": round(zt_m3 * 100.0, 2),
        "gate_STRICT_MA30": "满仓 100%" if hs_ok30 else "空仓 0%",
        "gate_SOFT_R11": "满仓 100%" if hs_ok30 else ("降半仓 50%" if ld_cnt < 15 else "空仓 0% (跌停>=15)"),
        "gate_LOOSE_R9": "空仓 0% (双弱)" if ((not hs_ok20) and (zt_m3 <= 0.0)) else "满仓 100%",
    }

    return {
        "asof": asof,
        "next_day": ds["next_day"],
        "generated_at": datetime.now(SH_TZ).strftime("%Y-%m-%d %H:%M:%S CST"),
        "n_main_stocks": len(stock_bars),
        "gates": gates,
        "evo_p2": evo_p2_cands,
        "evo_bf": evo_bf_cands,
        "evo_fb": evo_fb_cands,
        "u1_ranked": u1_ranked,
        "etf_signal": etf_sig,
        "v2_top800_top15": v2_watch,
    }


def render_markdown(res: dict) -> str:
    g = res["gates"]
    e = res["etf_signal"]
    lines = [
        f"# 每日全策略盘后候选报告（{res['asof']} 收盘 → 供 {res['next_day']} 09:25 竞价执行）",
        "",
        f"> 生成时间：`{res['generated_at']}` ｜ 覆盖沪深主板有效个股：`{res['n_main_stocks']}` 只 ｜ 零未来函数",
        "",
        "---",
        "",
        "## 一、大盘与短线情绪门控仪表盘",
        "",
        "| 指标 / 门控档位 | 当前数值 | 明日指令 / 状态 |",
        "| :--- | ---: | :--- |",
        f"| 沪深300 (`sh000300`) vs MA20 / MA30 / MA60 | `{g['hs300_close']}` vs `{g['hs300_ma20']}` / `{g['hs300_ma30']}` / `{g['hs300_ma60']}` | MA20: **{'ON' if g['hs300_ok20'] else 'OFF'}** ｜ MA30: **{'ON' if g['hs300_ok30'] else 'OFF'}** |",
        f"| 主板等权 vs MA20（`BF` 内部门控） | - | **{'ON (允许BF反包)' if g['ew_main_ok20'] else 'OFF (暂停BF反包)'}** |",
        f"| 中证1000 (`sh000852`) vs MA20（`FB` 弱市补位） | `{g['zz1000_close']}` vs `{g['zz1000_ma20']}` | **{'弱市激活 FB 补位' if g['zz1000_lt_ma20'] else '非弱市 (FB关闭)'}** |",
        f"| 全主板涨停 / 跌停家数 | `{g['lu_cnt']}` 涨停 / `{g['ld_cnt']}` 跌停 | {'⚠️ 跌停潮预警 (>=15)' if g['ld_cnt']>=15 else '正常'} |",
        f"| 近 3 日昨日涨停平均溢价 (`ZTM3`) | `{g['zt_prem_ma3_pct']:+.2f}%` | {'赚钱效应为正' if g['zt_prem_ma3_pct']>0 else '⚠️ 亏钱效应'} |",
        f"| **严格门控 (`STRICT_MA30`)** | 适用于 #5 `U1+MA30` | **{g['gate_STRICT_MA30']}** |",
        f"| **柔性半仓 (`SOFT_R11`)** | 适用于 #7 `Evo-S1+R11` | **{g['gate_SOFT_R11']}** |",
        f"| **极松双弱 (`LOOSE_R9`)** | 适用于 #1 `Evo-S1+R9` | **{g['gate_LOOSE_R9']}** |",
        "",
        "---",
        "",
        "## 二、#1 / #3 旗舰：`Evo-S1 V2` & `V2-Cap` 明日 09:25 候选清单",
        "",
        "> **执行规则**：明日 09:25 集合竞价读取真实开盘价，**席位 1** 选落入 `Evo-P2` 触发区间内**低开最深（优先）的第 1 名**；**席位 2** 若 `Evo-BF` 激活且触发则选 `Evo-BF` 第 1 名，否则选 `Evo-FB` 第 1 名（自动跳过开盘涨停与同行业共振股，重合则当日只买 1 只）；T+1 日 14:55 尾盘卖出。",
        "",
        f"### 2.1 `Evo-P2` 席位1候选（首板换手 2%~12%、额≥1亿、60日位<1.15，共 `{len(res['evo_p2'])}` 只）",
        "",
    ]
    if res["evo_p2"]:
        lines += ["| 代码 | 名称 | 行业 | 昨收 | 换手% | 成交额(亿) | 60日位 | **明日 09:25 触发开盘价 `(-5%, -1.5%]`** |", "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | :---: |"]
        for x in res["evo_p2"][:25]:
            lines.append(f"| `{x['symbol']}` | **{x['name']}** | {x['ind']} | {x['close']:.2f} | {x['to_pct']:.2f}% | {x['amt_yi']:.2f} | {x['pos60']:.2f} | **{x['trig_lo']:.2f} ~ {x['trig_hi']:.2f}** |")
    else:
        lines.append("_今日无满足 `Evo-P2` 条件的盘前候选。_")

    lines += ["", f"### 2.2 `Evo-BF` 席位2主线候选（反包得分≥2、20日均额≥2亿、60日位<1.60，内部门控：**{'ON' if g['ew_main_ok20'] else 'OFF'}**，共 `{len(res['evo_bf'])}` 只）", ""]
    if res["evo_bf"]:
        lines += ["| 代码 | 名称 | 行业 | 昨收 | BF得分 | 20日均额(亿) | 60日位 | **明日 09:25 触发开盘价 `(-10%, +8%]`** |", "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | :---: |"]
        for x in res["evo_bf"][:15]:
            lines.append(f"| `{x['symbol']}` | **{x['name']}** | {x['ind']} | {x['close']:.2f} | {x['bf_score']} | {x['s20_yi']:.2f} | {x['pos60']:.2f} | **{x['trig_lo']:.2f} ~ {x['trig_hi']:.2f}** |")
    else:
        lines.append("_今日无满足 `Evo-BF` 条件的盘前候选。_")

    lines += ["", f"### 2.3 `Evo-FB` 席位2弱市补位候选（纯低开不追高，弱市门控 `ZZ1000<MA20`：**{'ON' if g['zz1000_lt_ma20'] else 'OFF'}**，共 `{len(res['evo_fb'])}` 只）", ""]
    if res["evo_fb"]:
        lines += ["| 代码 | 名称 | 行业 | 昨收 | 档位 | 60日位 | 成交额(亿) | **明日 09:25 触发开盘价** |", "| :--- | :--- | :--- | ---: | :--- | ---: | ---: | :---: |"]
        for x in res["evo_fb"][:20]:
            lines.append(f"| `{x['symbol']}` | **{x['name']}** | {x['ind']} | {x['close']:.2f} | {x['bucket']} | {x['pos60']:.2f} | {x['amt_yi']:.2f} | **{x['trig_lo']:.2f} ~ {x['trig_hi']:.2f}** (`{x['gap_rule']}`) |")
    else:
        lines.append("_今日无满足 `Evo-FB` 条件的盘前候选。_")

    lines += [
        "",
        "---",
        "",
        f"## 三、#5 / #6 旗舰：`U1-Pure6Fac-Top1` & `ZF2` 6因子统一排名清单（共 `{len(res['u1_ranked'])}` 只）",
        "",
        f"> **当前门控提示**：`STRICT_MA30` = **{g['gate_STRICT_MA30']}** ｜ `SOFT_R11` = **{g['gate_SOFT_R11']}**。若门控允许开仓，明日 09:25 从下表由上往下看，**第一只开盘价落入对应触发价区间的标的即为 `U1` 唯一第 1 名**！",
        "",
    ]
    if res["u1_ranked"]:
        lines += ["| 排名 | 代码 | 名称 | 行业 | **6因子分** | 触发池 | 昨收 | 60日位 | 换手% | 明日 09:25 触发价区间 |", "| :---: | :--- | :--- | :--- | ---: | :---: | ---: | ---: | ---: | :--- |"]
        for i, x in enumerate(res["u1_ranked"][:20], 1):
            lines.append(f"| {i} | `{x['symbol']}` | **{x['name']}** | {x['ind']} | **{x['score6']:.4f}** | `{x['triggers']}` | {x['close']:.2f} | {x['pos60']:.2f} | {x['to_pct']:.2f}% | {x['trigger_windows']} |")
    else:
        lines.append("_今日无联合池候选。_")

    lines += [
        "",
        "---",
        "",
        "## 四、#4 旗舰：`AlphaFlow-Switch-22` 纯权益 ETF 轮动（0 债 0 黄金）",
        "",
        f"- **当前状态**：**{e['state']}**",
        f"- **国内门控**：`{'ON' if e['domestic_gate_on'] else 'OFF'}`（沪深300≥MA60: `{e['hs300_above_ma60']}`，国内42只纯权益ETF多头广度: `{e['domestic_breadth']*100:.1f}%`，阈值 `20%`）",
        f"- **明日开盘目标持仓**（剩余现金权重：**`{e['cash_weight']*100:.1f}%`**）：",
        "",
    ]
    if e["targets"]:
        lines += ["| 轨道 | ETF 代码 | 名称 | **目标权重** | 风险调整动量分 | R² |", "| :--- | :--- | :--- | ---: | ---: | ---: |"]
        for t in e["targets"]:
            lines.append(f"| {t['sleeve']} | `{t['symbol']}` | **{t['name']}** | **{t['weight']*100:.1f}%** | {t['score']} | {t['r2']} |")
    else:
        lines.append("**100% 纯现金空仓避险（0 债基、0 黄金）。**")

    lines += ["", "**海外 4 核纯权益替补状态（`R² >= 0.40` 且多头趋势达标）：**", "", "| 代码 | 名称 | 现价 | 动量分 | R² | 趋势达标 | 是否合格 |", "| :--- | :--- | ---: | ---: | ---: | :---: | :---: |"]
    for o in e["overseas_status"]:
        lines.append(f"| `{o['symbol']}` | {o['name']} | {o['close']:.3f} | {o['score']} | {o['r2']} | {'YES' if o['trend_ok'] else 'NO'} | **{'✅ 合格' if o['qualified'] else '❌ 不达标'}** |")

    lines += [
        "",
        "---",
        "",
        "## 五、#15 稳健底仓：`Champion-V2 Top800` 大盘 5 因子 Top 15 观察池",
        "",
        "| 排名 | 代码 | 名称 | 行业 | 现价 | 20日均额(亿) | 5因子综合分 |",
        "| :---: | :--- | :--- | :--- | ---: | ---: | ---: |",
    ]
    for i, x in enumerate(res["v2_top800_top15"], 1):
        lines.append(f"| {i} | `{x['symbol']}` | **{x['name']}** | {x['ind']} | {x['close']:.2f} | {x['s20_yi']:.2f} | {x['score']:.4f} |")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Alpha Quant Hub Daily Candidate Scanner")
    parser.add_argument("--date", default="", help="Trading date YYYY-MM-DD (default: today CST)")
    parser.add_argument("--retry", action="store_true", help="Retry cloud fetch up to 5 times if post-close data lags")
    parser.add_argument("--skip-if-exists", action="store_true", help="Skip if reports/{date}.md already exists (for backup cron slots)")
    parser.add_argument("--local-tdx", default="", help="Optional local TDX Parquet cache dir for offline execution")
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.local_tdx:
        ds = load_local_tdx_dataset(args.local_tdx, asof=args.date or None)
    else:
        asof = args.date.strip() or datetime.now(SH_TZ).strftime("%Y-%m-%d")
        if args.skip_if_exists and (REPORTS_DIR / f"{asof}.md").exists():
            print(f"[SKIP] reports/{asof}.md already published. Exiting cleanly.")
            sys.exit(0)
        attempts = 5 if args.retry else 1
        ds = None
        for i in range(attempts):
            try:
                ds = fetch_cloud_dataset(asof)
                break
            except Exception as exc:
                print(f"[WARN] Cloud fetch attempt {i+1}/{attempts} failed: {exc}", flush=True)
                if i + 1 == attempts:
                    raise
                sleep(120)
        if not ds or not ds.get("is_trading_day"):
            print(f"[INFO] {asof} is not a trading day. Skipping report generation.")
            sys.exit(0)

    res = scan_all_strategies(ds)
    md = render_markdown(res)

    dated_md = REPORTS_DIR / f"{res['asof']}.md"
    latest_md = REPORTS_DIR / "latest.md"
    latest_json = REPORTS_DIR / "latest.json"
    dated_md.write_text(md, encoding="utf-8")
    latest_md.write_text(md, encoding="utf-8")
    latest_json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] Published {dated_md.name}, latest.md, latest.json | Evo-P2={len(res['evo_p2'])}, Evo-BF={len(res['evo_bf'])}, Evo-FB={len(res['evo_fb'])}, ETF={res['etf_signal']['state']}")


if __name__ == "__main__":
    main()
