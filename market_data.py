"""Market Data Loader for Alpha Quant Hub (Cloud GitHub Actions + Local TDX Cache).

Cloud mode (default on GitHub Actions ubuntu-latest):
  - Uses BaoStock for official SSE/SZSE trading calendar & dynamic Main-Board active stock list (0 static list!)
  - Uses Tencent `web.ifzq.gtimg.cn` / `web.sqt.gtimg.cn` (`fqkline` / `newfqkline`) for raw + qfq daily OHLCV,
    indices (`sh000300`, `sh000852`), and all 46 AlphaFlow-Switch-22 pure-equity ETFs (0 bonds, 0 gold).
Local TDX mode (`--local-tdx <cache_dir>`):
  - Reads pre-built local TDX Parquet caches (`stock_daily.parquet`, `etf_daily.parquet`, `index_daily.parquet`,
    `instruments.parquet`) in ~8 seconds with zero network dependency.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from threading import Lock
from time import monotonic, sleep
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json
import re

import numpy as np
import pandas as pd

MAIN_RE = re.compile(r"^(?:600|601|603|605|000|001|002|003)\d{3}$")
HOSTS = ("https://web.ifzq.gtimg.cn", "https://web.sqt.gtimg.cn")
_LOCK = Lock()
_NEXT_REQ = 0.0

OVERSEAS_CORE_ETFS = {
    "sh513100": "纳指ETF",
    "sh513500": "标普500ETF",
    "sh513520": "日经ETF",
    "sh513030": "德国ETF",
}

DOMESTIC_EQUITY_ETFS = {
    "sh510300": "沪深300ETF", "sh510050": "上证50ETF", "sh510500": "中证500ETF", "sh512100": "中证1000ETF",
    "sz159915": "创业板ETF", "sh588000": "科创50ETF", "sz159901": "深证100ETF", "sh510880": "红利ETF",
    "sh512480": "半导体ETF", "sh515880": "通信ETF", "sh515000": "科技ETF", "sh515030": "新能车ETF",
    "sh515790": "光伏ETF", "sh512660": "军工ETF", "sh512720": "计算机ETF", "sz159819": "人工智能ETF",
    "sh515070": "AI智能ETF", "sh512980": "传媒ETF", "sh516160": "新能源ETF", "sh562500": "机器人ETF",
    "sh512880": "证券ETF", "sh512800": "银行ETF", "sh512070": "非银ETF", "sz159928": "消费ETF",
    "sh512690": "酒ETF", "sh512010": "医药ETF", "sh512170": "医疗ETF", "sz159938": "医药卫生ETF",
    "sh512200": "房地产ETF", "sz159996": "家电ETF",
    "sh512400": "有色金属ETF", "sh515220": "煤炭ETF", "sh515210": "钢铁ETF", "sh516020": "化工ETF",
    "sz159611": "电力ETF", "sh516150": "稀土ETF", "sh515230": "软件ETF", "sh512900": "证券基金ETF",
    "sh513180": "恒生科技ETF", "sh513050": "中概互联ETF", "sh513090": "香港证券ETF", "sz159792": "港股通互联网ETF",
}

ALL_TARGET_ETFS = {**DOMESTIC_EQUITY_ETFS, **OVERSEAS_CORE_ETFS}


def _fetch_tencent(endpoint: str, params: str) -> dict:
    global _NEXT_REQ
    for attempt in range(2):
        with _LOCK:
            sleep(max(0.0, _NEXT_REQ - monotonic()))
            _NEXT_REQ = monotonic() + 0.08
        try:
            url = f"{HOSTS[attempt]}/appstock/app/{endpoint}/get?" + urlencode({"param": params})
            with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=10) as resp:
                res = json.load(resp)
            if not res.get("data"):
                raise ValueError("empty response")
            return res
        except (OSError, ValueError):
            if attempt == 1:
                raise
    raise RuntimeError("unreachable")


def query_exchange_calendar_and_stocks(asof: str) -> tuple[str | None, dict[str, str] | None, dict[str, str]]:
    """Query BaoStock for trading day status, next trading date, active Main-Board non-ST list, and industry tags.

    If BaoStock has not published `query_all_stock(day=asof)` yet (common between 16:15 and 17:45 BJT),
    automatically falls back to the most recent prior trading day's stock roster while Tencent fetches today's bars.
    """
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {lg.error_msg}")
    try:
        beg_d = (date.fromisoformat(asof) - timedelta(days=15)).isoformat()
        end_d = (date.fromisoformat(asof) + timedelta(days=20)).isoformat()
        rs = bs.query_trade_dates(start_date=beg_d, end_date=end_d)
        if rs.error_code != "0":
            raise RuntimeError(rs.error_msg)
        cal = rs.get_data()
        row = cal[cal["calendar_date"] == asof]
        if row.empty:
            raise RuntimeError(f"Calendar missing {asof}")
        if row.iloc[0]["is_trading_day"] != "1":
            return None, None, {}
        fut = cal[(cal["calendar_date"] > asof) & (cal["is_trading_day"] == "1")]
        next_day = str(fut.iloc[0]["calendar_date"]) if not fut.empty else "下一交易日"

        # Try `asof` first, then up to 5 previous trading days if today's BaoStock roster is not published yet
        past_tdays = list(cal[(cal["calendar_date"] <= asof) & (cal["is_trading_day"] == "1")]["calendar_date"].sort_values(ascending=False))[:6]
        df = pd.DataFrame()
        used_day = asof
        for cand_day in past_tdays:
            st = bs.query_all_stock(day=str(cand_day))
            if st.error_code != "0":
                continue
            recs = []
            while st.next():
                recs.append(st.get_row_data())
            if not recs:
                continue
            tmp = pd.DataFrame.from_records(recs, columns=st.fields)
            tmp["code"] = tmp["code"].str.split(".").str[-1]
            tmp = tmp[tmp["code"].str.fullmatch(MAIN_RE) & (tmp["tradeStatus"] == "1")]
            if len(tmp) >= 2500:
                df = tmp
                used_day = str(cand_day)
                break
        if len(df) < 2500:
            raise RuntimeError(f"Abnormal Main-Board stock count across {past_tdays}: {len(df)}")
        if used_day != asof:
            print(f"[CLOUD] BaoStock roster for {asof} not published yet -> seamlessly fell back to {used_day} ({len(df)} stocks)", flush=True)

        names = {}
        for _, r in df.iterrows():
            nm = str(r["code_name"])
            if "ST" in nm.upper() or "退" in nm:
                continue
            code = str(r["code"])
            sym = ("sh" if code.startswith("6") else "sz") + code
            names[sym] = nm

        # Best-effort Chinese industry lookup (never blocks daily scan if unavailable)
        ind_names: dict[str, str] = {}
        try:
            rs_ind = bs.query_stock_industry()
            if rs_ind.error_code == "0":
                irecs = []
                while rs_ind.next():
                    irecs.append(rs_ind.get_row_data())
                if irecs:
                    idf = pd.DataFrame.from_records(irecs, columns=rs_ind.fields)
                    for _, r in idf.iterrows():
                        raw_c = str(r.get("code", ""))
                        ind_v = str(r.get("industry", "")).strip()
                        if "." in raw_c and ind_v:
                            mkt, num = raw_c.split(".", 1)
                            # Strip leading alphanumeric classification codes like "C39 " if present
                            ind_clean = re.sub(r"^[A-Z]\d+\s*", "", ind_v).strip() or ind_v
                            ind_names[mkt.lower() + num] = ind_clean
        except Exception as exc:
            print(f"[WARN] BaoStock industry lookup skipped: {exc}", flush=True)

        return next_day, names, ind_names
    finally:
        bs.logout()


def _parse_tencent_bars(rows: list, cols: list[str], idxs: list[int]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=cols)
    arr = [[r[i] if i < len(r) else None for i in idxs] for r in rows]
    df = pd.DataFrame(arr, columns=["date", *cols]).drop_duplicates("date", keep="last").set_index("date")
    return df.apply(pd.to_numeric, errors="coerce").sort_index()


def fetch_cloud_dataset(asof: str, lookback_bars: int = 135) -> dict:
    """Fetch all required cloud data (indices, Main-Board stocks, 46 ETFs) for `asof`."""
    next_day, names, ind_names = query_exchange_calendar_and_stocks(asof)
    if next_day is None or names is None:
        return {"is_trading_day": False, "asof": asof}

    # 1. Indices (sh000300, sh000852)
    indices = {}
    for sym in ("sh000300", "sh000852"):
        res = _fetch_tencent("fqkline", f"{sym},day,,{asof},160,qfq")["data"][sym]
        rows = res.get("day") or res.get("qfqday") or []
        s = pd.Series({r[0]: float(r[2]) for r in rows if r[0] <= asof}, dtype=float).sort_index()
        if asof not in s.index:
            raise RuntimeError(f"Index {sym} not updated to {asof} yet (latest={s.index[-1] if len(s) else None})")
        indices[sym] = s

    # 2. 46 Pure-Equity ETFs (AlphaFlow-Switch-22)
    etf_bars = {}
    def _fetch_etf(sym: str):
        res = _fetch_tencent("fqkline", f"{sym},day,,{asof},150,qfq")["data"].get(sym, {})
        rows = res.get("qfqday") or res.get("day") or []
        df = _parse_tencent_bars([r for r in rows if r[0] <= asof], ["open", "close", "high", "low", "volume"], [0, 1, 2, 3, 4, 5])
        return sym, df

    with ThreadPoolExecutor(max_workers=16) as ex:
        for fut in as_completed([ex.submit(_fetch_etf, s) for s in ALL_TARGET_ETFS]):
            try:
                sym, df = fut.result()
                if len(df) >= 60:
                    etf_bars[sym] = df
            except Exception:
                pass

    # 3. Dynamic Main-Board stocks (raw + qfq)
    since = (date.fromisoformat(asof) - timedelta(days=230)).isoformat()
    stock_bars = {}
    errors = 0

    def _fetch_stock(sym: str):
        p_raw = _fetch_tencent("newfqkline", f"{sym},day,{since},{asof},{lookback_bars},")["data"][sym]
        p_qfq = _fetch_tencent("newfqkline", f"{sym},day,{since},{asof},{lookback_bars},qfq")["data"][sym]
        r_raw = [r for r in (p_raw.get("day") or []) if r[0] <= asof and len(r) >= 9]
        r_qfq = [r for r in (p_qfq.get("qfqday") or p_qfq.get("day") or []) if r[0] <= asof and len(r) >= 6]
        if not r_raw or r_raw[-1][0] != asof or not r_qfq or r_qfq[-1][0] != asof:
            raise ValueError("missing asof bar")
        raw_df = _parse_tencent_bars(r_raw, ["ro", "rc", "rh", "rl", "vol", "to_pct", "amt_wan"], [0, 1, 2, 3, 4, 5, 7, 8])
        qfq_df = _parse_tencent_bars(r_qfq, ["qo", "qc", "qh", "ql"], [0, 1, 2, 3, 4])
        df = raw_df.join(qfq_df, how="inner")
        df["amount"] = df["amt_wan"] * 1e4
        # Tencent vol is in lots (100 shares); convert to shares so amount/vol matches raw price!
        df["volume"] = df["vol"] * 100.0
        return sym, df[["ro", "rc", "rh", "rl", "qo", "qc", "qh", "ql", "volume", "amount", "to_pct"]]

    with ThreadPoolExecutor(max_workers=24) as ex:
        futs = {ex.submit(_fetch_stock, s): s for s in names}
        for fut in as_completed(futs):
            try:
                sym, df = fut.result()
                stock_bars[sym] = df
            except Exception:
                errors += 1

    cov = len(stock_bars) / max(len(names), 1)
    print(f"[CLOUD] Main-Board coverage: {len(stock_bars)}/{len(names)} ({cov*100:.1f}%), ETFs: {len(etf_bars)}/{len(ALL_TARGET_ETFS)}", flush=True)
    if cov < 0.90:
        raise RuntimeError(f"Cloud coverage too low: {len(stock_bars)}/{len(names)} (errors={errors})")

    return {
        "is_trading_day": True,
        "asof": asof,
        "next_day": next_day,
        "indices": indices,
        "etf_bars": etf_bars,
        "stock_bars": stock_bars,
        "names": names,
        "ind_names": ind_names,
    }


def load_local_tdx_dataset(cache_dir: str | Path, asof: str | None = None) -> dict:
    """Offline loader from local TDX Parquet cache (zero network, completes in ~8 seconds)."""
    cdir = Path(cache_dir)
    inst = pd.read_parquet(cdir / "instruments.parquet")
    idx_df = pd.read_parquet(cdir / "index_daily.parquet")
    etf_df = pd.read_parquet(cdir / "etf_daily.parquet")
    stk_df = pd.read_parquet(cdir / "stock_daily.parquet")

    idx_df["date_str"] = pd.to_datetime(idx_df["date"].astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")
    if asof is None:
        asof = str(idx_df[idx_df["symbol"] == "sh000300"]["date_str"].max())
    start_cut = (date.fromisoformat(asof) - timedelta(days=240)).strftime("%Y%m%d")
    end_cut = asof.replace("-", "")

    # 1. Indices
    indices = {}
    for sym in ("sh000300", "sh000852"):
        sub = idx_df[(idx_df["symbol"] == sym) & (idx_df["date_str"] <= asof)].sort_values("date_str").tail(160)
        indices[sym] = pd.Series(sub["close"].to_numpy(dtype=float), index=sub["date_str"])

    # 2. ETFs
    etf_sub = etf_df[(etf_df["symbol"].isin(ALL_TARGET_ETFS.keys())) & (etf_df["date"].astype(str) >= start_cut) & (etf_df["date"].astype(str) <= end_cut)].copy()
    etf_sub["date_str"] = pd.to_datetime(etf_sub["date"].astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")
    etf_bars = {}
    for sym, grp in etf_sub.groupby("symbol"):
        g = grp.sort_values("date_str").drop_duplicates("date_str").set_index("date_str")
        # Use adj_close/adj_open if present
        c_col = "adj_close" if "adj_close" in g.columns else "close"
        o_col = "adj_open" if "adj_open" in g.columns else "open"
        h_col = "adj_high" if "adj_high" in g.columns else "high"
        l_col = "adj_low" if "adj_low" in g.columns else "low"
        etf_bars[str(sym)] = pd.DataFrame({
            "open": g[o_col].astype(float), "close": g[c_col].astype(float),
            "high": g[h_col].astype(float), "low": g[l_col].astype(float),
            "volume": g["volume"].astype(float),
        })

    # 3. Main-Board non-ST stocks
    inst_main = inst[inst["symbol"].str.match(r"^(?:sh600|sh601|sh603|sh605|sz000|sz001|sz002|sz003)\d{3}$")].copy()
    inst_main = inst_main[~inst_main["name"].str.upper().str.contains("ST|退", na=False)]
    names = dict(zip(inst_main["symbol"].astype(str), inst_main["name"].astype(str)))
    ind_names = dict(zip(inst_main["symbol"].astype(str), inst_main["ind_name"].astype(str)))

    stk_sub = stk_df[(stk_df["symbol"].isin(names.keys())) & (stk_df["date"].astype(str) >= start_cut) & (stk_df["date"].astype(str) <= end_cut)].copy()
    stk_sub["date_str"] = pd.to_datetime(stk_sub["date"].astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")
    # Convert fraction turnover (0..1) -> percentage (0..100) to match Tencent API `to_pct`
    if "turnover" in stk_sub.columns:
        stk_sub["turnover_pct"] = stk_sub["turnover"].astype(float) * 100.0
    elif "turnover_pct" not in stk_sub.columns:
        stk_sub["turnover_pct"] = np.nan

    stock_bars = {}
    for sym, grp in stk_sub.groupby("symbol"):
        g = grp.sort_values("date_str").drop_duplicates("date_str").set_index("date_str")
        if len(g) < 62 or g.index[-1] != asof:
            continue
        stock_bars[str(sym)] = pd.DataFrame({
            "ro": g["open"].astype(float), "rc": g["close"].astype(float),
            "rh": g["high"].astype(float), "rl": g["low"].astype(float),
            "qo": g["adj_open"].astype(float), "qc": g["adj_close"].astype(float),
            "qh": g["adj_high"].astype(float), "ql": g["adj_low"].astype(float),
            "volume": g["volume"].astype(float), "amount": g["amount"].astype(float),
            "to_pct": g["turnover_pct"].astype(float),
        })

    print(f"[LOCAL-TDX] Loaded asof={asof} | Main-Board={len(stock_bars)} stocks, ETFs={len(etf_bars)}/{len(ALL_TARGET_ETFS)}")
    return {
        "is_trading_day": True,
        "asof": asof,
        "next_day": "下一交易日",
        "indices": indices,
        "etf_bars": etf_bars,
        "stock_bars": stock_bars,
        "names": names,
        "ind_names": ind_names,
    }
