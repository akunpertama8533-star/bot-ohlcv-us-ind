"""Bot Telegram OHLCV (yfinance) — v3: timeframe 5m + validasi chunk & anomali + Web Server Keep-Alive (Flask).

Menjalankan:
    pip install pandas numpy yfinance pyTelegramBotAPI requests flask
    python bot_ohlcv.py
"""
import html
import io
import os
import re
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import telebot
import yfinance as yf
from flask import Flask
from requests import exceptions as req_exc

# ====================== KONFIGURASI BOT & AKSES ======================
# Masukkan token bot Telegram Anda langsung di sini (didapat dari @BotFather)
TELEGRAM_BOT_TOKEN = "8833515035:AAGKrxeesIBgt_B0TbbV4UvZDLcpojLb29M"

# Masukkan daftar ID chat Telegram yang diizinkan menggunakan bot ini.
# Contoh: {123456789, 987654321} 
# Jika dikosongkan ({}), maka siapa saja dapat mengakses bot ini.
ALLOWED_CHAT_IDS = {5292651856}
# ====================================================================

# ====================== KONFIGURASI WEB SERVER (KEEP ALIVE) ======================
app = Flask("")


@app.route("/")
def home():
    return "Bot OHLCV Web Service is Running!"


def run_web():
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)


def keep_alive():
    t = threading.Thread(target=run_web)
    t.daemon = True
    t.start()


# Bot diinisialisasi secara global
bot = None

SUPPORTED_INTERVALS = {
    "5m": ("5m", None),
    "15m": ("15m", None),
    "30m": ("30m", None),
    "45m": ("15m", "45min"),
    "1h": ("1h", None),
    "60m": ("1h", None),
    "2h": ("1h", "2h"),
    "3h": ("1h", "3h"),
    "4h": ("1h", "4h"),
    "1d": ("1d", None),
    "1day": ("1d", None),
    "1w": ("1wk", None),
    "1wk": ("1wk", None),
    "weekly": ("1wk", None),
    "1mo": ("1mo", None),
    "1month": ("1mo", None),
    "monthly": ("1mo", None),
}

CHUNK_DAYS = {
    "5m": 50,
    "15m": 50,
    "30m": 50,
    "1h": 700,
}

INTRADAY_LIMIT_DAYS = {
    "5m": 59,
    "15m": 59,
    "30m": 59,
    "1h": 729,
}

RANGE_SHORTCUTS = {
    "7d": 7,
    "1w": 7,
    "14d": 14,
    "2w": 14,
    "30d": 30,
    "1mo": 30,
    "90d": 90,
    "3mo": 90,
    "180d": 180,
    "6mo": 180,
    "1y": 365,
    "2y": 730,
    "5y": 1825,
    "ytd": "ytd",
    "max": "max",
}

TICKER_RE = re.compile(r"^[A-Z0-9.\-=^]{1,20}$")
NETWORK_ERRORS = (req_exc.ConnectionError, req_exc.ReadTimeout, req_exc.Timeout, OSError)
MAX_RETRIES = 3
TELEGRAM_TEXT_LIMIT = 4096
TELEGRAM_CAPTION_LIMIT = 1000

OHLC = ["Open", "High", "Low", "Close"]
COLS = OHLC + ["Volume"]
INTRADAY_INTERVALS = {"5m", "15m", "30m", "1h"}

CHUNK_OVERLAP_DAYS = 1
CHUNK_MAX_ATTEMPTS = 3
COVERAGE_TOL_DAYS = 5
MIN_EXPECTED_BDAYS = 3
MIN_BDAYS_DENSITY = 10
DENSITY_MIN_RATIO = 0.6
MIN_TEMPLATE_DAYS = 3
TEMPLATE_PRESENCE_RATIO = 0.5
MISSING_RUN_WARN = 5
MAX_REPAIR_WINDOWS = 6
OHLC_REL_TOL = 1e-4
SPIKE_Z = 10.0
SPIKE_THRESHOLD = {
    "5m": 0.08, "15m": 0.10, "30m": 0.12, "1h": 0.15,
    "1d": 0.30, "1wk": 0.50, "1mo": 0.80,
}
LEVEL_ORDER = {"ERROR": 0, "WARN": 1, "INFO": 2}

_busy_chats = set()
_busy_lock = threading.Lock()


@dataclass
class Finding:
    level: str
    text: str


@dataclass
class ChunkReport:
    index: int
    start: datetime
    end: datetime
    eff_end: datetime
    rows: int = 0
    attempts: int = 0
    notes: list = field(default_factory=list)
    issues: list = field(default_factory=list)
    error: Optional[str] = None
    level: str = "OK"


@dataclass
class FetchResult:
    df: pd.DataFrame
    failed: list = field(default_factory=list)
    reports: list = field(default_factory=list)
    boundary_dates: set = field(default_factory=set)
    findings: list = field(default_factory=list)


@dataclass
class GapDay:
    day: date
    missing: int
    expected: int
    kind: str
    near: bool
    sample: list


@dataclass
class Analysis:
    findings: list
    targets: list
    completeness: Optional[float] = None
    boundary_problem: bool = False


@dataclass
class Outcome:
    verdict: str
    summary: str
    text: str


def is_date(text):
    try:
        datetime.strptime(text, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def detect_timezone(ticker: str) -> str:
    if ticker.upper().endswith(".JK"):
        return "Asia/Jakarta"
    return "America/New_York"


def today_in_exchange_tz(ticker: str) -> datetime:
    try:
        now = pd.Timestamp.now(tz=detect_timezone(ticker))
        return now.tz_localize(None).normalize().to_pydatetime()
    except Exception as e:
        print(f"[TZ WARNING] {e}")
        return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)


def apply_timezone(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df.empty:
        return df
    tz = detect_timezone(ticker)
    try:
        if df.index.tz is None:
            if (df.index == df.index.normalize()).all():
                df.index = df.index.tz_localize(tz)
            else:
                df.index = df.index.tz_localize("UTC").tz_convert(tz)
        else:
            df.index = df.index.tz_convert(tz)
    except Exception as e:
        print(f"[TZ WARNING] {e}")
    return df


def resample_per_day(df, rule):
    def _agg(day_df):
        return day_df.resample(rule, origin="start").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        })
    out = df.groupby(df.index.date, group_keys=False).apply(_agg)
    return out.dropna(subset=["Open"])


def is_transient_error(e: Exception) -> bool:
    if isinstance(e, NETWORK_ERRORS):
        return True
    name = type(e).__name__
    return "RateLimit" in name or type(e).__module__.startswith("curl_cffi")


def fetch_history_with_retry(ticker, **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return yf.Ticker(ticker).history(**kwargs)
        except Exception as e:
            if not is_transient_error(e) or attempt == MAX_RETRIES:
                raise
            wait = 2 * attempt * (3 if "RateLimit" in type(e).__name__ else 1)
            print(f"[RETRY {attempt}/{MAX_RETRIES}] {ticker}: {e} (tunggu {wait}s)")
            time.sleep(wait)


def create_progress_bar(percentage: int, length: int = 16) -> str:
    filled = int(length * percentage / 100)
    bar = "█" * filled + "░" * (length - filled)
    return f"[{bar}] {percentage}%"


def plan_chunks(start_dt: datetime, end_dt: datetime, max_days: int):
    chunks = []
    cur = start_dt
    while cur <= end_dt:
        chunk_end = min(cur + timedelta(days=max_days), end_dt)
        chunks.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    return chunks


def clamp_to_provider_limit(yf_interval, start_dt, end_dt, today):
    limit = INTRADAY_LIMIT_DAYS.get(yf_interval)
    if limit is None:
        return start_dt, False, False
    earliest = today - timedelta(days=limit)
    if end_dt < earliest:
        return start_dt, False, True
    if start_dt < earliest:
        return earliest, True, False
    return start_dt, False, False


def _safe_edit(message, progress_msg, text):
    if not progress_msg:
        return
    try:
        bot.edit_message_text(
            chat_id=message.chat.id,
            message_id=progress_msg.message_id,
            text=text,
            parse_mode="HTML",
        )
    except Exception:
        pass


def _bdays(a: datetime, b: datetime) -> int:
    return max(0, int(np.busday_count(a.date(), (b + timedelta(days=1)).date())))


def _fmt_ts(ts, intraday: bool) -> str:
    return ts.strftime("%Y-%m-%d %H:%M" if intraday else "%Y-%m-%d")


def _examples(index, mask, intraday: bool, n: int = 3) -> str:
    mask = np.asarray(mask, dtype=bool)
    return ", ".join(_fmt_ts(t, intraday) for t in index[mask][:n])


def _mm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _more(items, shown: int) -> str:
    return f" … (+{len(items) - shown} lagi)" if len(items) > shown else ""


def clean_frame(df: pd.DataFrame):
    findings = []
    if df.empty:
        return df, findings
    n0 = len(df)
    df = df[~df[OHLC].isna().all(axis=1)]
    if len(df) < n0:
        findings.append(Finding("INFO", f"{n0 - len(df)} baris kosong total dibuang"))
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
        findings.append(Finding("INFO", "Data diurutkan ulang"))
    dup = df.index.duplicated(keep="first")
    if dup.any():
        findings.append(Finding("INFO", f"{int(dup.sum())} duplikat dibuang"))
        df = df[~dup]
    return df, findings


def check_integrity(df: pd.DataFrame, intraday: bool):
    out = []
    if df.empty:
        return out
    idx = df.index
    o, h, l, c, v = (df[k] for k in COLS)
    price = df[OHLC]

    m = price.isna().any(axis=1).to_numpy()
    if m.any():
        out.append(Finding("ERROR", f"{int(m.sum())} baris OHLC kosong/NaN"))

    m = (price <= 0).any(axis=1).to_numpy()
    if m.any():
        out.append(Finding("WARN", f"{int(m.sum())} baris harga <= 0"))

    eps = OHLC_REL_TOL * price.abs().max(axis=1)
    oc = pd.concat([o, c], axis=1)
    bad = ((h < l - eps) | (h < oc.max(axis=1) - eps) | (l > oc.min(axis=1) + eps)).to_numpy()
    if bad.any():
        out.append(Finding("ERROR", f"{int(bad.sum())} bar OHLC tidak konsisten"))

    m = (v < 0).to_numpy()
    if m.any():
        out.append(Finding("ERROR", f"{int(m.sum())} bar volume negatif"))

    m = v.isna().to_numpy()
    if m.any():
        out.append(Finding("WARN", f"{int(m.sum())} bar volume kosong/NaN"))

    zero = (v == 0).to_numpy()
    if zero.any():
        out.append(Finding("INFO", f"{int(zero.sum())} bar volume 0"))
    return out


def check_spikes(df: pd.DataFrame, interval: str):
    out = []
    if len(df) < 3:
        return out
    intraday = interval in INTRADAY_INTERVALS
    thr = SPIKE_THRESHOLD.get(interval, 0.30)
    close = df["Close"].astype(float)
    if intraday:
        prev = close.groupby(np.asarray(df.index.date)).shift(1)
    else:
        prev = close.shift(1)
    ret = (close / prev - 1.0).replace([np.inf, -np.inf], np.nan)
    valid = ret.dropna()
    if not valid.empty:
        med = float(valid.median())
        mad = float((valid - med).abs().median())
        scale = max(1.4826 * mad, 1e-9)
        z = (ret - med).abs() / scale
        m = ((ret.abs() > thr) & (z > SPIKE_Z)).to_numpy()
        if m.any():
            out.append(Finding("WARN", f"{int(m.sum())} bar loncat > {thr * 100:.0f}%"))
    return out


def check_coverage(df: pd.DataFrame, interval: str, req_start, req_end):
    out, targets = [], []
    if req_start is None or req_end is None or df.empty:
        return out, targets
    tol = {"1wk": 8, "1mo": 35}.get(interval, COVERAGE_TOL_DAYS)
    first_d, last_d = df.index.min().date(), df.index.max().date()
    lead = (first_d - req_start.date()).days
    trail = (req_end.date() - last_d).days
    if lead > tol:
        out.append(Finding("WARN", f"Data mulai terlambat {lead} hari"))
        targets.append((2, req_start.date(), first_d))
    if trail > tol:
        out.append(Finding("WARN", f"Data berakhir lebih awal {trail} hari"))
        targets.append((0, last_d, req_end.date()))
    return out, targets


def check_period_gaps(df: pd.DataFrame, interval: str):
    out = []
    if len(df) < 2:
        return out
    limit = 10 if interval == "1wk" else 40
    prev, nxt = df.index[:-1], df.index[1:]
    days = np.asarray((nxt - prev).days)
    bad = np.where(days > limit)[0]
    if len(bad):
        out.append(Finding("WARN", f"{len(bad)} celah candle > {limit} hari"))
    return out


def find_absent_runs(df: pd.DataFrame):
    dates = pd.DatetimeIndex(df.index.tz_localize(None).normalize()).unique()
    if len(dates) < 2:
        return []
    present = set(dates)
    allb = pd.bdate_range(dates.min(), dates.max())
    pos = {d: i for i, d in enumerate(allb)}
    runs = []
    for d in allb:
        if d in present:
            continue
        if runs and pos[d] == pos[runs[-1][-1]] + 1:
            runs[-1].append(d)
        else:
            runs.append([d])
    return runs


def analyze_missing_sessions(df: pd.DataFrame, boundary_dates):
    findings, targets = [], []
    runs = find_absent_runs(df)
    if not runs:
        return findings, targets, False
    total = sum(len(r) for r in runs)
    findings.append(Finding("INFO", f"{total} hari kerja tanpa data"))
    warns, boundary_problem = [], False
    for r in runs:
        n = len(r)
        touch = any(d.date() in boundary_dates for d in r)
        label = f"{r[0]:%Y-%m-%d} → {r[-1]:%Y-%m-%d} ({n} hari)"
        if n >= MISSING_RUN_WARN or (touch and n >= 3):
            warns.append(label)
        if touch and n >= 3:
            boundary_problem = True
        if touch or n >= 3:
            s = (r[0] - timedelta(days=1)).date()
            e = (r[-1] + timedelta(days=1)).date()
            targets.append((0 if touch else 2, s, e))
    return findings, targets, boundary_problem


def analyze_intraday_gaps(df: pd.DataFrame, boundary_dates):
    findings, targets = [], []
    idx = df.index
    frame = pd.DataFrame({"d": np.asarray(idx.date), "m": np.asarray(idx.hour * 60 + idx.minute)})
    day_sets = {d: frozenset(g["m"].tolist()) for d, g in frame.groupby("d")}
    days = sorted(day_sets)

    templates = {}
    for wd in range(7):
        ds = [d for d in days if d.weekday() == wd]
        if len(ds) >= MIN_TEMPLATE_DAYS:
            cnt = Counter(m for d in ds for m in day_sets[d])
            templates[wd] = {m for m, c in cnt.items() if c / len(ds) >= TEMPLATE_PRESENCE_RATIO}
    if not templates:
        return findings, targets, None, False

    gap_days, exp_total, miss_total = [], 0, 0
    for d in days:
        tpl = templates.get(d.weekday())
        if not tpl:
            continue
        present = day_sets[d]
        missing = sorted(tpl - present)
        exp_total += len(tpl)
        miss_total += len(missing)
        if not missing:
            continue
        present_tpl = [m for m in sorted(tpl) if m in present]
        kind = "seluruh sesi" if not present_tpl else "sebagian"
        gap_days.append(GapDay(d, len(missing), len(tpl), kind, d in boundary_dates, [_mm(m) for m in missing[:4]]))

    completeness = (1 - miss_total / exp_total) if exp_total else None
    if completeness is not None:
        findings.append(Finding("INFO", f"Kelengkapan bar terhadap pola sesi: {completeness * 100:.2f}%"))

    near = [g for g in gap_days if g.near]
    sig = [g for g in gap_days if not g.near and g.kind != "akhir" and (g.missing >= 3 or g.missing / g.expected >= 0.10)]
    
    for g in near:
        targets.append((0, g.day - timedelta(days=1), g.day + timedelta(days=1)))
    for g in sig:
        targets.append((1, g.day - timedelta(days=1), g.day + timedelta(days=1)))
    return findings, targets, completeness, bool(near)


def analyze_dataframe(df, interval, req_start=None, req_end=None, boundary_dates=frozenset()):
    if df is None or df.empty:
        return Analysis([Finding("ERROR", "Tidak ada data")], [])
    intraday = interval in INTRADAY_INTERVALS
    findings, targets = [], []
    completeness, boundary_problem = None, False

    findings += check_integrity(df, intraday)
    findings += check_spikes(df, interval)
    cf, ct = check_coverage(df, interval, req_start, req_end)
    findings += cf
    targets += ct

    if intraday:
        gf, gt, completeness, bp = analyze_intraday_gaps(df, boundary_dates)
        findings += gf
        targets += gt
        boundary_problem = boundary_problem or bp
    if intraday or interval == "1d":
        sf, st, bp = analyze_missing_sessions(df, boundary_dates)
        findings += sf
        targets += st
        boundary_problem = boundary_problem or bp
    elif interval in ("1wk", "1mo"):
        findings += check_period_gaps(df, interval)

    findings.sort(key=lambda f: LEVEL_ORDER[f.level])
    return Analysis(findings, targets, completeness, boundary_problem)


def check_resample_consistency(base: pd.DataFrame, res: pd.DataFrame, label: str):
    if base.empty or res.empty:
        return []
    vb, vr = float(base["Volume"].sum()), float(res["Volume"].sum())
    ok_vol = abs(vb - vr) <= 1e-6 * max(1.0, abs(vb))
    ok_hl = (np.isclose(base["High"].max(), res["High"].max()) and np.isclose(base["Low"].min(), res["Low"].min()))
    if ok_vol and ok_hl:
        return [Finding("INFO", f"Resample {label}: konsisten")]
    return [Finding("ERROR", f"Resample {label}: volume atau High/Low tidak cocok")]


def assess_chunk(df_chunk, c_start: datetime, c_end: datetime, is_first: bool, is_last: bool):
    expected = _bdays(c_start, c_end)
    if df_chunk is None or df_chunk.empty:
        if expected >= MIN_EXPECTED_BDAYS:
            return True, [f"kosong padahal ada {expected} hari kerja"]
        return False, []
    notes, retry = [], False
    first_d, last_d = df_chunk.index.min().date(), df_chunk.index.max().date()
    lead = (first_d - c_start.date()).days
    trail = (c_end.date() - last_d).days
    if lead > COVERAGE_TOL_DAYS:
        notes.append("data mulai terlambat")
        retry = retry or not is_first
    if trail > COVERAGE_TOL_DAYS:
        notes.append("data berakhir lebih cepat")
        retry = retry or not is_last
    return retry, notes


def classify_chunks(reports):
    out = []
    total = len(reports)
    nonempty = [r.index for r in reports if r.rows > 0]
    for r in reports:
        exp = _bdays(r.start, r.eff_end)
        if r.rows == 0:
            if r.error:
                r.level = "ERROR"
                out.append(Finding("ERROR", f"Chunk {r.index}/{total} gagal: {r.error}"))
            elif exp >= MIN_EXPECTED_BDAYS:
                if not nonempty or r.index < min(nonempty):
                    out.append(Finding("INFO", f"Chunk {r.index}/{total} kosong (sebelum listing)"))
                elif r.index > max(nonempty):
                    r.level = "WARN"
                    out.append(Finding("WARN", f"Chunk {r.index}/{total} kosong (setelah data ada)"))
                else:
                    r.level = "ERROR"
                    out.append(Finding("ERROR", f"Chunk {r.index}/{total} kosong di tengah"))
            continue
        if r.notes:
            r.level = "WARN"
            out.append(Finding("WARN", f"Chunk {r.index}/{total} ada catatan"))
        if r.issues:
            r.level = "ERROR"
    return out


def fetch_long_history(ticker, start_date, end_date, yf_interval, message=None, today=None):
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    max_days = CHUNK_DAYS.get(yf_interval)
    plan = plan_chunks(start_dt, end_dt, max_days) if max_days else [(start_dt, end_dt)]
    total = len(plan)
    safe_ticker = html.escape(ticker)

    progress_msg = None
    if message and total > 1:
        try:
            progress_msg = bot.send_message(
                message.chat.id,
                f"⏳ Mengambil data <b>{safe_ticker}</b> ({yf_interval})\n{create_progress_bar(0)}",
                parse_mode="HTML",
            )
        except Exception:
            progress_msg = None

    chunks, failed, reports = [], [], []
    for i, (c_start, c_end) in enumerate(plan, start=1):
        eff_end = min(c_end, today) if today else c_end
        rep = ChunkReport(index=i, start=c_start, end=c_end, eff_end=eff_end)
        reports.append(rep)

        req_start = c_start - timedelta(days=CHUNK_OVERLAP_DAYS) if i > 1 else c_start
        start_str = req_start.strftime("%Y-%m-%d")
        end_label = c_end.strftime("%Y-%m-%d")
        end_str = (c_end + timedelta(days=1)).strftime("%Y-%m-%d")
        percentage = int((i - 1) / total * 100)

        best, best_notes, last_notes, last_err = None, [], [], None
        for attempt in range(1, CHUNK_MAX_ATTEMPTS + 1):
            rep.attempts = attempt
            _safe_edit(
                message, progress_msg,
                f"⏳ Mengambil data <b>{safe_ticker}</b> ({yf_interval})\n{create_progress_bar(percentage)}\nChunk {i}/{total}",
            )
            try:
                df_chunk = fetch_history_with_retry(ticker, start=start_str, end=end_str, interval=yf_interval)
                last_err = None
            except Exception as e:
                df_chunk, last_err = None, e
                if not is_transient_error(e):
                    break

            if last_err is None:
                retry, last_notes = assess_chunk(df_chunk, c_start, eff_end, i == 1, i == total)
                if df_chunk is not None and not df_chunk.empty and (best is None or len(df_chunk) > len(best)):
                    best, best_notes = df_chunk, last_notes
                if not retry:
                    break
            if attempt < CHUNK_MAX_ATTEMPTS:
                time.sleep(2 * attempt)

        if best is not None and not best.empty:
            chunks.append(best)
            rep.rows = len(best)
            rep.notes = best_notes
        else:
            rep.notes = last_notes
            if last_err is not None:
                rep.error = str(last_err)
                failed.append(end_label)

        if i < total:
            time.sleep(1.1)

    findings = classify_chunks(reports)
    _safe_edit(
        message, progress_msg,
        f"✅ Selesai mengambil data <b>{safe_ticker}</b> ({yf_interval})\n{create_progress_bar(100)}",
    )

    boundary = set()
    for k in range(total - 1):
        boundary.add(plan[k][1].date())
        boundary.add(plan[k + 1][0].date())

    if not chunks:
        return FetchResult(pd.DataFrame(), failed, reports, boundary, findings)

    df = pd.concat(chunks)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return FetchResult(df, failed, reports, boundary, findings)


def merge_windows(items):
    out = []
    for s, e in sorted(items):
        if out and s <= out[-1][1] + timedelta(days=1):
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def select_repair_windows(targets, lo: Optional[date], hi: Optional[date]):
    selected = []
    for _, s, e in sorted(targets, key=lambda t: (t[0], t[1])):
        if lo and s < lo:
            s = lo
        if hi and e > hi:
            e = hi
        if s > e:
            continue
        cand = merge_windows(selected + [(s, e)])
        if len(cand) <= MAX_REPAIR_WINDOWS:
            selected = cand
    return selected


def repair_gaps(ticker, df, yf_interval, windows, req_start: datetime, req_end: datetime):
    added_total = 0
    lo, hi = pd.Timestamp(req_start.date()), pd.Timestamp(req_end.date())
    for ws, we in windows:
        try:
            new = fetch_history_with_retry(
                ticker,
                start=ws.strftime("%Y-%m-%d"),
                end=(we + timedelta(days=1)).strftime("%Y-%m-%d"),
                interval=yf_interval,
            )
        except Exception:
            new = None
        if new is not None and not new.empty:
            new = apply_timezone(new[COLS].copy(), ticker)
            new, _ = clean_frame(new)
            day = pd.DatetimeIndex(new.index.tz_localize(None).normalize())
            new = new[(day >= lo) & (day <= hi)]
            new = new[~new.index.isin(df.index)]
            if len(new):
                df = pd.concat([df, new]).sort_index()
                added_total += len(new)
    return df, added_total


def validate_and_repair(ticker, df, yf_interval, req_start, req_end, fetch: FetchResult, pre_findings):
    boundary = fetch.boundary_dates
    analysis = analyze_dataframe(df, yf_interval, req_start, req_end, boundary)
    extra = []

    if req_start is not None and req_end is not None and analysis.targets:
        windows = select_repair_windows(analysis.targets, req_start.date(), req_end.date())
        if windows:
            df2, added = repair_gaps(ticker, df, yf_interval, windows, req_start, req_end)
            if added:
                df = df2
                analysis = analyze_dataframe(df, yf_interval, req_start, req_end, boundary)
                extra.append(Finding("INFO", f"Pemulihan otomatis: {added} bar ditambahkan"))

    findings = list(pre_findings) + list(fetch.findings) + extra + analysis.findings
    return df, findings, analysis.completeness


def build_outcome(ticker, interval_input, yf_interval, range_info, tz_name,
                  rows, reports, findings, completeness) -> Outcome:
    findings = sorted(findings, key=lambda f: LEVEL_ORDER[f.level])
    counts = {lvl: sum(1 for f in findings if f.level == lvl) for lvl in LEVEL_ORDER}
    verdict = "ERROR" if counts["ERROR"] else ("WARN" if counts["WARN"] else "OK")

    comp = f", kelengkapan {completeness * 100:.1f}%" if completeness is not None else ""
    summary = f"✅ Validasi: VALID{comp}" if verdict == "OK" else f"⚠️ Validasi: {counts['WARN']} peringatan"

    lines = [
        "LAPORAN VALIDASI DATA OHLCV",
        "=" * 27,
        f"Kode        : {ticker}",
        f"Timeframe   : {interval_input}",
        f"Rentang     : {range_info}",
        f"Total baris : {rows:,}",
        "",
    ]
    lines.append("TEMUAN")
    if findings:
        lines += [f"[{f.level}] {f.text}" for f in findings]
    else:
        lines.append("(tidak ada)")
    return Outcome(verdict, summary, "\n".join(lines))


def build_csv_filename(ticker, interval_input, df, start_date=None, end_date=None):
    if not (start_date and end_date) and not df.empty:
        start_date = df.index.min().strftime("%Y-%m-%d")
        end_date = df.index.max().strftime("%Y-%m-%d")
    if start_date and end_date:
        return f"{ticker}_{interval_input}_{start_date}_to_{end_date}.csv"
    return f"{ticker}_{interval_input}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


def _make_caption(ticker, info, rows):
    return f"📁 <b>{html.escape(ticker)}</b>\n{html.escape(info)}\nJumlah baris: {rows}"


def send_validation_report(message, ticker, interval_input, df, outcome, start_date, end_date):
    name = build_csv_filename(ticker, interval_input, df, start_date, end_date).replace(".csv", "_validasi.txt")
    buf = io.BytesIO(outcome.text.encode("utf-8"))
    buf.name = name
    bot.send_document(message.chat.id, buf, caption=html.escape(outcome.summary), parse_mode="HTML")


def send_result(message, ticker, df, info, want_csv, interval_input=None, start_date=None, end_date=None,
                outcome=None, want_report=False):
    if want_csv:
        filename = build_csv_filename(ticker, interval_input, df, start_date, end_date)
        buf = io.BytesIO(df.to_csv().encode("utf-8"))
        buf.name = filename
        bot.send_document(message.chat.id, buf, caption=_make_caption(ticker, info, len(df)), parse_mode="HTML")
    else:
        rows = 25
        while True:
            df_show = df.tail(rows)
            body = html.escape(df_show.to_string())
            text = f"📈 <b>{html.escape(ticker)}</b>\n{html.escape(info)}\n<pre>{body}</pre>"
            if len(text) <= TELEGRAM_TEXT_LIMIT or rows <= 1:
                break
            rows -= 1
        bot.reply_to(message, text, parse_mode="HTML")

    if outcome is not None and (want_report or outcome.verdict != "OK"):
        send_validation_report(message, ticker, interval_input, df, outcome, start_date, end_date)


def fail(message, text):
    bot.reply_to(message, text)


def authorized(message) -> bool:
    return not ALLOWED_CHAT_IDS or message.chat.id in ALLOWED_CHAT_IDS


def parse_command(text):
    raw = (text or "").split()
    head, tail = raw[:2], raw[2:]
    known = {"csv", "report"}
    flags = {a.lower() for a in tail if a.lower() in known}
    args = head + [a for a in tail if a.lower() not in known]
    return args, flags


def welcome(message):
    text = "📊 <b>Bot OHLCV + CSV Aktif</b>\nGunakan perintah <code>/ohlcv KODE TIMEFRAME RANGE</code>"
    bot.reply_to(message, text, parse_mode="HTML")


def get_ohlcv(message):
    if not authorized(message):
        fail(message, "⛔ Anda tidak memiliki akses ke bot ini.")
        return

    chat_id = message.chat.id
    with _busy_lock:
        if chat_id in _busy_chats:
            fail(message, "⏳ Permintaan sebelumnya masih diproses.")
            return
        _busy_chats.add(chat_id)
    try:
        _handle_ohlcv(message)
    finally:
        with _busy_lock:
            _busy_chats.discard(chat_id)


def _handle_ohlcv(message):
    try:
        args, flags = parse_command(message.text)
        want_csv = "csv" in flags
        want_report = "report" in flags

        if len(args) < 3:
            fail(message, "Format salah. Contoh: /ohlcv BBCA.JK 5m 30d")
            return

        ticker = args[1].upper()
        if not TICKER_RE.match(ticker):
            fail(message, f"❌ Kode saham '{ticker}' tidak valid.")
            return

        interval_input = args[2].lower()
        if interval_input not in SUPPORTED_INTERVALS:
            fail(message, f"❌ Timeframe '{interval_input}' tidak didukung.")
            return

        yf_interval, resample_rule = SUPPORTED_INTERVALS[interval_input]
        tz_name = detect_timezone(ticker)
        today = today_in_exchange_tz(ticker)

        notes = []
        use_period_max = False
        label_dates = True

        if len(args) >= 5 and is_date(args[3]) and is_date(args[4]):
            start_dt = datetime.strptime(args[3], "%Y-%m-%d")
            end_dt = datetime.strptime(args[4], "%Y-%m-%d")
            range_info = f"{args[3]} → {args[4]}"
        elif len(args) >= 4:
            range_input = args[3].lower()
            if range_input not in RANGE_SHORTCUTS:
                fail(message, f"❌ Range '{range_input}' tidak dikenali.")
                return
            value = RANGE_SHORTCUTS[range_input]
            end_dt = today
            if value == "ytd":
                start_dt = datetime(today.year, 1, 1)
                label_dates = False
                range_info = range_input.upper()
            elif value == "max":
                label_dates = False
                range_info = range_input.upper()
                limit = INTRADAY_LIMIT_DAYS.get(yf_interval)
                start_dt = None if limit is None else today - timedelta(days=limit)
                use_period_max = (limit is None)
            else:
                start_dt = end_dt - timedelta(days=value)
                range_info = f"{range_input}"
        else:
            fail(message, "Format salah.")
            return

        if use_period_max:
            df_raw = fetch_history_with_retry(ticker, period="max", interval=yf_interval)
            fetch = FetchResult(df=df_raw if df_raw is not None else pd.DataFrame())
            start_date = end_date = v_start = v_end = None
        else:
            start_dt, clamped, out_of_range = clamp_to_provider_limit(yf_interval, start_dt, end_dt, today)
            if out_of_range:
                fail(message, "❌ Rentang di luar batas Yahoo Finance.")
                return
            start_date = start_dt.strftime("%Y-%m-%d")
            end_date = end_dt.strftime("%Y-%m-%d")
            fetch = fetch_long_history(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                yf_interval=yf_interval,
                message=message,
                today=today,
            )
            v_start, v_end = start_dt, min(end_dt, today)

        df = fetch.df
        if df is None or df.empty:
            fail(message, f"❌ Tidak ada data untuk {ticker}.")
            return

        df = df[COLS].copy()
        df = apply_timezone(df, ticker)
        df, clean_findings = clean_frame(df)
        if df.empty:
            fail(message, f"❌ Data kosong setelah dibersihkan.")
            return

        df, findings, completeness = validate_and_repair(
            ticker, df, yf_interval, v_start, v_end, fetch, clean_findings
        )

        if resample_rule:
            base_df = df
            df = resample_per_day(base_df, resample_rule)
            findings += check_resample_consistency(base_df, df, interval_input)
        df = df.round(2)

        outcome = build_outcome(
            ticker, interval_input, yf_interval, range_info, tz_name,
            len(df), fetch.reports, findings, completeness,
        )

        info = f"{interval_input} | {range_info} | TZ: {tz_name}"
        send_result(
            message, ticker, df, info, want_csv,
            interval_input=interval_input,
            start_date=start_date if label_dates else None,
            end_date=end_date if label_dates else None,
            outcome=outcome,
            want_report=want_report,
        )

    except Exception as e:
        fail(message, f"⚠️ Error: {e}")


def main():
    global bot
    token = TELEGRAM_BOT_TOKEN
    
    # Cek sederhana: jika token kosong atau masih berupa placeholder, hentikan program
    if not token or "MASUKKAN_TOKEN" in token:
        print("❌ TELEGRAM_BOT_TOKEN belum diisi dengan benar di dalam skrip.")
        sys.exit(1)

    keep_alive()

    bot = telebot.TeleBot(token, parse_mode=None)
    bot.message_handler(commands=["start", "help"])(welcome)
    bot.message_handler(commands=["ohlcv"])(get_ohlcv)

    print("✅ Bot OHLCV (Web Service + Flask Keep-Alive) sedang berjalan...")

    while True:
        try:
            bot.infinity_polling(skip_pending=True, timeout=60, long_polling_timeout=60)
        except Exception as e:
            print(f"[POLLING ERROR] {e}")
            time.sleep(10)
