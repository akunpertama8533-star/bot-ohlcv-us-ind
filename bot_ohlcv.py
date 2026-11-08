"""Bot Telegram OHLCV (yfinance) — v3: timeframe 5m + validasi chunk & anomali.

Baru di versi ini
  1. Timeframe 5m (batas Yahoo ±60 hari; di-chunk per 50 hari seperti 15m/30m).
  2. Validasi tiap chunk saat diambil (kosong? bolong di tepi? struktur OHLC rusak?)
     dan validasi hasil gabungan (celah bar, hari hilang, lonjakan harga, overlap).
  3. Pemulihan otomatis: chunk kosong/bolong diambil ulang; celah di batas chunk
     ditambal lewat jendela refetch. Celah yang tetap ada sesudah refetch dilaporkan
     sebagai celah dari sumber (Yahoo), bukan akibat proses chunk.
  4. Chunk saling tumpang-tindih 1 hari; bar kembar dibandingkan untuk memastikan
     sambungan antar-chunk konsisten.
  5. Laporan validasi (.txt) dikirim bila ada peringatan/masalah, atau selalu
     dengan flag `report`.

Menjalankan:
    pip install pandas numpy yfinance pyTelegramBotAPI requests
    set TELEGRAM_BOT_TOKEN=<token dari BotFather>        (Windows CMD)
    export TELEGRAM_BOT_TOKEN=<token dari BotFather>     (Linux/macOS)
    # opsional: batasi pengguna (pisahkan dengan koma)
    set ALLOWED_CHAT_IDS=123456789,987654321
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
from requests import exceptions as req_exc

# ====================== KONFIGURASI ======================
# Bot dibuat di main() agar modul bisa di-import/diuji tanpa token.
bot = None

# Opsional: daftar chat id yang boleh memakai bot. Kosong = semua orang boleh.
ALLOWED_CHAT_IDS = set()
for _part in os.getenv("ALLOWED_CHAT_IDS", "").split(","):
    _part = _part.strip()
    if _part.lstrip("-").isdigit():
        ALLOWED_CHAT_IDS.add(int(_part))

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

# Batas riwayat intraday Yahoo Finance (hari ke belakang dari hari ini).
# Nilai sengaja diberi margin 1 hari dari batas yang saya ketahui
# (60 hari untuk <1h, 730 hari untuk 1h). Perlu diverifikasi dengan data asli.
INTRADAY_LIMIT_DAYS = {
    "5m": 59,
    "15m": 59,
    "30m": 59,
    "1h": 729,
}

# Shortcut range → jumlah hari ("ytd"/"max" ditangani khusus)
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
TELEGRAM_CAPTION_LIMIT = 1000  # batas asli 1024; sisakan margin

# ---------- Parameter validasi ----------
OHLC = ["Open", "High", "Low", "Close"]
COLS = OHLC + ["Volume"]
INTRADAY_INTERVALS = {"5m", "15m", "30m", "1h"}

CHUNK_OVERLAP_DAYS = 1        # chunk ke-2 dst. mulai 1 hari lebih awal (tumpang-tindih)
CHUNK_MAX_ATTEMPTS = 3        # percobaan ulang bila chunk kosong/bolong di tepi
COVERAGE_TOL_DAYS = 5         # toleransi akhir pekan + libur sebelum dianggap bolong
MIN_EXPECTED_BDAYS = 3        # jendela < 3 hari kerja: kosong dianggap wajar
MIN_BDAYS_DENSITY = 10        # cek kepadatan bar/chunk hanya untuk chunk >= 10 hari kerja
DENSITY_MIN_RATIO = 0.6       # chunk < 60% dari median kepadatan chunk lain = curiga
MIN_TEMPLATE_DAYS = 3         # sampel minimum per hari-dalam-pekan untuk menyusun pola sesi
TEMPLATE_PRESENCE_RATIO = 0.5 # jam bar dianggap "normal" bila ada di >= 50% hari
MISSING_RUN_WARN = 5          # >= 5 hari kerja beruntun tanpa data = peringatan
MAX_REPAIR_WINDOWS = 6        # batas jumlah jendela refetch per permintaan
OHLC_REL_TOL = 1e-4          # toleransi relatif cek High/Low terhadap Open/Close
SPIKE_Z = 10.0                # lonjakan juga harus > 10x simpangan robust (MAD)
SPIKE_THRESHOLD = {           # lonjakan return minimum yang dicurigai
    "5m": 0.08, "15m": 0.10, "30m": 0.12, "1h": 0.15,
    "1d": 0.30, "1wk": 0.50, "1mo": 0.80,
}
LEVEL_ORDER = {"ERROR": 0, "WARN": 1, "INFO": 2}

_busy_chats = set()
_busy_lock = threading.Lock()


# ====================== STRUKTUR DATA VALIDASI ======================
@dataclass
class Finding:
    level: str  # "ERROR" | "WARN" | "INFO"
    text: str


@dataclass
class ChunkReport:
    index: int
    start: datetime
    end: datetime
    eff_end: datetime                 # end dibatasi hari ini (rentang masa depan tak dihitung)
    rows: int = 0
    attempts: int = 0
    notes: list = field(default_factory=list)    # catatan cakupan (tepi bolong, dll.)
    issues: list = field(default_factory=list)   # masalah struktur data di dalam chunk
    error: Optional[str] = None
    level: str = "OK"                 # OK | WARN | ERROR (diisi classify_chunks)


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
    targets: list                      # (prioritas, tgl_awal, tgl_akhir) kandidat refetch
    completeness: Optional[float] = None
    boundary_problem: bool = False


@dataclass
class Outcome:
    verdict: str   # OK | WARN | ERROR
    summary: str
    text: str


# ====================== UTILITAS ======================
def is_date(text):
    try:
        datetime.strptime(text, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def detect_timezone(ticker: str) -> str:
    if ticker.upper().endswith(".JK"):
        return "Asia/Jakarta"
    return "America/New_York"  # nama kanonik; alias "US/Eastern" tidak ada di semua sistem


def today_in_exchange_tz(ticker: str) -> datetime:
    """Tanggal 'hari ini' (naive, jam 00:00) menurut zona waktu bursa, bukan zona server."""
    try:
        now = pd.Timestamp.now(tz=detect_timezone(ticker))
        return now.tz_localize(None).normalize().to_pydatetime()
    except Exception as e:  # mis. database zona waktu tidak ada -> pakai tanggal server
        print(f"[TZ WARNING] {e}")
        return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)


def apply_timezone(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df.empty:
        return df
    tz = detect_timezone(ticker)
    try:
        if df.index.tz is None:
            # Indeks naive berisi tanggal saja (jam 00:00) = tanggal lokal bursa,
            # jangan digeser lewat UTC. Selain itu, asumsikan UTC seperti semula.
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
    """Error yang pantas dicoba ulang: jaringan, rate limit, atau error curl_cffi."""
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
    """Daftar (awal, akhir) inklusif; tiap potongan mencakup max_days+1 hari."""
    chunks = []
    cur = start_dt
    while cur <= end_dt:
        chunk_end = min(cur + timedelta(days=max_days), end_dt)
        chunks.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    return chunks


def clamp_to_provider_limit(yf_interval, start_dt, end_dt, today):
    """Return (start_baru, di_clamp, kosong). kosong=True jika seluruh range di luar batas."""
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


# ====================== VALIDASI DATA: PEMBANTU ======================
def _bdays(a: datetime, b: datetime) -> int:
    """Jumlah hari kerja (Sen–Jum) pada [a, b] inklusif; 0 bila b < a."""
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
    """Buang baris kosong total, urutkan, hapus timestamp duplikat. Return (df, findings)."""
    findings = []
    if df.empty:
        return df, findings
    n0 = len(df)
    df = df[~df[OHLC].isna().all(axis=1)]
    if len(df) < n0:
        findings.append(Finding("INFO", f"{n0 - len(df)} baris kosong total (OHLC semua NaN) dibuang"))
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
        findings.append(Finding("INFO", "Urutan waktu tidak monoton; data diurutkan ulang"))
    dup = df.index.duplicated(keep="first")
    if dup.any():
        findings.append(Finding("INFO", f"{int(dup.sum())} timestamp duplikat dibuang"))
        df = df[~dup]
    return df, findings


def check_integrity(df: pd.DataFrame, intraday: bool):
    """Cek struktur per-bar: NaN, harga <= 0, OHLC tidak konsisten, volume negatif/kosong."""
    out = []
    if df.empty:
        return out
    idx = df.index
    o, h, l, c, v = (df[k] for k in COLS)
    price = df[OHLC]

    m = price.isna().any(axis=1).to_numpy()
    if m.any():
        out.append(Finding("ERROR", f"{int(m.sum())} baris punya OHLC kosong/NaN (mis. {_examples(idx, m, intraday)})"))

    m = (price <= 0).any(axis=1).to_numpy()
    if m.any():
        out.append(Finding("WARN", f"{int(m.sum())} baris punya harga <= 0 (mis. {_examples(idx, m, intraday)})"))

    eps = OHLC_REL_TOL * price.abs().max(axis=1)
    oc = pd.concat([o, c], axis=1)
    bad = ((h < l - eps) | (h < oc.max(axis=1) - eps) | (l > oc.min(axis=1) + eps)).to_numpy()
    if bad.any():
        out.append(Finding(
            "ERROR",
            f"{int(bad.sum())} bar OHLC tidak konsisten (High < Low/Open/Close atau Low > Open/Close), "
            f"mis. {_examples(idx, bad, intraday)}"))

    m = (v < 0).to_numpy()
    if m.any():
        out.append(Finding("ERROR", f"{int(m.sum())} bar bervolume negatif (mis. {_examples(idx, m, intraday)})"))

    m = v.isna().to_numpy()
    if m.any():
        out.append(Finding("WARN", f"{int(m.sum())} bar volumenya kosong/NaN (mis. {_examples(idx, m, intraday)})"))

    zero = (v == 0).to_numpy()
    if zero.any():
        out.append(Finding(
            "INFO", f"{int(zero.sum())} bar bervolume 0 ({zero.mean() * 100:.1f}% dari {len(df)} bar; "
                    "umumnya tidak ada transaksi)"))
    return out


def check_spikes(df: pd.DataFrame, interval: str):
    """Lonjakan harga antar-bar yang tidak wajar (tick salah / data rusak). Hanya peringatan."""
    out = []
    if len(df) < 3:
        return out
    intraday = interval in INTRADAY_INTERVALS
    thr = SPIKE_THRESHOLD.get(interval, 0.30)
    close = df["Close"].astype(float)
    if intraday:  # jangan hitung gap semalam sebagai lonjakan
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
            out.append(Finding(
                "WARN",
                f"{int(m.sum())} bar loncat > {thr * 100:.0f}% dari bar sebelumnya dan jauh di luar pola normal "
                f"(mis. {_examples(df.index, m, intraday)}) — periksa kemungkinan tick salah/aksi korporasi"))
    low = df["Low"].where(df["Low"] > 0)
    rng = ((df["High"] - df["Low"]) / low).replace([np.inf, -np.inf], np.nan)
    m = (rng > 2 * thr).to_numpy()
    if m.any():
        out.append(Finding(
            "WARN",
            f"{int(m.sum())} bar dengan rentang High–Low > {2 * thr * 100:.0f}% "
            f"(mis. {_examples(df.index, m, intraday)}) — kemungkinan wick/tick anomali"))
    return out


def check_coverage(df: pd.DataFrame, interval: str, req_start, req_end):
    """Apakah data mencakup rentang yang diminta (di luar libur/akhir pekan).

    Return (findings, targets). Target refetch dibuat untuk ujung yang bolong, mis. chunk
    terakhir yang gagal, karena celah di ujung tidak terlihat oleh pencarian hari-hilang.
    """
    out, targets = [], []
    if req_start is None or req_end is None or df.empty:
        return out, targets
    tol = {"1wk": 8, "1mo": 35}.get(interval, COVERAGE_TOL_DAYS)
    first_d, last_d = df.index.min().date(), df.index.max().date()
    lead = (first_d - req_start.date()).days
    trail = (req_end.date() - last_d).days
    if lead > tol:
        out.append(Finding(
            "WARN", f"Data baru mulai {first_d}, {lead} hari setelah awal rentang ({req_start.date()}) — "
                    "saham baru listing, atau data awal tidak tersedia di Yahoo"))
        targets.append((2, req_start.date(), first_d))
    if trail > tol:
        out.append(Finding(
            "WARN", f"Data berakhir {last_d}, {trail} hari sebelum akhir rentang ({req_end.date()}) — "
                    "data terbaru belum tersedia atau terpotong"))
        targets.append((0, last_d, req_end.date()))
    return out, targets


def check_period_gaps(df: pd.DataFrame, interval: str):
    """Candle mingguan/bulanan yang hilang (selisih antar-candle terlalu besar)."""
    out = []
    if len(df) < 2:
        return out
    limit = 10 if interval == "1wk" else 40
    prev, nxt = df.index[:-1], df.index[1:]
    days = np.asarray((nxt - prev).days)
    bad = np.where(days > limit)[0]
    if len(bad):
        ex = "; ".join(f"{prev[i]:%Y-%m-%d} → {nxt[i]:%Y-%m-%d}" for i in bad[:3])
        out.append(Finding(
            "WARN", f"{len(bad)} celah antar-candle > {limit} hari (candle hilang?): {ex}{_more(bad, 3)}"))
    return out


def find_absent_runs(df: pd.DataFrame):
    """Deretan hari kerja (Sen–Jum) yang tidak punya satu pun bar. Return list of list[Timestamp]."""
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
    findings.append(Finding("INFO", f"{total} hari kerja tanpa data sama sekali (umumnya libur bursa)"))
    warns, boundary_problem = [], False
    for r in runs:
        n = len(r)
        touch = any(d.date() in boundary_dates for d in r)
        label = f"{r[0]:%Y-%m-%d} → {r[-1]:%Y-%m-%d} ({n} hari kerja{', di batas chunk' if touch else ''})"
        if n >= MISSING_RUN_WARN or (touch and n >= 3):
            warns.append(label)
        if touch and n >= 3:
            boundary_problem = True
        if touch or n >= 3:
            s = (r[0] - timedelta(days=1)).date()
            e = (r[-1] + timedelta(days=1)).date()
            targets.append((0 if touch else 2, s, e))
    if warns:
        findings.append(Finding(
            "WARN", "Celah hari kerja tanpa data (libur panjang atau data hilang): "
                    + "; ".join(warns[:5]) + _more(warns, 5)))
    return findings, targets, boundary_problem


def analyze_intraday_gaps(df: pd.DataFrame, boundary_dates):
    """Deteksi bar yang hilang di dalam hari berdasarkan pola sesi hasil belajar dari data.

    Pola tiap hari-dalam-pekan = jam bar yang muncul di >= 50% hari bersangkutan, sehingga
    jam istirahat siang, sesi Jumat, dsb. otomatis dianggap normal tanpa jam bursa hardcode.
    Return (findings, targets, completeness, boundary_problem).
    """
    findings, targets = [], []
    idx = df.index
    frame = pd.DataFrame({"d": np.asarray(idx.date), "m": np.asarray(idx.hour * 60 + idx.minute)})
    day_sets = {d: frozenset(g["m"].tolist()) for d, g in frame.groupby("d")}
    days = sorted(day_sets)

    templates, skipped = {}, 0
    for wd in range(7):
        ds = [d for d in days if d.weekday() == wd]
        if len(ds) >= MIN_TEMPLATE_DAYS:
            cnt = Counter(m for d in ds for m in day_sets[d])
            templates[wd] = {m for m, c in cnt.items() if c / len(ds) >= TEMPLATE_PRESENCE_RATIO}
        else:
            skipped += len(ds)
    if not templates:
        findings.append(Finding(
            "INFO", f"Data terlalu pendek untuk analisis kelengkapan bar per sesi "
                    f"(perlu >= {MIN_TEMPLATE_DAYS} hari pada hari-dalam-pekan yang sama)"))
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
        if not present_tpl:
            kind = "seluruh sesi"
        else:
            fp, lp = present_tpl[0], present_tpl[-1]
            parts = []
            if any(m < fp for m in missing):
                parts.append("awal")
            if any(fp < m < lp for m in missing):
                parts.append("tengah")
            if any(m > lp for m in missing):
                parts.append("akhir")
            kind = "+".join(parts)
        gap_days.append(GapDay(d, len(missing), len(tpl), kind, d in boundary_dates,
                               [_mm(m) for m in missing[:4]]))

    completeness = (1 - miss_total / exp_total) if exp_total else None
    if completeness is not None:
        lvl = "WARN" if completeness < 0.90 else "INFO"
        findings.append(Finding(
            lvl, f"Kelengkapan bar terhadap pola sesi: {completeness * 100:.2f}% "
                 f"({exp_total - miss_total:,} dari {exp_total:,} bar yang diharapkan)"))

    def _desc(g: GapDay) -> str:
        more = "…" if g.missing > len(g.sample) else ""
        return f"{g.day} kurang {g.missing}/{g.expected} bar ({g.kind}: {', '.join(g.sample)}{more})"

    near = [g for g in gap_days if g.near]
    sig = [g for g in gap_days if not g.near and g.kind != "akhir"
           and (g.missing >= 3 or g.missing / g.expected >= 0.10)]
    sig_ids = {id(g) for g in sig}
    minor = [g for g in gap_days if not g.near and id(g) not in sig_ids]

    if near:
        findings.append(Finding(
            "WARN", f"Celah di batas chunk ({len(near)} hari), tidak bisa dipulihkan: "
                    + "; ".join(_desc(g) for g in near[:5]) + _more(near, 5)))
    if sig:
        findings.append(Finding(
            "WARN", f"Bar hilang di awal/tengah sesi ({len(sig)} hari): "
                    + "; ".join(_desc(g) for g in sig[:5]) + _more(sig, 5)))
    if minor:
        findings.append(Finding(
            "INFO", f"{len(minor)} hari lain kehilangan sedikit bar (umumnya tanpa transaksi, "
                    "sesi pendek, atau hari terakhir belum selesai)"))
    if skipped:
        findings.append(Finding(
            "INFO", f"{skipped} hari tidak dianalisis kelengkapan bar-nya "
                    f"(sampel hari-dalam-pekan < {MIN_TEMPLATE_DAYS})"))

    for g in near:
        targets.append((0, g.day - timedelta(days=1), g.day + timedelta(days=1)))
    for g in sig:
        targets.append((1, g.day - timedelta(days=1), g.day + timedelta(days=1)))
    return findings, targets, completeness, bool(near)


def analyze_dataframe(df, interval, req_start=None, req_end=None, boundary_dates=frozenset()):
    """Analisis menyeluruh atas data gabungan. Return Analysis."""
    if df is None or df.empty:
        return Analysis([Finding("ERROR", "Tidak ada data untuk dianalisis")], [])
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
    """Pastikan agregasi (mis. 15m → 45m) tidak menghilangkan bar/volume."""
    if base.empty or res.empty:
        return []
    vb, vr = float(base["Volume"].sum()), float(res["Volume"].sum())
    ok_vol = abs(vb - vr) <= 1e-6 * max(1.0, abs(vb))
    ok_hl = (np.isclose(base["High"].max(), res["High"].max())
             and np.isclose(base["Low"].min(), res["Low"].min()))
    if ok_vol and ok_hl:
        return [Finding("INFO", f"Resample {label}: {len(base):,} bar → {len(res):,} bar; "
                                "total volume serta High tertinggi/Low terendah cocok")]
    out = []
    if not ok_vol:
        out.append(Finding("ERROR", f"Resample {label}: total volume berubah ({vb:,.0f} → {vr:,.0f}); "
                                    "ada bar yang hilang saat agregasi"))
    if not ok_hl:
        out.append(Finding("ERROR", f"Resample {label}: High tertinggi/Low terendah tidak cocok dengan data dasar"))
    return out


# ====================== VALIDASI CHUNK & PEMULIHAN ======================
def assess_chunk(df_chunk, c_start: datetime, c_end: datetime, is_first: bool, is_last: bool):
    """Nilai satu chunk mentah. Return (perlu_diulang, catatan).

    Tepi chunk interior yang bolong biasanya akibat throttling/pemotongan jendela,
    jadi layak diambil ulang. Tepi terluar (awal rentang / hari ini) bisa wajar:
    saham baru listing atau data terbaru belum ada, hanya dicatat.
    """
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
        notes.append(f"data mulai {lead} hari setelah awal chunk")
        retry = retry or not is_first
    if trail > COVERAGE_TOL_DAYS:
        notes.append(f"data berakhir {trail} hari sebelum akhir chunk")
        retry = retry or not is_last
    return retry, notes


def overlap_stats(df_all: pd.DataFrame):
    """Bandingkan bar kembar (hasil tumpang-tindih chunk). Return (n_kembar, n_beda_ohlc, n_beda_volume)."""
    dup = df_all.loc[df_all.index.duplicated(keep=False), COLS]
    if dup.empty:
        return 0, 0, 0
    g = dup.groupby(level=0)
    hi, lo = g[OHLC].max(), g[OHLC].min()
    rel = (hi - lo) / lo.abs().clip(lower=1e-9)
    bad_ohlc = int((rel > 1e-4).any(axis=1).sum())
    vmax, vmin = g["Volume"].max(), g["Volume"].min()
    bad_vol = int(((vmax - vmin) / vmax.clip(lower=1) > 0.01).sum())
    return len(hi), bad_ohlc, bad_vol


def classify_chunks(reports):
    """Beri level tiap chunk dan hasilkan temuan tingkat-chunk."""
    out = []
    total = len(reports)
    nonempty = [r.index for r in reports if r.rows > 0]
    win = lambda r: f"{r.start:%Y-%m-%d} → {r.end:%Y-%m-%d}"  # noqa: E731

    for r in reports:
        exp = _bdays(r.start, r.eff_end)
        if r.rows == 0:
            if r.error:
                r.level = "ERROR"
                out.append(Finding(
                    "ERROR", f"Chunk {r.index}/{total} ({win(r)}) gagal diambil setelah "
                             f"{r.attempts} percobaan: {r.error}"))
            elif exp >= MIN_EXPECTED_BDAYS:
                if not nonempty or r.index < min(nonempty):
                    out.append(Finding(
                        "INFO", f"Chunk {r.index}/{total} ({win(r)}) kosong — sebelum data pertama "
                                "(kemungkinan belum listing)"))
                elif r.index > max(nonempty):
                    r.level = "WARN"
                    out.append(Finding(
                        "WARN", f"Chunk {r.index}/{total} ({win(r)}) kosong setelah chunk berisi data — "
                                "data terbaru hilang atau saham berhenti diperdagangkan"))
                else:
                    r.level = "ERROR"
                    out.append(Finding(
                        "ERROR", f"Chunk {r.index}/{total} ({win(r)}) kosong di tengah rentang — "
                                 "data bolong akibat proses chunk/sumber"))
            continue
        if r.notes:
            r.level = "WARN"
            out.append(Finding("WARN", f"Chunk {r.index}/{total} ({win(r)}): " + "; ".join(r.notes)))
        if r.issues:
            r.level = "ERROR"

    dens = [(r, r.rows / _bdays(r.start, r.eff_end)) for r in reports
            if r.rows > 0 and _bdays(r.start, r.eff_end) >= MIN_BDAYS_DENSITY]
    if len(dens) >= 2:
        med = float(np.median([d for _, d in dens]))
        for r, d in dens:
            if med > 0 and d < DENSITY_MIN_RATIO * med:
                if r.level == "OK":
                    r.level = "WARN"
                out.append(Finding(
                    "WARN", f"Chunk {r.index}/{total} ({win(r)}) kepadatan bar rendah: {d:.1f} bar/hari kerja "
                            f"vs median {med:.1f} — kemungkinan sebagian data hilang"))
    return out


def fetch_long_history(ticker, start_date, end_date, yf_interval, message=None, today=None):
    """Ambil data (di-chunk bila perlu) dan validasi tiap chunk. Return FetchResult.

    - Tiap chunk dinilai (assess_chunk); chunk kosong/bolong di tepi interior diambil ulang
      sampai CHUNK_MAX_ATTEMPTS kali.
    - Chunk ke-2 dst. mulai CHUNK_OVERLAP_DAYS lebih awal; bar kembar dibandingkan
      untuk membuktikan sambungan antar-chunk konsisten.
    - Chunk yang tetap gagal tidak membatalkan proses; data lain tetap dikembalikan.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    max_days = CHUNK_DAYS.get(yf_interval)
    intraday = yf_interval in INTRADAY_INTERVALS

    plan = plan_chunks(start_dt, end_dt, max_days) if max_days else [(start_dt, end_dt)]
    total = len(plan)
    safe_ticker = html.escape(ticker)

    progress_msg = None
    if message and total > 1:
        try:
            progress_msg = bot.send_message(
                message.chat.id,
                f"⏳ Mengambil data <b>{safe_ticker}</b> ({yf_interval})\n"
                f"{create_progress_bar(0)}\nMemulai...",
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
                f"⏳ Mengambil data <b>{safe_ticker}</b> ({yf_interval})\n"
                f"{create_progress_bar(percentage)}\n"
                f"Chunk {i}/{total} • {start_str} → {end_label}"
                + (f" • percobaan {attempt}/{CHUNK_MAX_ATTEMPTS}" if attempt > 1 else ""),
            )
            print(f"[CHUNK {i}/{total}] {ticker} {yf_interval}: {start_str} → {end_label} (percobaan {attempt})")
            try:
                df_chunk = fetch_history_with_retry(
                    ticker, start=start_str, end=end_str, interval=yf_interval
                )
                last_err = None
            except Exception as e:
                df_chunk, last_err = None, e
                print(f"[CHUNK ERROR] {ticker} {start_str} → {end_label}: {type(e).__name__}: {e}")
                if not is_transient_error(e):
                    break  # error permanen: mengulang tidak membantu

            if last_err is None:
                retry, last_notes = assess_chunk(df_chunk, c_start, eff_end, i == 1, i == total)
                if df_chunk is not None and not df_chunk.empty and (best is None or len(df_chunk) > len(best)):
                    best, best_notes = df_chunk, last_notes
                if not retry:
                    break
                print(f"[CHUNK CURIGA] {i}/{total}: {'; '.join(last_notes)} → coba ulang")
            if attempt < CHUNK_MAX_ATTEMPTS:
                time.sleep(2 * attempt)

        if best is not None and not best.empty:
            chunks.append(best)
            rep.rows = len(best)
            rep.notes = best_notes
            sane = best[COLS]
            sane = sane[~sane[OHLC].isna().all(axis=1)]
            rep.issues = [f.text for f in check_integrity(sane, intraday) if f.level == "ERROR"]
        else:
            rep.notes = last_notes
            if last_err is not None:
                rep.error = f"{type(last_err).__name__}: {last_err}"
                failed.append(f"{c_start:%Y-%m-%d} → {end_label}")

        if i < total:
            time.sleep(1.1)

    findings = classify_chunks(reports)
    ok = sum(1 for r in reports if r.level == "OK")
    status = "✅ Selesai" if ok == total else f"⚠️ Selesai, {total - ok} chunk perlu perhatian"
    _safe_edit(
        message, progress_msg,
        f"{status} mengambil data <b>{safe_ticker}</b> ({yf_interval})\n"
        f"{create_progress_bar(100)}\n{len(chunks)} dari {total} chunk berisi data • validasi chunk OK: {ok}/{total}",
    )

    boundary = set()
    for k in range(total - 1):
        boundary.add(plan[k][1].date())
        boundary.add(plan[k + 1][0].date())

    if not chunks:
        return FetchResult(pd.DataFrame(), failed, reports, boundary, findings)

    df = pd.concat(chunks)
    if total > 1:
        n_dup, n_bad_ohlc, n_bad_vol = overlap_stats(df)
        if n_dup:
            if n_bad_ohlc:
                findings.append(Finding(
                    "WARN", f"Overlap antar-chunk: {n_bad_ohlc} dari {n_dup} bar kembar punya OHLC berbeda "
                            "(dipakai versi chunk lebih awal) — sambungan chunk tidak sepenuhnya konsisten"))
            else:
                findings.append(Finding(
                    "INFO", f"Overlap antar-chunk: {n_dup} bar kembar dibandingkan, OHLC identik "
                            "— sambungan chunk konsisten"))
            if n_bad_vol:
                findings.append(Finding(
                    "INFO", f"Overlap antar-chunk: {n_bad_vol} bar kembar volumenya beda > 1%"))
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return FetchResult(df, failed, reports, boundary, findings)


def merge_windows(items):
    """Gabungkan jendela tanggal (awal, akhir) yang bersinggungan/bersebelahan."""
    out = []
    for s, e in sorted(items):
        if out and s <= out[-1][1] + timedelta(days=1):
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def select_repair_windows(targets, lo: Optional[date], hi: Optional[date]):
    """Pilih jendela refetch berdasarkan prioritas, dibatasi MAX_REPAIR_WINDOWS."""
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
    """Ambil ulang jendela bermasalah; tambahkan hanya bar yang belum ada. Return (df, bar_ditambah)."""
    added_total = 0
    lo, hi = pd.Timestamp(req_start.date()), pd.Timestamp(req_end.date())
    for k, (ws, we) in enumerate(windows):
        try:
            new = fetch_history_with_retry(
                ticker,
                start=ws.strftime("%Y-%m-%d"),
                end=(we + timedelta(days=1)).strftime("%Y-%m-%d"),
                interval=yf_interval,
            )
        except Exception as e:
            print(f"[REPAIR GAGAL] {ticker} {ws} → {we}: {type(e).__name__}: {e}")
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
        if k < len(windows) - 1:
            time.sleep(1.1)
    return df, added_total


def validate_and_repair(ticker, df, yf_interval, req_start, req_end, fetch: FetchResult, pre_findings):
    """Analisis data gabungan → refetch jendela bermasalah → analisis ulang.

    Return (df, findings, completeness).
    """
    boundary = fetch.boundary_dates
    analysis = analyze_dataframe(df, yf_interval, req_start, req_end, boundary)
    extra = []

    if req_start is not None and req_end is not None and analysis.targets:
        windows = select_repair_windows(analysis.targets, req_start.date(), req_end.date())
        if windows:
            wtxt = "; ".join(f"{s} → {e}" for s, e in windows[:4]) + _more(windows, 4)
            df2, added = repair_gaps(ticker, df, yf_interval, windows, req_start, req_end)
            if added:
                df = df2
                analysis = analyze_dataframe(df, yf_interval, req_start, req_end, boundary)
                extra.append(Finding(
                    "INFO", f"Pemulihan otomatis: {len(windows)} jendela diambil ulang ({wtxt}); "
                            f"{added} bar yang hilang berhasil ditambahkan"))
            else:
                extra.append(Finding(
                    "INFO", f"Pemulihan otomatis: {len(windows)} jendela diambil ulang ({wtxt}); "
                            "tidak ada bar baru — celah yang tersisa kemungkinan besar berasal dari sumber (Yahoo) "
                            "atau memang tidak ada transaksi, bukan dari proses chunk"))

    if boundary and yf_interval in INTRADAY_INTERVALS and analysis.completeness is not None \
            and not analysis.boundary_problem and all(r.rows > 0 for r in fetch.reports):
        extra.append(Finding(
            "INFO", f"Batas antar-chunk ({len(boundary) // 2} sambungan) diperiksa: tidak ditemukan celah bar"))

    findings = list(pre_findings) + list(fetch.findings) + extra + analysis.findings
    return df, findings, analysis.completeness


# ====================== LAPORAN VALIDASI ======================
def build_outcome(ticker, interval_input, yf_interval, range_info, tz_name,
                  rows, reports, findings, completeness) -> Outcome:
    findings = sorted(findings, key=lambda f: LEVEL_ORDER[f.level])
    counts = {lvl: sum(1 for f in findings if f.level == lvl) for lvl in LEVEL_ORDER}
    verdict = "ERROR" if counts["ERROR"] else ("WARN" if counts["WARN"] else "OK")

    comp = f", kelengkapan {completeness * 100:.1f}%" if completeness is not None else ""
    chunk = ""
    if len(reports) > 1:
        chunk = f", chunk OK {sum(1 for r in reports if r.level == 'OK')}/{len(reports)}"
    if verdict == "OK":
        summary = f"✅ Validasi: VALID{comp}{chunk}"
    elif verdict == "WARN":
        summary = f"⚠️ Validasi: {counts['WARN']} peringatan{comp}{chunk}"
    else:
        summary = f"❌ Validasi: {counts['ERROR']} masalah, {counts['WARN']} peringatan{comp}{chunk}"

    verdict_txt = {"OK": "VALID", "WARN": "VALID DENGAN PERINGATAN", "ERROR": "ADA MASALAH — periksa temuan ERROR"}[verdict]
    lines = [
        "LAPORAN VALIDASI DATA OHLCV",
        "=" * 27,
        f"Kode        : {ticker}",
        f"Timeframe   : {interval_input} (sumber Yahoo: {yf_interval})",
        f"Rentang     : {range_info}",
        f"Zona waktu  : {tz_name}",
        f"Total baris : {rows:,}",
        f"Status      : {verdict_txt}",
        f"Temuan      : {counts['ERROR']} ERROR, {counts['WARN']} WARN, {counts['INFO']} INFO",
        "",
    ]
    if reports:
        lines += [f"CHUNK ({len(reports)})",
                  f"{'#':>3}  {'Rentang':<25} {'Baris':>7} {'Coba':>4}  Status"]
        for r in reports:
            st = {"OK": "OK", "WARN": "PERINGATAN", "ERROR": "MASALAH"}[r.level]
            if r.rows == 0 and r.level == "OK":
                st = "KOSONG (wajar)"
            lines.append(f"{r.index:>3}  {r.start:%Y-%m-%d} → {r.end:%Y-%m-%d} {r.rows:>7,} {r.attempts:>4}  {st}")
            for n in r.notes:
                lines.append(f"       - {n}")
            for n in r.issues:
                lines.append(f"       - {n}")
        lines.append("")
    lines.append("TEMUAN")
    if findings:
        lines += [f"[{f.level}] {f.text}" for f in findings]
    else:
        lines.append("(tidak ada)")
    lines += [
        "",
        "Catatan: bar yang tidak muncul di Yahoo untuk saham kurang likuid sering berarti tidak ada",
        "transaksi pada menit tersebut, bukan kerusakan data. Data tidak pernah diubah/diisi sintetis;",
        "hanya duplikat dan baris kosong total yang dibuang, dan bar yang hilang ditambal dari refetch.",
    ]
    return Outcome(verdict, summary, "\n".join(lines))


INTERVAL_LABELS = {
    "60m": "1h", "1day": "1d", "1wk": "1w", "weekly": "1w",
    "1month": "1mo", "monthly": "1mo",
}


def build_csv_filename(ticker, interval_input, df, start_date=None, end_date=None):
    """Format: KODE_TIMEFRAME_TANGGALMULAI_to_TANGGALAKHIR.csv
    Contoh: BBCA.JK_1h_2025-01-01_to_2025-06-30.csv
    Jika rentang tidak diketahui (ytd/max), dipakai tanggal data pertama dan terakhir di df.
    """
    label = INTERVAL_LABELS.get(interval_input, interval_input) if interval_input else "data"
    if not (start_date and end_date) and not df.empty:
        start_date = df.index.min().strftime("%Y-%m-%d")
        end_date = df.index.max().strftime("%Y-%m-%d")
    if start_date and end_date:
        return f"{ticker}_{label}_{start_date}_to_{end_date}.csv"
    return f"{ticker}_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


def _make_caption(ticker, info, rows):
    head = f"📁 <b>{html.escape(ticker)}</b>\n"
    tail = f"\nJumlah baris: {rows}"
    room = TELEGRAM_CAPTION_LIMIT - len(head) - len(tail)
    esc = html.escape(info)
    while len(esc) > room and info:
        info = info[:-20]
        esc = html.escape(info) + "…"
    return head + esc + tail


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
        bot.send_document(
            message.chat.id,
            buf,
            caption=_make_caption(ticker, info, len(df)),
            parse_mode="HTML",
        )
    else:
        rows = 25
        while True:
            df_show = df.tail(rows)
            body = html.escape(df_show.to_string())
            text = (
                f"📈 <b>{html.escape(ticker)}</b>\n"
                f"{html.escape(info)}\n"
                f"Menampilkan {len(df_show)} baris terakhir dari total {len(df)} baris\n\n"
                f"<pre>{body}</pre>"
            )
            # Batas Telegram 4096 karakter; kurangi baris sampai muat.
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
    """Pisahkan flag ('csv', 'report') dari argumen. Return (args, flags).

    Token flag hanya dianggap flag mulai posisi ke-3 (setelah perintah dan kode),
    supaya ticker sungguhan bernama CSV (NYSE) tidak ikut terhapus.
    """
    raw = (text or "").split()
    head, tail = raw[:2], raw[2:]
    known = {"csv", "report"}
    flags = {a.lower() for a in tail if a.lower() in known}
    args = head + [a for a in tail if a.lower() not in known]
    return args, flags


# ====================== HANDLER ======================
def welcome(message):
    text = (
        "📊 <b>Bot OHLCV + CSV</b>\n\n"
        "<b>Cara Pakai:</b>\n\n"
        "<b>1. Pakai Shortcut Range (Paling Cepat)</b>\n"
        "<code>/ohlcv KODE TIMEFRAME RANGE [csv] [report]</code>\n\n"
        "Contoh:\n"
        "• <code>/ohlcv BBCA.JK 5m 30d</code>\n"
        "• <code>/ohlcv BBCA.JK 1h 3mo</code>\n"
        "• <code>/ohlcv BBCA.JK 15m 30d csv</code>\n"
        "• <code>/ohlcv TLKM.JK 5m max csv report</code>\n"
        "• <code>/ohlcv TLKM.JK 4h 1y</code>\n"
        "• <code>/ohlcv AAPL 1d 2y csv</code>\n"
        "• <code>/ohlcv BBCA.JK 1mo 5y csv</code>\n\n"
        "<b>Daftar Shortcut Range:</b>\n"
        "• <code>7d</code> / <code>1w</code> → 7 hari terakhir\n"
        "• <code>14d</code> / <code>2w</code> → 14 hari terakhir\n"
        "• <code>30d</code> / <code>1mo</code> → 1 bulan terakhir\n"
        "• <code>90d</code> / <code>3mo</code> → 3 bulan terakhir\n"
        "• <code>180d</code> / <code>6mo</code> → 6 bulan terakhir\n"
        "• <code>1y</code> → 1 tahun terakhir\n"
        "• <code>2y</code> → 2 tahun terakhir\n"
        "• <code>5y</code> → 5 tahun terakhir\n"
        "• <code>ytd</code> → Tahun berjalan\n"
        "• <code>max</code> → Data maksimal\n\n"
        "<b>2. Custom Range (Tanggal Bebas)</b>\n"
        "<code>/ohlcv KODE TIMEFRAME TANGGAL_MULAI TANGGAL_AKHIR [csv] [report]</code>\n\n"
        "Contoh:\n"
        "• <code>/ohlcv BBCA.JK 1h 2024-01-01 2025-06-30</code>\n"
        "• <code>/ohlcv TSLA 5m 2025-03-01 2025-10-20 csv</code>\n\n"
        "<b>Timeframe yang didukung:</b>\n"
        "5m, 15m, 30m, 45m, 1h, 2h, 3h, 4h, 1d, 1w, 1mo (bulanan)\n\n"
        "✅ Fitur: Auto Chunking • Validasi Tiap Chunk • Deteksi Gap/Anomali • Auto-Repair • "
        "Progress Bar • Auto Timezone • Nama File Rapi\n\n"
        "🔎 <b>Validasi data:</b> setiap hasil diperiksa (chunk kosong/bolong, bar hilang, OHLC tidak "
        "konsisten, lonjakan harga, sambungan antar-chunk). Laporan lengkap (.txt) dikirim otomatis bila "
        "ada peringatan/masalah; tambahkan <code>report</code> untuk selalu menerimanya.\n\n"
        "⚠️ <b>Batas data Yahoo Finance (intraday):</b>\n"
        "• 5m / 15m / 30m / 45m → hanya ±60 hari terakhir\n"
        "• 1h / 2h / 3h / 4h → hanya ±2 tahun terakhir\n"
        "• 1d / 1w / 1mo → bisa bertahun-tahun (1mo: pakai range ≥ 6mo agar ada beberapa candle)\n"
        "Jika range melebihi batas, tanggal mulai otomatis dipangkas dan Anda diberi tahu."
    )
    bot.reply_to(message, text, parse_mode="HTML")


def get_ohlcv(message):
    if not authorized(message):
        fail(message, "⛔ Anda tidak memiliki akses ke bot ini.")
        return

    chat_id = message.chat.id
    with _busy_lock:
        if chat_id in _busy_chats:
            fail(message, "⏳ Permintaan sebelumnya masih diproses. Tunggu sampai selesai.")
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
            fail(message, "Format salah. Ketik /help untuk melihat cara pakai.")
            return

        ticker = args[1].upper()
        if not TICKER_RE.match(ticker):
            fail(message, f"❌ Kode saham '{ticker}' tidak valid.")
            return

        interval_input = args[2].lower()
        if interval_input not in SUPPORTED_INTERVALS:
            fail(message,
                 f"❌ Timeframe '{interval_input}' tidak didukung.\n"
                 "Gunakan: 5m, 15m, 30m, 45m, 1h, 2h, 3h, 4h, 1d, 1w, 1mo\n"
                 "(catatan: 1m = 1 menit tidak didukung; untuk bulanan pakai 1mo)")
            return

        yf_interval, resample_rule = SUPPORTED_INTERVALS[interval_input]
        tz_name = detect_timezone(ticker)
        today = today_in_exchange_tz(ticker)

        notes = []
        use_period_max = False
        label_dates = True  # tampilkan tanggal di info & nama file

        # ---------- Tentukan rentang ----------
        if len(args) >= 5 and is_date(args[3]) and is_date(args[4]):
            # MODE 1: custom range
            start_dt = datetime.strptime(args[3], "%Y-%m-%d")
            end_dt = datetime.strptime(args[4], "%Y-%m-%d")
            if end_dt < start_dt:
                fail(message, "❌ Tanggal akhir lebih awal dari tanggal mulai.")
                return
            range_info = f"{args[3]} → {args[4]}"
        elif len(args) >= 4:
            # MODE 2: shortcut range
            range_input = args[3].lower()
            if range_input not in RANGE_SHORTCUTS:
                fail(message,
                     f"❌ Range '{range_input}' tidak dikenali.\n"
                     "Gunakan shortcut (3mo, 90d, 1y, dll) atau tanggal custom.\n"
                     "Ketik /help untuk melihat daftar lengkap.")
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
                if limit is None:
                    use_period_max = True
                    start_dt = None
                else:
                    start_dt = today - timedelta(days=limit)
            else:
                start_dt = end_dt - timedelta(days=value)
                range_info = f"{range_input} ({start_dt:%Y-%m-%d} → {end_dt:%Y-%m-%d})"
        else:
            fail(message, "Format salah. Ketik /help untuk melihat cara pakai.")
            return

        # ---------- Ambil data ----------
        if use_period_max:
            df_raw = fetch_history_with_retry(ticker, period="max", interval=yf_interval)
            fetch = FetchResult(df=df_raw if df_raw is not None else pd.DataFrame())
            start_date = end_date = None
            v_start = v_end = None  # rentang tak diketahui: cek cakupan & refetch dilewati
        else:
            start_dt, clamped, out_of_range = clamp_to_provider_limit(
                yf_interval, start_dt, end_dt, today
            )
            if out_of_range:
                limit = INTRADAY_LIMIT_DAYS[yf_interval]
                fail(message,
                     f"❌ Data {interval_input} hanya tersedia ±{limit} hari terakhir "
                     f"dari Yahoo Finance; rentang yang diminta sepenuhnya di luar batas itu.")
                return
            if clamped:
                notes.append(f"tanggal mulai dipangkas ke {start_dt:%Y-%m-%d} (batas Yahoo)")
            if yf_interval == "1mo" and (end_dt - start_dt).days < 62:
                notes.append("range <2 bulan: candle bulanan sangat sedikit")
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
            if clamped and label_dates:
                range_info = f"{start_date} → {end_date}"

        df = fetch.df
        if df is None or df.empty:
            fail(message, f"❌ Tidak ada data untuk {ticker} pada rentang tersebut.")
            return

        # ---------- Olah, validasi & pulihkan ----------
        df = df[COLS].copy()
        df = apply_timezone(df, ticker)
        df, clean_findings = clean_frame(df)
        if df.empty:
            fail(message, f"❌ Semua baris data {ticker} kosong (NaN) pada rentang tersebut.")
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
        if notes:
            info += " | " + "; ".join(notes)
        info += " | " + outcome.summary

        send_result(
            message, ticker, df, info, want_csv,
            interval_input=interval_input,
            start_date=start_date if label_dates else None,
            end_date=end_date if label_dates else None,
            outcome=outcome,
            want_report=want_report,
        )

    except NETWORK_ERRORS as e:
        print(f"[NETWORK ERROR] {e}")
        fail(message, "⚠️ Koneksi terputus. Silakan coba lagi beberapa saat.")
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        fail(message, f"⚠️ Error ({type(e).__name__}):\n{e}")


# ====================== MAIN ======================
def main():
    global bot
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("❌ Variabel lingkungan TELEGRAM_BOT_TOKEN belum diatur. Bot tidak dijalankan.")
        sys.exit(1)

    bot = telebot.TeleBot(token, parse_mode=None)
    bot.message_handler(commands=["start", "help"])(welcome)
    bot.message_handler(commands=["ohlcv"])(get_ohlcv)

    if not ALLOWED_CHAT_IDS:
        print("⚠️ ALLOWED_CHAT_IDS kosong: siapa pun yang menemukan bot ini bisa memakainya.")
    print("✅ Bot OHLCV (Shortcut Range + Validasi Chunk) sedang berjalan...")
    print("Tekan Ctrl+C untuk menghentikan.")

    while True:
        try:
            bot.infinity_polling(skip_pending=True, timeout=60, long_polling_timeout=60)
        except KeyboardInterrupt:
            print("Dihentikan oleh pengguna.")
            break
        except Exception as e:
            print(f"[POLLING ERROR] {e}")
            print("Mencoba menghubungkan ulang dalam 10 detik...")
            time.sleep(10)


if __name__ == "__main__":
    main()
