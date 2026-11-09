"""Bot Telegram OHLCV (yfinance) — v3: timeframe 5m + validasi chunk & anomali + Web Server Keep-Alive (Flask).

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
  6. Server Web Flask internal (`keep_alive`) agar Web Service tidak sleep.

Menjalankan:
    pip install pandas numpy yfinance pyTelegramBotAPI requests flask
    python -u bot_ohlcv.py

Token & chat ID ditulis langsung di bawah (variabel lingkungan TELEGRAM_BOT_TOKEN
diabaikan). ALLOWED_CHAT_IDS di environment, bila ada, ditambahkan ke daftar.
"""
print("BOOT bot_ohlcv dimulai", flush=True)

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
TELEGRAM_BOT_TOKEN = "8833515035:AAGKrxeesIBgt_B0TbbV4UvZDLcpojLb29M"

# Chat ID yang boleh memakai bot. Kosongkan set() agar semua orang boleh.
ALLOWED_CHAT_IDS = {5292651856}
for _part in os.getenv("ALLOWED_CHAT_IDS", "").split(","):
    _part = _part.strip()
    if _part.lstrip("-").isdigit():
        ALLOWED_CHAT_IDS.add(int(_part))

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


# ====================== KONFIGURASI BOT ======================
# Bot dibuat di main() agar modul bisa di-import/diuji tanpa token.
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

# Batas riwayat intraday Yahoo Finance (hari ke belakang dari hari ini).
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
OHLC_REL_TOL = 1e-4           # toleransi relatif cek High/Low terhadap Open/Close
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
    eff_end: datetime                     # end dibatasi hari ini (rentang masa depan tak dihitung)
    rows: int = 0
    attempts: int = 0
    notes: list = field(default_factory=list)    # catatan cakupan (tepi bolong, dll.)
    issues: list = field(default_factory=list)   # masalah struktur data di dalam chunk
    error: Optional[str] = None
    level: str = "OK"                     # OK | WARN | ERROR (diisi classify_chunks)


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
    targets: list                        # (prioritas, tgl_awal, tgl_akhir) kandidat refetch
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
            "Close": "last
