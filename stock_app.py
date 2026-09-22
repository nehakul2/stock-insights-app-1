"""
Stock Insights App
-------------------
Multiple named "profiles" (e.g. Tech, Retirement), each with its own
watchlist and alerts (price threshold + SMA crossover). Alerts are checked
and shown as in-app banners each time the app loads. Click a ticker in the
SMA table to see full details, charts, and news.

Run with:
    streamlit run stock_app.py
"""

import io
import json
import os
import re
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh


st.set_page_config(page_title="Stock Insights", layout="centered", page_icon="📈")

SMA_WINDOWS = [5, 20, 50, 100, 200]
SMA_COLORS = {
    5: "#e15759",
    20: "#f28e2b",
    50: "#59a14f",
    100: "#af7aa1",
    200: "#76b7b2",
}

RANGE_OPTIONS = {
    "1mo": ("3mo", 22),
    "6mo": ("9mo", 130),
    "1y": ("2y", 252),
    "5y": ("6y", 1260),
}

# Alerts always check against enough history to support a 200-day SMA,
# regardless of whatever chart range the user has selected.
ALERT_FETCH_PERIOD = "2y"

# ---------- Persistence (JSON files on disk, one pair per person) ----------
# Note: on most hosting platforms (including free Streamlit Community Cloud
# tiers), the filesystem is NOT guaranteed to survive a redeploy or a long
# sleep/restart of the app — it persists for the life of the running
# container, which is enough to keep alerts across days of normal use, but
# isn't a substitute for a real database if you need guaranteed durability.
#
# This is name-based separation, not real authentication — anyone can type
# any name and see that name's data. It's just enough to stop testers from
# overwriting each other's watchlists during a testing round.
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_DIR = os.path.join(DATA_DIR, "users")
os.makedirs(USERS_DIR, exist_ok=True)

# A profile's "settings" are its saved view template — chart range, dark
# mode, which SMAs to show, and refresh preferences — so switching to a
# profile restores how that profile likes to be viewed instead of making
# the user re-pick every option each time.
DEFAULT_SETTINGS = {
    "range_label": "6mo",
    "dark_mode": False,
    "selected_smas": [20, 50],
    "auto_refresh_on": True,
    "refresh_interval_label": "2 min",
}

# Major indexes are related context for any watchlist (a quick sense of
# whether the market overall is up or down), so they're pre-loaded into the
# default template and every newly created profile rather than something
# each user has to remember to add. They can be removed like any other
# ticker (the ✕ next to it in the sidebar) if not wanted.
DEFAULT_INDEX_TICKERS = ["^GSPC", "^DJI", "^IXIC"]  # S&P 500, Dow Jones Industrial Average, Nasdaq Composite

DEFAULT_PROFILES = {
    "My Watchlist": {"tickers": list(DEFAULT_INDEX_TICKERS), "alerts": {}, "settings": dict(DEFAULT_SETTINGS)}
}


def ensure_profile_settings(profile: dict) -> dict:
    """Backfills a 'settings' block on profiles saved before this feature
    existed, and fills in any newly-added setting keys. Returns the dict."""
    if "settings" not in profile or not isinstance(profile.get("settings"), dict):
        profile["settings"] = dict(DEFAULT_SETTINGS)
    else:
        for key, value in DEFAULT_SETTINGS.items():
            profile["settings"].setdefault(key, value)
    return profile["settings"]


def apply_profile_settings_to_widgets(settings: dict) -> None:
    """Pushes a profile's saved settings into the session-state keys the
    sidebar widgets read from, so the next render shows that profile's
    saved view instead of whatever was on screen before."""
    for key, value in settings.items():
        st.session_state[key] = value


def sanitize_user_id(raw: str) -> str:
    """Turns whatever someone types into a safe filename fragment."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", raw.strip().lower())
    cleaned = cleaned.strip("_")
    return cleaned[:50] if cleaned else "guest"


def profiles_file(user_id: str) -> str:
    return os.path.join(USERS_DIR, f"{user_id}_profiles.json")


def alert_log_file(user_id: str) -> str:
    return os.path.join(USERS_DIR, f"{user_id}_alert_log.json")


def load_profiles(user_id: str) -> dict:
    path = profiles_file(user_id)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return json.loads(json.dumps(DEFAULT_PROFILES))  # deep copy


def save_profiles(user_id: str, profiles: dict) -> None:
    try:
        with open(profiles_file(user_id), "w") as f:
            json.dump(profiles, f, indent=2)
    except Exception as e:
        st.toast(f"Couldn't save profiles: {e}", icon="⚠️")


def load_alert_log(user_id: str) -> list:
    path = alert_log_file(user_id)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_alert_log(user_id: str, log: list) -> None:
    try:
        with open(alert_log_file(user_id), "w") as f:
            json.dump(log, f, indent=2)
    except Exception as e:
        st.toast(f"Couldn't save alert history: {e}", icon="⚠️")


def record_triggered_alerts(user_id: str, profile_name: str, messages: list) -> bool:
    """Appends newly-triggered alerts to this user's persistent log, skipping
    ones already logged today for the same profile+message (so a rerun or
    page refresh doesn't spam duplicate entries). Returns True if any new
    entries were actually added."""
    if not messages:
        return False

    log = load_alert_log(user_id)
    today = datetime.now().strftime("%Y-%m-%d")
    existing_today = {(e["date"], e["profile"], e["message"]) for e in log if e["date"] == today}

    changed = False
    for message in messages:
        key = (today, profile_name, message)
        if key not in existing_today:
            log.append({"date": today, "time": datetime.now().strftime("%H:%M"), "profile": profile_name, "message": message})
            existing_today.add(key)
            changed = True

    if changed:
        save_alert_log(user_id, log)
    return changed


# ---------- Sign-in gate ----------
# A simple name/ID entry — not a password — just enough to give each tester
# their own separate data instead of one shared file for everyone.

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if st.session_state.user_id is None:
    st.title("📈 Stock Insights")
    st.write("Enter your name to load your own profiles, watchlists, and alerts.")
    with st.form("sign_in"):
        name_input = st.text_input("Your name or ID", placeholder="e.g. Neha")
        submitted = st.form_submit_button("Continue")
    if submitted and name_input.strip():
        st.session_state.user_id = sanitize_user_id(name_input)
        st.rerun()
    st.stop()

USER_ID = st.session_state.user_id


# ---------- Auto-refresh ----------
# Reruns the whole script on a timer so prices/SMAs/alerts stay current
# without the user having to click anything. The actual st_autorefresh call
# happens after the sidebar defines the interval/toggle, further down.


# ---------- Session state: profiles ----------

if "profiles" not in st.session_state:
    st.session_state.profiles = load_profiles(USER_ID)
    for _profile in st.session_state.profiles.values():
        ensure_profile_settings(_profile)  # backfill for profiles saved before this feature existed
if "active_profile" not in st.session_state:
    st.session_state.active_profile = list(st.session_state.profiles.keys())[0]
if "extra_tickers_text" not in st.session_state:
    st.session_state.extra_tickers_text = ""
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None


def get_active_profile() -> dict:
    return st.session_state.profiles[st.session_state.active_profile]


def persist() -> None:
    """Call after any mutation to profiles/watchlists/alerts to save to disk."""
    save_profiles(USER_ID, st.session_state.profiles)


# Seed the sidebar widgets' session-state keys from the active profile's
# saved settings, but only the first time (if a key is already set, the
# user or a profile switch already put a value there this session).
if "settings_seeded" not in st.session_state:
    apply_profile_settings_to_widgets(ensure_profile_settings(get_active_profile()))
    st.session_state["settings_seeded"] = True


# ---------- Alert sound ----------
# A short beep generated on the fly and embedded as a base64 WAV — no
# external file or network request needed, so it works the same whether
# run locally or deployed.

@st.cache_data
def generate_beep_base64(freq: int = 880, duration_sec: float = 0.35, sample_rate: int = 44100) -> str:
    import base64
    import io as _io
    import math
    import struct
    import wave

    n_samples = int(sample_rate * duration_sec)
    buf = _io.BytesIO()
    with wave.open(buf, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for i in range(n_samples):
            # Fade out near the end so it doesn't click, gentle sine beep.
            t = i / sample_rate
            fade = min(1.0, (n_samples - i) / (sample_rate * 0.1))
            sample = int(32767 * 0.3 * fade * math.sin(2 * math.pi * freq * t))
            wav_file.writeframes(struct.pack("<h", sample))

    return base64.b64encode(buf.getvalue()).decode("ascii")


def play_alert_sound() -> None:
    b64_audio = generate_beep_base64()
    st.markdown(
        f"""
        <audio autoplay>
            <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
        </audio>
        """,
        unsafe_allow_html=True,
    )


def price_change_colors(current: float, previous_close: float) -> tuple:
    """Returns (background_rgba, border_rgba) for a price relative to
    yesterday's close — green if up, red if down, gray if unchanged/unknown."""
    if current is None or previous_close is None:
        return "rgba(158, 158, 158, 0.15)", "rgba(158, 158, 158, 0.6)"
    if current > previous_close:
        return "rgba(76, 175, 80, 0.15)", "rgba(76, 175, 80, 0.6)"
    elif current < previous_close:
        return "rgba(244, 67, 54, 0.15)", "rgba(244, 67, 54, 0.6)"
    return "rgba(158, 158, 158, 0.15)", "rgba(158, 158, 158, 0.6)"


# ---------- Cached data functions ----------

@st.cache_data(ttl=60, show_spinner=False)
def get_stock_summary(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    info = stock.info

    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    week_52_high = info.get("fiftyTwoWeekHigh")
    week_52_low = info.get("fiftyTwoWeekLow")
    long_name = info.get("longName", ticker)
    currency = info.get("currency", "")
    exchange = info.get("fullExchangeName") or info.get("exchange", "")

    if current_price is None:
        raise ValueError(f"Couldn't find data for ticker '{ticker}'. Check the symbol.")

    return {
        "name": long_name,
        "ticker": ticker.upper(),
        "price": current_price,
        "high_52wk": week_52_high,
        "low_52wk": week_52_low,
        "currency": currency,
        "exchange": exchange,
    }


@st.cache_data(ttl=60, show_spinner=False)
def fetch_history(ticker: str, period: str) -> pd.DataFrame:
    hist = yf.Ticker(ticker).history(period=period)
    if hist.empty:
        raise ValueError(f"No price history found for {ticker}. Check the symbol.")
    return hist


@st.cache_data(ttl=600, show_spinner=False)
def get_news(ticker: str, max_items: int = 5) -> list:
    try:
        news = yf.Ticker(ticker).news or []
    except Exception:
        return []

    items = []
    for item in news[:max_items]:
        content = item.get("content", item)
        title = content.get("title")
        provider = content.get("provider")
        publisher = provider.get("displayName") if isinstance(provider, dict) else content.get("publisher")
        canonical = content.get("canonicalUrl")
        link = canonical.get("url") if isinstance(canonical, dict) else content.get("link")
        if title:
            items.append({"title": title, "publisher": publisher or "Unknown source", "link": link})
    return items


# Common index symbols entered in non-Yahoo formats (Google Finance-style
# leading dot, or bare names) get mapped to Yahoo Finance's caret notation.
INDEX_ALIASES = {
    "IXIC": "^IXIC", ".IXIC": "^IXIC",          # Nasdaq Composite
    "DJI": "^DJI", ".DJI": "^DJI",              # Dow Jones Industrial Average
    "SPX": "^GSPC", ".SPX": "^GSPC", ".INX": "^GSPC", "GSPC": "^GSPC",  # S&P 500
    "SP100": "^SP100", ".SP100": "^SP100", "OEX": "^OEX", ".OEX": "^OEX",  # S&P 100
    "RUT": "^RUT", ".RUT": "^RUT",              # Russell 2000
    "VIX": "^VIX", ".VIX": "^VIX",              # CBOE Volatility Index
    "FTSE": "^FTSE", ".FTSE": "^FTSE",          # FTSE 100
    "N225": "^N225", ".N225": "^N225",          # Nikkei 225
}


def normalize_ticker(raw: str) -> str:
    t = raw.strip().upper()
    if t in INDEX_ALIASES:
        return INDEX_ALIASES[t]
    if t.startswith(".") and t[1:] not in ("",):
        # Unrecognized leading-dot symbol: try the caret convention as a best guess.
        return "^" + t[1:]
    return t


def add_smas(hist: pd.DataFrame, windows: list = SMA_WINDOWS) -> pd.DataFrame:
    hist = hist.copy()
    for window in windows:
        hist[f"SMA_{window}"] = hist["Close"].rolling(window=window).mean()
    return hist


def estimate_trend(hist: pd.DataFrame, lookback_days: int = 90):
    recent = hist.tail(min(lookback_days, len(hist)))
    if len(recent) < 10:
        raise ValueError("Not enough recent price history to estimate a trend.")

    closes = recent["Close"].values
    days = np.arange(len(closes)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(days, closes)

    last_day_index = len(closes) - 1
    horizons = {"1_day": 1, "7_day": 7, "30_day": 30}
    predictions = {
        label: round(float(model.predict(np.array([[last_day_index + h]]))[0]), 2)
        for label, h in horizons.items()
    }

    slope = model.coef_[0]
    direction = "upward" if slope > 0 else "downward" if slope < 0 else "flat"

    return model, predictions, direction, recent


def build_sma_table(tickers: list, period: str) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        try:
            hist = fetch_history(ticker, period)
            hist_with_sma = add_smas(hist)
            latest = hist_with_sma.iloc[-1]

            try:
                exchange = get_stock_summary(ticker).get("exchange", "")
            except Exception:
                exchange = ""

            row = {
                "ticker": ticker.upper(),
                "exchange": exchange,
                "date": hist_with_sma.index[-1].strftime("%Y-%m-%d"),
                "close": round(float(latest["Close"]), 2),
                "prev_close": round(float(hist_with_sma.iloc[-2]["Close"]), 2) if len(hist_with_sma) >= 2 else None,
            }
            for window in SMA_WINDOWS:
                value = latest.get(f"SMA_{window}")
                row[f"sma_{window}"] = round(float(value), 2) if pd.notna(value) else None
            rows.append(row)
        except Exception as e:
            rows.append({"ticker": ticker.upper(), "exchange": "", "date": None, "close": None, "error": str(e)})

    return pd.DataFrame(rows)


# ---------- Alerts ----------

def check_ticker_alerts(ticker: str, alerts: list) -> list:
    """Returns a list of triggered alert messages for one ticker's alert configs."""
    if not alerts:
        return []

    try:
        hist = fetch_history(ticker, ALERT_FETCH_PERIOD)
        hist_with_sma = add_smas(hist)
    except Exception:
        return [f"⚠️ Couldn't check alerts for {ticker} (data unavailable)."]

    if len(hist_with_sma) < 2:
        return []

    latest = hist_with_sma.iloc[-1]
    previous = hist_with_sma.iloc[-2]
    latest_close = float(latest["Close"])
    previous_close = float(previous["Close"])

    triggered = []
    for alert in alerts:
        a_type = alert["type"]

        if a_type == "price_above" and latest_close > alert["value"]:
            triggered.append(f"🔔 {ticker}: price {latest_close:.2f} is above your threshold of {alert['value']}")

        elif a_type == "price_below" and latest_close < alert["value"]:
            triggered.append(f"🔔 {ticker}: price {latest_close:.2f} is below your threshold of {alert['value']}")

        elif a_type in ("sma_cross_above", "sma_cross_below"):
            window = alert["value"]
            sma_col = f"SMA_{window}"
            latest_sma = latest.get(sma_col)
            previous_sma = previous.get(sma_col)
            if pd.isna(latest_sma) or pd.isna(previous_sma):
                continue

            was_below = previous_close <= previous_sma
            is_above = latest_close > latest_sma
            was_above = previous_close >= previous_sma
            is_below = latest_close < latest_sma

            if a_type == "sma_cross_above" and was_below and is_above:
                triggered.append(f"🔔 {ticker}: price crossed ABOVE its {window}-day SMA")
            elif a_type == "sma_cross_below" and was_above and is_below:
                triggered.append(f"🔔 {ticker}: price crossed BELOW its {window}-day SMA")

    return triggered


def check_profile_alerts(profile: dict) -> list:
    all_triggered = []
    for ticker, alerts in profile.get("alerts", {}).items():
        if ticker in profile["tickers"]:  # skip alerts for tickers removed from the profile
            all_triggered.extend(check_ticker_alerts(ticker, alerts))
    return all_triggered


# ---------- Chart functions ----------

def chart_style_context(dark_mode: bool):
    return plt.style.context("dark_background" if dark_mode else "default")


def make_price_chart(ticker, hist_with_sma, recent_for_trend, model, selected_smas, chart_days, dark_mode):
    plot_hist = hist_with_sma.tail(chart_days)

    closes_for_trend = recent_for_trend["Close"].values
    future_days = np.arange(len(closes_for_trend), len(closes_for_trend) + 30).reshape(-1, 1)
    future_preds = model.predict(future_days)
    last_date = recent_for_trend.index[-1]
    future_dates = [last_date + timedelta(days=i + 1) for i in range(30)]

    with chart_style_context(dark_mode):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(plot_hist.index, plot_hist["Close"], label=f"{ticker.upper()} actual", color="steelblue", linewidth=1.5)

        for window in selected_smas:
            col = f"SMA_{window}"
            if col in plot_hist.columns:
                ax.plot(plot_hist.index, plot_hist[col], label=f"SMA {window}", color=SMA_COLORS.get(window), linewidth=1, alpha=0.85)

        trend_color = "white" if dark_mode else "black"
        ax.plot(future_dates, future_preds, label="Trend estimate (30d)", color=trend_color, linestyle="--", linewidth=1)
        ax.set_title(f"{ticker.upper()} — Price History, SMAs & Trend Estimate")
        ax.set_xlabel("Date")
        ax.set_ylabel("Closing Price")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
    return fig


def make_sma_chart(ticker, hist_with_sma, chart_days, dark_mode):
    plot_hist = hist_with_sma.tail(chart_days)

    with chart_style_context(dark_mode):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for window in SMA_WINDOWS:
            col = f"SMA_{window}"
            if col in plot_hist.columns:
                ax.plot(plot_hist.index, plot_hist[col], label=f"SMA {window}", color=SMA_COLORS.get(window), linewidth=1.5)

        ax.set_title(f"{ticker.upper()} — Moving Averages (5/20/50/100/200-day)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
    return fig


def make_volume_chart(ticker, hist, chart_days, dark_mode):
    plot_hist = hist.tail(chart_days)

    with chart_style_context(dark_mode):
        fig, ax = plt.subplots(figsize=(8, 2.5))
        colors = np.where(plot_hist["Close"] >= plot_hist["Open"], "#59a14f", "#e15759")
        ax.bar(plot_hist.index, plot_hist["Volume"], color=colors, width=1.0)
        ax.set_title(f"{ticker.upper()} — Volume")
        ax.set_xlabel("Date")
        ax.set_ylabel("Shares traded")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
    return fig


def make_comparison_chart(histories: dict, dark_mode: bool):
    with chart_style_context(dark_mode):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for ticker, hist in histories.items():
            closes = hist["Close"]
            pct_change = (closes / closes.iloc[0] - 1) * 100
            ax.plot(hist.index, pct_change, label=ticker.upper(), linewidth=1.5)

        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_title("Comparison — % Change Over Selected Range")
        ax.set_xlabel("Date")
        ax.set_ylabel("% Change")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
    return fig


def make_interactive_ohlc_chart(ticker: str, hist: pd.DataFrame, dark_mode: bool):
    """An interactive candlestick chart (Plotly) with a built-in hover tooltip
    showing Date/Open/High/Low/Close at whatever point the cursor is over,
    plus a range slider/selector for zooming into a sub-period."""
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=hist.index,
                open=hist["Open"],
                high=hist["High"],
                low=hist["Low"],
                close=hist["Close"],
                increasing_line_color="#4caf50",
                decreasing_line_color="#f44336",
                name=ticker.upper(),
            )
        ]
    )

    fig.update_layout(
        title=f"{ticker.upper()} — Interactive Price Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark" if dark_mode else "plotly_white",
        height=500,
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=[
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all", label="All"),
                ]
            ),
        ),
        hovermode="x unified",
    )

    return fig


def render_stock_details(ticker: str, fetch_period: str, chart_days: int, selected_smas: list, dark_mode: bool):
    try:
        summary = get_stock_summary(ticker)
        hist = fetch_history(ticker, fetch_period)
        hist_with_sma = add_smas(hist)
        model, predictions, direction, recent_for_trend = estimate_trend(hist_with_sma)
    except Exception as e:
        st.error(f"Error loading {ticker}: {e}")
        return

    profile = get_active_profile()
    header_col, watch_col = st.columns([4, 1])
    header_col.subheader(f"{summary['name']} ({summary['ticker']})")
    if summary.get("exchange"):
        header_col.caption(f"Exchange: {summary['exchange']}")
    if ticker not in profile["tickers"]:
        if watch_col.button("⭐ Watch", key=f"watch_{ticker}"):
            profile["tickers"].append(ticker)
            persist()
            st.rerun()

    col1, col2, col3 = st.columns(3)
    with col1:
        prev_close_for_color = float(hist_with_sma.iloc[-2]["Close"]) if len(hist_with_sma) >= 2 else None
        bg_color, border_color = price_change_colors(summary["price"], prev_close_for_color)
        st.markdown(
            f"""
            <div style="background-color: {bg_color}; border: 1px solid {border_color};
                        border-radius: 8px; padding: 10px 12px;">
                <div style="font-size: 0.8rem; opacity: 0.8;">Current Price</div>
                <div style="font-size: 1.6rem; font-weight: 700;">{summary['price']} {summary['currency']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    col2.metric("52-Week High", f"{summary['high_52wk']} {summary['currency']}")
    col3.metric("52-Week Low", f"{summary['low_52wk']} {summary['currency']}")

    if selected_smas:
        st.markdown("**SMA (Simple Moving Averages)**")
        latest_smas = hist_with_sma.iloc[-1]
        sma_cols = st.columns(len(selected_smas))
        for col, window in zip(sma_cols, selected_smas):
            value = latest_smas.get(f"SMA_{window}")
            display_value = f"{value:.2f}" if pd.notna(value) else "N/A"
            with col:
                st.markdown(
                    f"""
                    <div style="background-color: rgba(33, 150, 243, 0.15); border: 1px solid rgba(33, 150, 243, 0.6);
                                border-radius: 8px; padding: 8px 10px; text-align: center;">
                        <div style="font-size: 0.75rem; opacity: 0.8;">SMA {window}</div>
                        <div style="font-size: 1.2rem; font-weight: 700;">{display_value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("**Previous trading day**")
    if len(hist_with_sma) >= 2:
        prev_day = hist_with_sma.iloc[-2]
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Open", f"{prev_day['Open']:.2f}")
        p2.metric("High", f"{prev_day['High']:.2f}")
        p3.metric("Low", f"{prev_day['Low']:.2f}")
        p4.metric("Close", f"{prev_day['Close']:.2f}")
        st.caption(f"Trading date: {hist_with_sma.index[-2].strftime('%Y-%m-%d')}")
    else:
        st.caption("Not enough history to show the previous trading day.")

    st.markdown(f"**Recent trend:** {direction}")
    t1, t7, t30 = st.columns(3)
    t1.metric("Est. in 1 day", predictions["1_day"])
    t7.metric("Est. in 7 days", predictions["7_day"])
    t30.metric("Est. in 30 days", predictions["30_day"])
    st.caption(
        "⚠️ These are naive statistical estimates based on recent price momentum only — "
        "not real forecasts. They don't account for news, earnings, or market sentiment. "
        "Not financial advice."
    )

    st.pyplot(make_price_chart(ticker, hist_with_sma, recent_for_trend, model, selected_smas, chart_days, dark_mode))

    st.markdown("**Volume**")
    st.pyplot(make_volume_chart(ticker, hist_with_sma, chart_days, dark_mode))

    st.markdown("**Moving averages (all 5, on their own scale)**")
    st.pyplot(make_sma_chart(ticker, hist_with_sma, chart_days, dark_mode))

    st.markdown("**Interactive chart — hover for Date/Open/High/Low/Close at any point**")
    st.plotly_chart(make_interactive_ohlc_chart(ticker, hist_with_sma, dark_mode), use_container_width=True, key=f"candlestick_{ticker}")
    st.caption("Drag the range slider below the chart, or use the 1w/1m/3m/6m/1y/All buttons, to zoom into a period.")

    st.markdown("**Open, High, Low, Close for a date range**")
    full_min_date = hist_with_sma.index.min().date()
    full_max_date = hist_with_sma.index.max().date()
    default_start = max(full_min_date, full_max_date - timedelta(days=30))

    date_col1, date_col2 = st.columns(2)
    range_start = date_col1.date_input(
        "From", value=default_start, min_value=full_min_date, max_value=full_max_date, key=f"ohlc_start_{ticker}"
    )
    range_end = date_col2.date_input(
        "To", value=full_max_date, min_value=full_min_date, max_value=full_max_date, key=f"ohlc_end_{ticker}"
    )

    if range_start > range_end:
        st.error("'From' date must be on or before 'To' date.")
    else:
        ranged = hist_with_sma.loc[str(range_start):str(range_end), ["Open", "High", "Low", "Close"]].round(2)
        if ranged.empty:
            st.caption("No trading days in that range.")
        else:
            ranged_display = ranged.copy()
            ranged_display.index = ranged_display.index.strftime("%Y-%m-%d")
            ranged_display.index.name = "Date"
            st.dataframe(ranged_display.iloc[::-1], use_container_width=True)  # most recent first

            ohlc_csv_buffer = io.StringIO()
            ranged_display.iloc[::-1].to_csv(ohlc_csv_buffer)
            st.download_button(
                label=f"Download {ticker} {range_start} to {range_end} (CSV)",
                data=ohlc_csv_buffer.getvalue(),
                file_name=f"{ticker}_{range_start}_to_{range_end}.csv",
                mime="text/csv",
                key=f"download_ohlc_{ticker}",
            )

    st.markdown("**Recent news**")
    news_items = get_news(ticker)
    if news_items:
        for n in news_items:
            if n["link"]:
                st.markdown(f"- [{n['title']}]({n['link']}) — *{n['publisher']}*")
            else:
                st.markdown(f"- {n['title']} — *{n['publisher']}*")
    else:
        st.caption("No recent news found for this ticker.")

    # ---- Alerts for this ticker ----
    st.markdown("**Alerts**")
    ticker_alerts = profile["alerts"].get(ticker, [])
    if ticker_alerts:
        for alert in ticker_alerts:
            label = {
                "price_above": f"Price above {alert['value']}",
                "price_below": f"Price below {alert['value']}",
                "sma_cross_above": f"Crosses above SMA {alert['value']}",
                "sma_cross_below": f"Crosses below SMA {alert['value']}",
            }[alert["type"]]
            acol1, acol2 = st.columns([4, 1])
            acol1.write(f"• {label}")
            if acol2.button("Remove", key=f"remove_alert_{alert['id']}"):
                profile["alerts"][ticker] = [a for a in ticker_alerts if a["id"] != alert["id"]]
                persist()
                st.rerun()
    else:
        st.caption("No alerts set for this ticker yet.")

    with st.form(key=f"add_alert_{ticker}"):
        st.write("Add a new alert")
        alert_type = st.selectbox(
            "Type",
            options=["price_above", "price_below", "sma_cross_above", "sma_cross_below"],
            format_func=lambda t: {
                "price_above": "Price goes above...",
                "price_below": "Price goes below...",
                "sma_cross_above": "Price crosses above SMA...",
                "sma_cross_below": "Price crosses below SMA...",
            }[t],
            key=f"alert_type_{ticker}",
        )
        if alert_type in ("price_above", "price_below"):
            value = st.number_input("Threshold price", min_value=0.0, step=1.0, key=f"alert_value_{ticker}")
        else:
            value = st.selectbox("SMA window", options=SMA_WINDOWS, key=f"alert_sma_{ticker}")

        if st.form_submit_button("Add alert"):
            profile["alerts"].setdefault(ticker, [])
            profile["alerts"][ticker].append({"id": str(uuid.uuid4()), "type": alert_type, "value": value})
            persist()
            st.rerun()


# ---------- Sidebar ----------

with st.sidebar:
    st.caption(f"Signed in as **{USER_ID}**")
    if st.button("Switch user"):
        for key in [
            "user_id", "profiles", "active_profile", "selected_ticker", "settings_seeded",
            "range_label", "dark_mode", "selected_smas", "auto_refresh_on", "refresh_interval_label",
        ]:
            st.session_state.pop(key, None)
        st.rerun()

    st.divider()
    st.header("Profile")
    profile_names = list(st.session_state.profiles.keys())
    active = st.selectbox("Active profile", options=profile_names, index=profile_names.index(st.session_state.active_profile))
    if active != st.session_state.active_profile:
        st.session_state.active_profile = active
        st.session_state.selected_ticker = None
        apply_profile_settings_to_widgets(ensure_profile_settings(get_active_profile()))
        st.rerun()

    new_profile_name = st.text_input("New profile name", key="new_profile_name", placeholder="e.g. Tech")
    pcol1, pcol2 = st.columns(2)
    if pcol1.button("Create profile") and new_profile_name.strip():
        name = new_profile_name.strip()
        if name not in st.session_state.profiles:
            st.session_state.profiles[name] = {
                "tickers": list(DEFAULT_INDEX_TICKERS),
                "alerts": {},
                "settings": dict(DEFAULT_SETTINGS),
            }
            st.session_state.active_profile = name
            apply_profile_settings_to_widgets(DEFAULT_SETTINGS)
            persist()
            st.rerun()
    if pcol2.button("Delete profile") and len(st.session_state.profiles) > 1:
        del st.session_state.profiles[st.session_state.active_profile]
        st.session_state.active_profile = list(st.session_state.profiles.keys())[0]
        st.session_state.selected_ticker = None
        apply_profile_settings_to_widgets(ensure_profile_settings(get_active_profile()))
        persist()
        st.rerun()

    st.divider()
    st.header("🔄 Data refresh")
    auto_refresh_on = st.toggle("Auto-refresh", key="auto_refresh_on")
    refresh_interval_label = st.selectbox(
        "Refresh every",
        options=["1 min", "2 min", "5 min", "10 min"],
        key="refresh_interval_label",
    )
    REFRESH_SECONDS = {"1 min": 60, "2 min": 120, "5 min": 300, "10 min": 600}[refresh_interval_label]

    if st.button("Refresh now"):
        st.cache_data.clear()
        st.session_state["last_refresh"] = datetime.now()
        st.rerun()

    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = datetime.now()
    st.caption(f"Last refreshed: {st.session_state['last_refresh'].strftime('%H:%M:%S')}")

    st.divider()
    st.header("Settings")
    range_label = st.selectbox("Chart time range", options=list(RANGE_OPTIONS.keys()), key="range_label")
    dark_mode = st.toggle("Dark mode charts", key="dark_mode")
    selected_smas = st.multiselect(
        "SMAs to show in detail chart",
        options=SMA_WINDOWS,
        key="selected_smas",
        format_func=lambda w: f"{w}-day SMA",
    )

    current_widget_settings = {
        "range_label": range_label,
        "dark_mode": dark_mode,
        "selected_smas": selected_smas,
        "auto_refresh_on": auto_refresh_on,
        "refresh_interval_label": refresh_interval_label,
    }
    is_unsaved = current_widget_settings != get_active_profile().get("settings", {})
    save_label = "💾 Save as this profile's default view" + (" •" if is_unsaved else "")
    if st.button(save_label):
        get_active_profile()["settings"] = current_widget_settings
        persist()
        st.toast(f"Saved as defaults for '{st.session_state.active_profile}'", icon="💾")
    if not is_unsaved:
        st.caption("✓ Matches this profile's saved defaults.")

    st.divider()
    st.header(f"⭐ Watchlist — {st.session_state.active_profile}")

    active_profile = get_active_profile()
    new_watch_ticker = st.text_input("Add ticker(s)", key="new_watch_ticker", placeholder="e.g. NVDA or NVDA, AMD")
    if st.button("Add to watchlist") and new_watch_ticker.strip():
        new_tickers = [normalize_ticker(t) for t in new_watch_ticker.split(",") if t.strip()]
        added = False
        for t in new_tickers:
            if t not in active_profile["tickers"]:
                active_profile["tickers"].append(t)
                added = True
        if added:
            persist()

    if active_profile["tickers"]:
        for t in list(active_profile["tickers"]):
            wcol1, wcol2 = st.columns([3, 1])
            wcol1.write(t)
            if wcol2.button("✕", key=f"remove_{t}"):
                active_profile["tickers"].remove(t)
                active_profile["alerts"].pop(t, None)
                if st.session_state.selected_ticker == t:
                    st.session_state.selected_ticker = None
                persist()
                st.rerun()
    else:
        st.caption("No tickers saved yet.")


# Fires a rerun every REFRESH_SECONDS while auto-refresh is on. Cache TTLs
# (60s for price/history data) are shorter than the shortest refresh option,
# so each auto-triggered rerun actually pulls fresh data instead of hitting
# a stale cache.
if auto_refresh_on:
    st_autorefresh(interval=REFRESH_SECONDS * 1000, key="data_autorefresh")
    st.session_state["last_refresh"] = datetime.now()


st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: rgba(127, 127, 127, 0.08);
        border-radius: 8px;
        padding: 10px 12px;
    }
    div[data-testid="column"] button[kind="secondary"] {
        background: none;
        border: none;
        color: #1a73e8;
        text-decoration: underline;
        padding: 0;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Main page ----------

st.title("📈 Stock Insights")
st.caption(f"Profile: **{st.session_state.active_profile}** — click a ticker in the table below to see full details.")

profile = get_active_profile()

# ---- Alert banner ----
triggered_alerts = check_profile_alerts(profile)
if triggered_alerts:
    st.warning("  \n".join(triggered_alerts))
    is_new = record_triggered_alerts(USER_ID, st.session_state.active_profile, triggered_alerts)
    if is_new:
        play_alert_sound()

ALERT_LABELS = {
    "price_above": "Price above {value}",
    "price_below": "Price below {value}",
    "sma_cross_above": "Crosses above SMA {value}",
    "sma_cross_below": "Crosses below SMA {value}",
}

with st.expander(f"⚙️ All configured alerts — {st.session_state.active_profile}", expanded=True):
    any_alerts = any(profile["alerts"].get(t) for t in profile["tickers"])
    if not any_alerts:
        st.caption("No alerts configured yet. Click a ticker below to add one.")
    else:
        for t in profile["tickers"]:
            ticker_alerts = profile["alerts"].get(t, [])
            if not ticker_alerts:
                continue
            st.markdown(f"**{t}**")
            for alert in ticker_alerts:
                label = ALERT_LABELS[alert["type"]].format(value=alert["value"])
                acol1, acol2 = st.columns([4, 1])
                acol1.write(f"• {label}")
                if acol2.button("Remove", key=f"remove_alert_overview_{alert['id']}"):
                    profile["alerts"][t] = [a for a in ticker_alerts if a["id"] != alert["id"]]
                    persist()
                    st.rerun()

with st.expander("🕘 Alert history (this profile)"):
    full_log = load_alert_log(USER_ID)
    profile_log = [e for e in full_log if e["profile"] == st.session_state.active_profile]
    profile_log.sort(key=lambda e: (e["date"], e["time"]), reverse=True)

    if profile_log:
        for entry in profile_log[:50]:
            st.write(f"**{entry['date']} {entry['time']}** — {entry['message']}")
        if st.button("Clear history for this profile"):
            remaining = [e for e in full_log if e["profile"] != st.session_state.active_profile]
            save_alert_log(USER_ID, remaining)
            st.rerun()
    else:
        st.caption("No alerts have triggered yet for this profile.")

extra_input = st.text_input(
    "Add extra tickers to this view (comma-separated, not saved to this profile)",
    key="extra_tickers_text",
    placeholder="e.g. TSLA, NFLX",
)
extra_tickers = [normalize_ticker(t) for t in extra_input.split(",") if t.strip()]

combined_tickers = list(dict.fromkeys(profile["tickers"] + extra_tickers))
fetch_period, chart_days = RANGE_OPTIONS[range_label]

if not combined_tickers:
    st.info("Add tickers to this profile's watchlist (sidebar) or the box above to get started.")
else:
    sma_table = build_sma_table(combined_tickers, fetch_period)

    st.subheader("SMA Overview")
    header_cols = st.columns([1.2, 1.3, 1, 1, 1, 1, 1, 1])
    for col, label in zip(header_cols, ["Ticker", "Exchange", "Close", "SMA 5", "SMA 20", "SMA 50", "SMA 100", "SMA 200"]):
        col.markdown(f"**{label}**")

    for _, row in sma_table.iterrows():
        cols = st.columns([1.2, 1.3, 1, 1, 1, 1, 1, 1])
        if cols[0].button(row["ticker"], key=f"select_{row['ticker']}"):
            st.session_state.selected_ticker = row["ticker"]
        cols[1].write(row.get("exchange") or "—")

        if "error" in row and pd.notna(row.get("error")):
            cols[2].markdown("⚠️ error")
        else:
            close_val = row["close"]
            prev_close_val = row.get("prev_close")
            if prev_close_val is not None:
                text_color = "#2e7d32" if close_val > prev_close_val else "#c62828" if close_val < prev_close_val else "inherit"
            else:
                text_color = "inherit"
            cols[2].markdown(f"<span style='color:{text_color}; font-weight:700;'>{close_val}</span>", unsafe_allow_html=True)
            for i, window in enumerate(SMA_WINDOWS, start=3):
                value = row.get(f"sma_{window}")
                cols[i].write(value if pd.notna(value) else "—")

    csv_buffer = io.StringIO()
    sma_table.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download SMA table as CSV",
        data=csv_buffer.getvalue(),
        file_name="sma_report.csv",
        mime="text/csv",
    )

    if len(combined_tickers) >= 2:
        with st.expander("📊 Compare all tickers (% change)"):
            histories = {}
            for t in combined_tickers:
                try:
                    histories[t] = add_smas(fetch_history(t, fetch_period)).tail(chart_days)
                except Exception:
                    pass
            if len(histories) >= 2:
                st.pyplot(make_comparison_chart(histories, dark_mode))
            else:
                st.caption("Not enough valid tickers to compare.")

    st.divider()

    if st.session_state.selected_ticker and st.session_state.selected_ticker in combined_tickers:
        if st.button("✕ Close details"):
            st.session_state.selected_ticker = None
            st.rerun()
        render_stock_details(st.session_state.selected_ticker, fetch_period, chart_days, selected_smas, dark_mode)
    else:
        st.caption("👆 Click a ticker above to see its full details, chart, alerts, and news.")
