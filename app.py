import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import pytz

st.set_page_config(page_title="Institutional Pro Signals", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .main-header { font-size: 2.2rem; font-weight: 700; background: linear-gradient(90deg, #00ff9f, #00b8ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; }
    .symbol-card { background-color: #161b22; border: 1px solid #30363d; border-radius: 14px; padding: 1rem; margin-bottom: 1rem; }
    .signal-badge { padding: 0.3rem 0.9rem; border-radius: 20px; font-weight: 700; font-size: 0.9rem; display: inline-block; }
    .strong-buy { background-color: #00c853; color: white; }
    .buy { background-color: #4caf50; color: white; }
    .neutral { background-color: #ff9800; color: white; }
    .sell { background-color: #f44336; color: white; }
    .strong-sell { background-color: #d32f2f; color: white; }
    .metric-value { font-size: 1.7rem; font-weight: 700; }
    .trade-box { background-color: #161b22; border: 2px solid #00b8ff; border-radius: 12px; padding: 1rem; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = None

# Assets aligned with your preference
MAIN_SYMBOLS = {
    "Bitcoin (BTC)": {"ticker": "BTC-USD", "display": "BTC/USD", "category": "Crypto"},
    "Ethereum (ETH)": {"ticker": "ETH-USD", "display": "ETH/USD", "category": "Crypto"},
    "Gold (XAU)": {"ticker": "GC=F", "display": "XAU/USD (Gold)", "category": "Commodity"},
}

def get_pakistan_time():
    tz = pytz.timezone('Asia/Karachi')
    return datetime.now(tz).strftime("%d %b %Y  |  %I:%M:%S %p PKT")

@st.cache_data(ttl=60, show_spinner=False)
def fetch_ohlcv(ticker, interval="15m", period="5d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty: return None
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            level0 = set(df.columns.get_level_values(0))
            if {"Close", "Open", "High", "Low", "Volume"} & level0:
                df.columns = [str(c).capitalize() for c in df.columns.get_level_values(0)]
            else:
                df.columns = [str(c).capitalize() for c in df.columns.get_level_values(1)]
        else:
            df.columns = [str(c).capitalize() for c in df.columns]
        
        rename_map = {}
        for col in df.columns:
            if "datetime" in col.lower() or "date" in col.lower(): rename_map[col] = "Datetime"
            elif col.lower() in ["close", "open", "high", "low", "volume"]: rename_map[col] = col.capitalize()
        df = df.rename(columns=rename_map)
        
        req_cols = ["Datetime", "Open", "High", "Low", "Close"]
        if "Volume" in df.columns:
            df["Volume"] = df["Volume"].fillna(0)
            req_cols.append("Volume")
        else:
            df["Volume"] = 0
            req_cols.append("Volume")
            
        return df[req_cols].dropna()
    except Exception:
        return None

def rsi(series, length=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    result = np.where(loss == 0, np.where(gain == 0, 50, 100), result)
    return pd.Series(result, index=series.index)

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(df, length=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=length).mean()

def calc_adx(df, period=14):
    high, low, close = df['High'], df['Low'], df['Close']
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_val)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_val)
    di_sum = plus_di + minus_di
    di_diff = abs(plus_di - minus_di)
    dx = 100 * (di_diff / di_sum.replace(0, np.nan))
    dx = dx.fillna(0)
    adx = dx.rolling(window=period).mean()
    return adx

@st.cache_data(ttl=120, show_spinner=False)
def get_htf_trend(ticker, current_tf):
    htf_map = {"5m": "1h", "15m": "1h", "30m": "4h", "1h": "4h", "4h": "1d"}
    htf = htf_map.get(current_tf, "1h")
    htf_df = fetch_ohlcv(ticker, interval=htf, period="30d")
    if htf_df is None or len(htf_df) < 50: return "NEUTRAL", 0
    htf_df['EMA_50'] = htf_df['Close'].ewm(span=50, adjust=False).mean()
    htf_df['EMA_200'] = htf_df['Close'].ewm(span=200, adjust=False).mean()
    last = htf_df.iloc[-1]
    if last['Close'] > last['EMA_50'] > last['EMA_200']: return "BULLISH", 2
    elif last['Close'] < last['EMA_50'] < last['EMA_200']: return "BEARISH", -2
    return "NEUTRAL", 0

def calculate_signal_and_levels(df, ticker, tf="15m"):
    if df is None or len(df) < 50: return None
    df = df.copy()
    close = df['Close']
    
    df['EMA_9'] = close.ewm(span=9, adjust=False).mean()
    df['EMA_21'] = close.ewm(span=21, adjust=False).mean()
    df['RSI'] = rsi(close, 14)
    _, _, macd_hist = macd(close)
    df['MACD_Hist'] = macd_hist
    df['ADX'] = calc_adx(df, 14)
    
    if df['Volume'].sum() > 0:
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        cumulative_tp_vol = (tp * df['Volume']).cumsum()
        cumulative_vol = df['Volume'].cumsum().replace(0, np.nan)
        df['VWAP'] = cumulative_tp_vol / cumulative_vol
        df['VWAP'] = df['VWAP'].fillna(close.rolling(20).mean())
    else:
        df['VWAP'] = close.rolling(20).mean()
        
    df['ATR'] = atr(df, 14)
    df = df.dropna()
    if len(df) < 12: return None

    last = df.iloc[-1]
    price = float(last['Close'])
    adx = float(last['ADX'])
    
    # 1. Higher Timeframe Trend
    htf_trend, htf_score = get_htf_trend(ticker, tf)
    
    # 2. Scoring System (Professional Confluence)
    score = htf_score
    reasons = []
    reasons.append(f"🌐 HTF Trend ({tf} -> Higher TF): {htf_trend}")
    
    # LTF Trend
    if price > float(last['EMA_9']) > float(last['EMA_21']):
        score += 2; reasons.append("✅ LTF Strong Bullish (Price > EMA9 > EMA21)")
    elif price < float(last['EMA_9']) < float(last['EMA_21']):
        score -= 2; reasons.append("❌ LTF Strong Bearish (Price < EMA9 < EMA21)")
    else:
        reasons.append("➖ LTF Mixed/Consolidating")
        
    # VWAP (Institutional Level)
    vwap = float(last['VWAP'])
    if price > vwap:
        score += 1; reasons.append("✅ Price Above VWAP (Institutional Bullish Bias)")
    else:
        score -= 1; reasons.append("❌ Price Below VWAP (Institutional Bearish Bias)")
        
    # Momentum (RSI + MACD)
    rsi_val = float(last['RSI'])
    macd_h = float(last['MACD_Hist'])
    if rsi_val > 55 and macd_h > 0:
        score += 1; reasons.append("✅ Bullish Momentum (RSI>55 + MACD+)")
    elif rsi_val < 45 and macd_h < 0:
        score -= 1; reasons.append("❌ Bearish Momentum (RSI<45 + MACD-)")
        
    # Trend Strength (ADX) - The Dead Signal Killer
    if adx < 20:
        reasons.append("⚠️ ADX < 20: Market is RANGING (Dead Signal Filtered)")
        if score > 1: score = 1
        elif score < -1: score = -1

    if score >= 4: signal_type, badge_class = "STRONG BUY", "strong-buy"
    elif score >= 2: signal_type, badge_class = "BUY", "buy"
    elif score <= -4: signal_type, badge_class = "STRONG SELL", "strong-sell"
    elif score <= -2: signal_type, badge_class = "SELL", "sell"
    else: signal_type, badge_class = "WAIT / NEUTRAL", "neutral"

    # SL / TP Logic based on Market Structure (Swing High/Low)
    lookback = min(50, len(df))
    recent_high = float(df['High'].tail(lookback).max())
    recent_low = float(df['Low'].tail(lookback).min())
    atr_val = float(last['ATR'])
    
    if "BUY" in signal_type:
        entry = round(price, 2)
        sl = round(recent_low - (atr_val * 0.5), 2)
        risk = entry - sl
        if risk <= 0: risk = atr_val
        tp1 = round(entry + risk * 1.5, 2)
        tp2 = round(entry + risk * 2.5, 2)
        tp3 = round(recent_high, 2) # Target the range high
        rr = f"1 : {round((tp1 - entry) / risk, 1)}"
    elif "SELL" in signal_type:
        entry = round(price, 2)
        sl = round(recent_high + (atr_val * 0.5), 2)
        risk = sl - entry
        if risk <= 0: risk = atr_val
        tp1 = round(entry - risk * 1.5, 2)
        tp2 = round(entry - risk * 2.5, 2)
        tp3 = round(recent_low, 2) # Target the range low
        rr = f"1 : {round((entry - tp1) / risk, 1)}"
    else:
        entry = sl = tp1 = tp2 = tp3 = round(price, 2)
        rr = "N/A"
        
    return {
        "signal": signal_type, "badge_class": badge_class, "score": score, "reasons": reasons,
        "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3, "rr": rr,
        "rsi": round(rsi_val, 1), "adx": round(adx, 1), "atr": round(atr_val, 2), 
        "vwap": round(vwap, 2), "last_price": round(price, 2)
    }

def build_chart(df, analysis, symbol_name, tf):
    if df is None or analysis is None: return None
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['Datetime'], open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name="Price",
        increasing_line_color="#00c853", decreasing_line_color="#f44336"))

    if 'VWAP' in df.columns:
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['VWAP'], mode='lines', 
            name='VWAP', line=dict(color='#ffeb3b', width=1, dash='dash')))

    last_price = float(df['Close'].iloc[-1])
    if "BUY" in analysis['signal']:
        fig.add_annotation(x=df['Datetime'].iloc[-1], y=last_price*0.992,
            text="▲ LONG", showarrow=True, arrowhead=2, arrowsize=1.8,
            arrowcolor="#00c853", font=dict(color="#00c853", size=15, family="Arial Black"))
    elif "SELL" in analysis['signal']:
        fig.add_annotation(x=df['Datetime'].iloc[-1], y=last_price*1.008,
            text="▼ SHORT", showarrow=True, arrowhead=2, arrowsize=1.8,
            arrowcolor="#f44336", font=dict(color="#f44336", size=15, family="Arial Black"))

    fig.update_layout(title=f"{symbol_name} — {tf} | {analysis['signal']}", template="plotly_dark",
        height=400, margin=dict(l=10, r=10, t=50, b=10), xaxis_rangeslider_visible=False)
    return fig

st.markdown('<h1 class="main-header">📈 Institutional Pro Signals</h1>', unsafe_allow_html=True)
st.caption(f"Pakistan Time: {get_pakistan_time()}  |  BTC • ETH • Gold")

# Actionable Snapshots (No Noise)
with st.expander("📱 Daily Actionable Snapshots (No Noise)"):
    if st.button("Generate Clean Levels"):
        snapshot_text = "📊 **ACTIONABLE LEVELS ONLY - NO NOISE**\n\n"
        for name, meta in MAIN_SYMBOLS.items():
            df_snap = fetch_ohlcv(meta['ticker'], interval="1h", period="7d")
            anal = calculate_signal_and_levels(df_snap, meta['ticker'], "1h")
            if anal and "NEUTRAL" not in anal['signal'] and "WAIT" not in anal['signal']:
                snapshot_text += f"**{meta['display']} ({anal['signal']})**\n"
                snapshot_text += f"Entry: {anal['entry']} | SL: {anal['sl']}\n"
                snapshot_text += f"TP1: {anal['tp1']} | TP2: {anal['tp2']} | TP3: {anal['tp3']}\n"
                snapshot_text += f"R:R -> {anal['rr']}\n"
                snapshot_text += f"Rule: Wait for 15m candle close to confirm.\n\n"
            else:
                snapshot_text += f"**{meta['display']}**: ⚠️ NO TRADE (Market Ranging/Neutral)\n\n"
        st.code(snapshot_text)

if st.button("🔄 Refresh All Data"):
    st.cache_data.clear()
    st.rerun()

# Grid Layout
cols = st.columns(3)
for idx, (disp_name, meta) in enumerate(list(MAIN_SYMBOLS.items())):
    col = cols[idx % 3]
    with col:
        ticker = meta["ticker"]
        quick_df = fetch_ohlcv(ticker, interval="1h", period="3d")
        price, pct, sig, badge = 0, 0, "WAIT", "neutral"
        if quick_df is not None and len(quick_df) > 1:
            price = float(quick_df['Close'].iloc[-1])
            pct = ((price - float(quick_df['Close'].iloc[0])) / float(quick_df['Close'].iloc[0])) * 100
            anal = calculate_signal_and_levels(quick_df, ticker, "1h")
            if anal: sig, badge = anal["signal"], anal["badge_class"]

        st.markdown(f"""
        <div class="symbol-card">
            <strong>{meta['display']}</strong><br>
            <span class="metric-value">{price:,.2f}</span>
            <span style="color:{'#00c853' if pct >= 0 else '#f44336'};"> {pct:+.2f}%</span><br>
            <span class="signal-badge {badge}">{sig}</span>
        </div>
        """, unsafe_allow_html=True)

        if st.button(f"View Analysis", key=f"btn_{disp_name}"):
            st.session_state.selected_symbol = disp_name
            st.rerun()

# Detailed View
if st.session_state.selected_symbol:
    selected = st.session_state.selected_symbol
    meta = MAIN_SYMBOLS.get(selected, {})
    ticker = meta.get("ticker", "")
    st.divider()
    st.subheader(f"📊 {selected}")

    tf = st.selectbox("Timeframe", ["5m", "15m", "30m", "1h", "4h"], index=1)
    period_map = {"5m": "3d", "15m": "5d", "30m": "7d", "1h": "14d", "4h": "60d"}

    df = fetch_ohlcv(ticker, interval=tf, period=period_map[tf])
    analysis = calculate_signal_and_levels(df, ticker, tf)

    if analysis:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Price", f"{analysis['last_price']}")
        c2.metric("Signal", analysis['signal'])
        c3.metric("RSI", analysis['rsi'])
        c4.metric("ADX", analysis['adx'])
        c5.metric("VWAP", analysis['vwap'])

        fig = build_chart(df, analysis, selected, tf)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🎯 Institutional Trade Setup")
        st.markdown(f"<span class='signal-badge {analysis['badge_class']}' style='font-size:1.3rem; padding:0.4rem 1.2rem;'>{analysis['signal']}</span>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="trade-box">
        <b>Entry:</b> {analysis['entry']}<br>
        <b>Stop Loss (Structure):</b> {analysis['sl']}<br>
        <b>TP1:</b> {analysis['tp1']} | <b>TP2:</b> {analysis['tp2']} | <b>TP3 (Structure):</b> {analysis['tp3']}<br>
        <b>Risk : Reward</b> → {analysis['rr']}<br>
        <b>Rule:</b> Risk max 1-2% of account. Wait for candle close.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🧠 Confluence & Logic")
        for r in analysis['reasons']:
            st.write(r)
    else:
        st.error("Not enough data for this timeframe. Try higher timeframe.")

st.caption("Professional Institutional Signals • Dead Logic Removed • Data via yfinance")
