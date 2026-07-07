import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Pro Trading Signals", layout="wide", initial_sidebar_state="expanded")

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
    .metric-value { font-size: 1.65rem; font-weight: 700; }
    .trade-box { background-color: #161b22; border: 2px solid #00b8ff; border-radius: 12px; padding: 1rem; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = None

# Gold hata diya
MAIN_SYMBOLS = {
    "Bitcoin (BTC)": {"ticker": "BTC-USD", "display": "BTC/USD", "category": "Crypto"},
    "USD/JPY": {"ticker": "USDJPY=X", "display": "USD/JPY", "category": "Forex"},
    "NAS100": {"ticker": "NQ=F", "display": "NAS100 (NQ)", "category": "Index"},
}

@st.cache_data(ttl=40, show_spinner=False)
def fetch_ohlcv(ticker, interval="15m", period="5d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty: return None
        df = df.reset_index()
        df.columns = [str(c[0]).capitalize() if isinstance(c, tuple) else str(c).capitalize() for c in df.columns]
        rename_map = {}
        for col in df.columns:
            if "datetime" in col.lower() or "date" in col.lower(): rename_map[col] = "Datetime"
            elif col.lower() in ["close", "open", "high", "low"]: rename_map[col] = col.capitalize()
        df = df.rename(columns=rename_map)
        if "Close" not in df.columns: return None
        return df[["Datetime", "Open", "High", "Low", "Close"]].dropna()
    except:
        return None

def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def rsi(series, length=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger_bands(series, length=20, std_dev=2):
    sma = series.rolling(window=length).mean()
    std = series.rolling(window=length).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def atr(df, length=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=length).mean()

def get_candle_prediction(signal_type, score, recent_green, recent_red):
    """Professional style candle prediction"""
    base = max(3, min(8, int(score * 1.3)))
    
    if "BUY" in signal_type:
        expected = f"{base}-{base+3} green candles"
        pullback = f"Possible {max(1, base//3)}-{max(2, base//2)} red candles in between"
    elif "SELL" in signal_type:
        expected = f"{base}-{base+3} red candles"
        pullback = f"Possible {max(1, base//3)}-{max(2, base//2)} green candles in between"
    else:
        expected = "Market is neutral"
        pullback = ""
    
    return expected, pullback

def calculate_signal_and_levels(df):
    if df is None or len(df) < 25: return None
    df = df.copy()
    close = df['Close']
    df['EMA_9'] = ema(close, 9)
    df['EMA_21'] = ema(close, 21)
    df['RSI'] = rsi(close, 14)
    _, _, macd_hist = macd(close)
    df['MACD_Hist'] = macd_hist
    bb_upper, _, bb_lower = bollinger_bands(close)
    df['BB_Upper'] = bb_upper
    df['BB_Lower'] = bb_lower
    df['ATR'] = atr(df, 14)
    df = df.dropna()
    if len(df) < 10: return None
    
    last = df.iloc[-1]
    price = float(last['Close'])
    ema9 = float(last['EMA_9'])
    ema21 = float(last['EMA_21'])
    rsi_val = float(last['RSI'])
    macd_h = float(last['MACD_Hist'])
    bb_u = float(last['BB_Upper'])
    bb_l = float(last['BB_Lower'])
    atr_val = float(last['ATR'])
    
    recent_high = float(df['High'].tail(15).max())
    recent_low = float(df['Low'].tail(15).min())
    
    high, low, close_p = float(last['High']), float(last['Low']), float(last['Close'])
    pp = (high + low + close_p) / 3
    r1 = 2 * pp - low
    s1 = 2 * pp - high
    
    score = 0
    reasons = []
    
    if price > ema9 > ema21: score += 2; reasons.append("✅ Strong bullish EMA alignment")
    elif price > ema9: score += 1; reasons.append("✅ Price above EMA9")
    elif price < ema9 < ema21: score -= 2; reasons.append("❌ Bearish EMA alignment")
    
    if rsi_val > 55: score += 1; reasons.append("✅ RSI bullish momentum")
    elif rsi_val < 45: score -= 1; reasons.append("❌ RSI bearish momentum")
    
    if macd_h > 0: score += 1; reasons.append("✅ MACD histogram positive")
    else: score -= 1; reasons.append("❌ MACD histogram negative")
    
    if price <= bb_l * 1.005: score += 1; reasons.append("✅ Near lower Bollinger Band")
    elif price >= bb_u * 0.995: score -= 1; reasons.append("❌ Near upper Bollinger Band")
    
    if score >= 4: signal_type, badge_class = "STRONG BUY", "strong-buy"
    elif score >= 2: signal_type, badge_class = "BUY", "buy"
    elif score <= -4: signal_type, badge_class = "STRONG SELL", "strong-sell"
    elif score <= -2: signal_type, badge_class = "SELL", "sell"
    else: signal_type, badge_class = "NEUTRAL", "neutral"
    
    # Realistic TP/SL
    if "BUY" in signal_type:
        entry = round(price, 2)
        sl = round(min(recent_low, s1) - (atr_val * 0.5), 2)
        risk = max(entry - sl, atr_val * 0.7)
        tp1 = round(entry + risk * 1.6, 2)
        tp2 = round(entry + risk * 2.3, 2)
        tp3 = round(max(r1, entry + risk * 3.2), 2)
        rr = "1 : 1.6+"
    elif "SELL" in signal_type:
        entry = round(price, 2)
        sl = round(max(recent_high, r1) + (atr_val * 0.5), 2)
        risk = max(sl - entry, atr_val * 0.7)
        tp1 = round(entry - risk * 1.6, 2)
        tp2 = round(entry - risk * 2.3, 2)
        tp3 = round(min(s1, entry - risk * 3.2), 2)
        rr = "1 : 1.6+"
    else:
        entry = sl = tp1 = tp2 = tp3 = round(price, 2)
        rr = "N/A"
    
    green, red = get_recent_candle_color(df, 12)
    expected, pullback = get_candle_prediction(signal_type, score, green, red)
    
    return {
        "signal": signal_type, "badge_class": badge_class, "score": score, "reasons": reasons,
        "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3, "rr": rr,
        "rsi": round(rsi_val, 1), "ema9": round(ema9, 2), "ema21": round(ema21, 2),
        "atr": round(atr_val, 2), "last_price": round(price, 2),
        "expected_candles": expected, "pullback": pullback
    }

def get_recent_candle_color(df, lookback=12):
    if len(df) < lookback: lookback = len(df)
    recent = df.tail(lookback)
    green = (recent['Close'] > recent['Open']).sum()
    red = (recent['Close'] < recent['Open']).sum()
    return green, red

def get_candle_prediction(signal_type, score, recent_green, recent_red):
    base = max(3, min(9, int(score * 1.4)))
    if "BUY" in signal_type:
        expected = f"{base}–{base+3} green candles expected"
        pullback = f"Possible {max(1, base//3)}–{max(2, base//2)} red candles pullback in between"
    elif "SELL" in signal_type:
        expected = f"{base}–{base+3} red candles expected"
        pullback = f"Possible {max(1, base//3)}–{max(2, base//2)} green candles pullback in between"
    else:
        expected = "Market is neutral"
        pullback = ""
    return expected, pullback

def build_chart(df, analysis, symbol_name, tf):
    if df is None or analysis is None: return None
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['Datetime'], open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name="Price",
        increasing_line_color="#00c853", decreasing_line_color="#f44336"))
    
    # Arrow
    last_price = float(df['Close'].iloc[-1])
    if "BUY" in analysis['signal']:
        fig.add_annotation(x=df['Datetime'].iloc[-1], y=last_price * 0.995,
            text="▲ LONG", showarrow=True, arrowhead=2, arrowsize=1.5,
            arrowcolor="#00c853", font=dict(color="#00c853", size=14))
    elif "SELL" in analysis['signal']:
        fig.add_annotation(x=df['Datetime'].iloc[-1], y=last_price * 1.005,
            text="▼ SHORT", showarrow=True, arrowhead=2, arrowsize=1.5,
            arrowcolor="#f44336", font=dict(color="#f44336", size=14))
    
    fig.update_layout(title=f"{symbol_name} — {tf} | {analysis['signal']}", template="plotly_dark",
        height=420, margin=dict(l=10, r=10, t=50, b=10), xaxis_rangeslider_visible=False)
    return fig

st.markdown('<h1 class="main-header">📈 Pro Trading Signals</h1>', unsafe_allow_html=True)
st.caption("BTC • USDJPY • NAS100 | Improved Professional Signals")

if st.button("🔄 Refresh All Data"):
    st.cache_data.clear()
    st.rerun()

# Grid layout
cols = st.columns(3)
for idx, (disp_name, meta) in enumerate(list(MAIN_SYMBOLS.items())):
    col = cols[idx % 3]
    with col:
        ticker = meta["ticker"]
        quick_df = fetch_ohlcv(ticker, interval="60m", period="2d")
        price, pct, sig, badge = 0, 0, "NEUTRAL", "neutral"
        if quick_df is not None and len(quick_df) > 1:
            price = float(quick_df['Close'].iloc[-1])
            pct = ((price - float(quick_df['Close'].iloc[0])) / float(quick_df['Close'].iloc[0])) * 100
            anal = calculate_signal_and_levels(quick_df)
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
    
    tf = st.selectbox("Timeframe", ["5m", "15m", "30m", "1h", "4h"], index=2)
    period_map = {"5m": "3d", "15m": "5d", "30m": "7d", "1h": "14d", "4h": "30d"}
    
    df = fetch_ohlcv(ticker, interval=tf, period=period_map[tf])
    analysis = calculate_signal_and_levels(df)
    
    if analysis:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", f"{analysis['last_price']}")
        c2.metric("Signal", analysis['signal'])
        c3.metric("RSI", analysis['rsi'])
        c4.metric("ATR", analysis['atr'])
        
        # Chart at bottom
        fig = build_chart(df, analysis, selected, tf)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Trade Setup
        st.markdown("### 🎯 Trade Setup")
        st.markdown(f"<span class='signal-badge {analysis['badge_class']}' style='font-size:1.3rem; padding:0.4rem 1.2rem;'>{analysis['signal']}</span>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="trade-box">
        <b>Entry:</b> {analysis['entry']}<br>
        <b>Stop Loss:</b> {analysis['sl']}<br>
        <b>TP1:</b> {analysis['tp1']} | <b>TP2:</b> {analysis['tp2']} | <b>TP3:</b> {analysis['tp3']}<br>
        <b>Risk : Reward</b> → {analysis['rr']}
        </div>
        """, unsafe_allow_html=True)
        
        st.code(f"Entry: {analysis['entry']}\nSL: {analysis['sl']}\nTP1: {analysis['tp1']}  TP2: {analysis['tp2']}  TP3: {analysis['tp3']}")
        
        # Candle Prediction
        st.markdown("### 🕯️ Candle Prediction")
        st.info(analysis['expected_candles'])
        if analysis['pullback']:
            st.warning(analysis['pullback'])
        
        st.markdown("### 🧠 Why this signal?")
        for r in analysis['reasons']:
            st.write(r)
    else:
        st.error("Not enough data for this timeframe.")

st.caption("Professional Signals • Free Tier • Data via yfinance • Verify with your broker")
