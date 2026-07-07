import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import pytz

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
    .metric-value { font-size: 1.7rem; font-weight: 700; }
    .trade-box { background-color: #161b22; border: 2px solid #00b8ff; border-radius: 12px; padding: 1rem; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = None

# Gold removed
MAIN_SYMBOLS = {
    "Bitcoin (BTC)": {"ticker": "BTC-USD", "display": "BTC/USD", "category": "Crypto"},
    "USD/JPY": {"ticker": "USDJPY=X", "display": "USD/JPY", "category": "Forex"},
    "NAS100": {"ticker": "NQ=F", "display": "NAS100 (NQ)", "category": "Index"},
}

def get_pakistan_time():
    tz = pytz.timezone('Asia/Karachi')
    return datetime.now(tz).strftime("%d %b %Y  |  %I:%M:%S %p PKT")

@st.cache_data(ttl=35, show_spinner=False)
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

def get_recent_candle_streak(df, lookback=15):
    if len(df) < lookback: lookback = len(df)
    recent = df.tail(lookback)
    colors = (recent['Close'] > recent['Open']).astype(int)
    streak = 1
    for i in range(len(colors)-2, -1, -1):
        if colors.iloc[i] == colors.iloc[-1]:
            streak += 1
        else:
            break
    return streak, colors.iloc[-1]

def calculate_signal_and_levels(df, tf="15m"):
    if df is None or len(df) < 30: return None
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
    if len(df) < 12: return None
    
    last = df.iloc[-1]
    price = float(last['Close'])
    ema9 = float(last['EMA_9'])
    ema21 = float(last['EMA_21'])
    rsi_val = float(last['RSI'])
    macd_h = float(last['MACD_Hist'])
    bb_u = float(last['BB_Upper'])
    bb_l = float(last['BB_Lower'])
    atr_val = float(last['ATR'])
    
    recent_high = float(df['High'].tail(18).max())
    recent_low = float(df['Low'].tail(18).min())
    
    high, low, close_p = float(last['High']), float(last['Low']), float(last['Close'])
    pp = (high + low + close_p) / 3
    r1 = 2 * pp - low
    s1 = 2 * pp - high
    
    score = 0
    reasons = []
    
    # Stronger confluence
    if price > ema9 > ema21:
        score += 2; reasons.append("✅ Strong bullish EMA alignment")
    elif price > ema9:
        score += 1; reasons.append("✅ Price above EMA9")
    elif price < ema9 < ema21:
        score -= 2; reasons.append("❌ Bearish EMA alignment")
    
    if rsi_val > 58: score += 1; reasons.append("✅ RSI strong bullish")
    elif rsi_val < 42: score -= 1; reasons.append("❌ RSI strong bearish")
    
    if macd_h > 0: score += 1; reasons.append("✅ MACD histogram positive")
    else: score -= 1; reasons.append("❌ MACD histogram negative")
    
    if price <= bb_l * 1.008: score += 1; reasons.append("✅ Near lower Bollinger (potential bounce)")
    elif price >= bb_u * 0.992: score -= 1; reasons.append("❌ Near upper Bollinger (potential rejection)")
    
    # Candle structure
    streak, last_color = get_recent_candle_streak(df, 12)
    if streak >= 5:
        if last_color == 1 and "BUY" in str(score): score += 1
        if last_color == 0 and "SELL" in str(score): score += 1
    
    if score >= 4: signal_type, badge_class = "STRONG BUY", "strong-buy"
    elif score >= 2: signal_type, badge_class = "BUY", "buy"
    elif score <= -4: signal_type, badge_class = "STRONG SELL", "strong-sell"
    elif score <= -2: signal_type, badge_class = "SELL", "sell"
    else: signal_type, badge_class = "NEUTRAL", "neutral"
    
    # Realistic TP/SL
    if "BUY" in signal_type:
        entry = round(price, 2)
        sl = round(min(recent_low, s1) - (atr_val * 0.45), 2)
        risk = max(entry - sl, atr_val * 0.65)
        tp1 = round(entry + risk * 1.7, 2)
        tp2 = round(entry + risk * 2.5, 2)
        tp3 = round(max(r1, entry + risk * 3.5), 2)
        rr = "1 : 1.7+"
    elif "SELL" in signal_type:
        entry = round(price, 2)
        sl = round(max(recent_high, r1) + (atr_val * 0.45), 2)
        risk = max(sl - entry, atr_val * 0.65)
        tp1 = round(entry - risk * 1.7, 2)
        tp2 = round(entry - risk * 2.5, 2)
        tp3 = round(min(s1, entry - risk * 3.5), 2)
        rr = "1 : 1.7+"
    else:
        entry = sl = tp1 = tp2 = tp3 = round(price, 2)
        rr = "N/A"
    
    # Smart Candle Prediction
    streak, last_color = get_recent_candle_streak(df, 12)
    base = max(3, min(9, int(abs(score) * 1.5)))
    
    if "BUY" in signal_type:
        expected = f"{base}–{base+3} green candles expected"
        pullback = f"Possible {max(1, base//3)}–{max(2, base//2)} red candles pullback"
    elif "SELL" in signal_type:
        expected = f"{base}–{base+3} red candles expected"
        pullback = f"Possible {max(1, base//3)}–{max(2, base//2)} green candles pullback"
    else:
        expected = "Market is neutral"
        pullback = ""
    
    return {
        "signal": signal_type, "badge_class": badge_class, "score": score, "reasons": reasons,
        "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3, "rr": rr,
        "rsi": round(rsi_val, 1), "ema9": round(ema9, 2), "ema21": round(ema21, 2),
        "atr": round(atr_val, 2), "last_price": round(price, 2),
        "expected_candles": expected, "pullback": pullback
    }

def build_chart(df, analysis, symbol_name, tf):
    if df is None or analysis is None: return None
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['Datetime'], open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name="Price",
        increasing_line_color="#00c853", decreasing_line_color="#f44336"))
    
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

st.markdown('<h1 class="main-header">📈 Pro Trading Signals</h1>', unsafe_allow_html=True)
st.caption(f"Pakistan Time: {get_pakistan_time()}  |  BTC • USDJPY • NAS100")

if st.button("🔄 Refresh All Data"):
    st.cache_data.clear()
    st.rerun()

# Grid Layout
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
        
        st.markdown("### 🕯️ Expected Candles")
        st.info(analysis['expected_candles'])
        if analysis.get('pullback'):
            st.warning(analysis['pullback'])
        
        st.markdown("### 🧠 Why this signal?")
        for r in analysis['reasons']:
            st.write(r)
    else:
        st.error("Not enough data for this timeframe. Try higher timeframe.")

st.caption("Professional Signals • Free Tier • Data via yfinance • Verify with your broker")
