import os
import logging
import json
import time
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import yfinance as yf
import ta

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Environment variables
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
CHANNEL_ID = os.getenv("CHANNEL_ID")

# Store subscribers
subscribers = set()
signals_log = []

# Trading instruments (all available via Yahoo Finance - LEGAL)
TRADING_PAIRS = {
    "EURUSD": {"symbol": "EURUSD=X", "name": "Euro/US Dollar", "spread": 0.0001},
    "GBPUSD": {"symbol": "GBPUSD=X", "name": "British Pound/US Dollar", "spread": 0.0001},
    "USDJPY": {"symbol": "USDJPY=X", "name": "US Dollar/Japanese Yen", "spread": 0.01},
    "AUDUSD": {"symbol": "AUDUSD=X", "name": "Australian Dollar/US Dollar", "spread": 0.0001},
    "USDCAD": {"symbol": "USDCAD=X", "name": "US Dollar/Canadian Dollar", "spread": 0.0001},
    "USDCHF": {"symbol": "USDCHF=X", "name": "US Dollar/Swiss Franc", "spread": 0.0001},
    "NZDUSD": {"symbol": "NZDUSD=X", "name": "New Zealand Dollar/US Dollar", "spread": 0.0001},
    "BTCUSD": {"symbol": "BTC-USD", "name": "Bitcoin/US Dollar", "spread": 50},
    "ETHUSD": {"symbol": "ETH-USD", "name": "Ethereum/US Dollar", "spread": 5},
    "GOLD": {"symbol": "GC=F", "name": "Gold Futures", "spread": 0.5},
    "SILVER": {"symbol": "SI=F", "name": "Silver Futures", "spread": 0.05},
    "OIL": {"symbol": "CL=F", "name": "Crude Oil", "spread": 0.05}
}

def load_subscribers():
    global subscribers
    try:
        with open('subscribers.json', 'r') as f:
            subscribers = set(json.load(f))
    except FileNotFoundError:
        subscribers = set()

def save_subscribers():
    with open('subscribers.json', 'w') as f:
        json.dump(list(subscribers), f)

def get_real_time_data(symbol, period='5m', interval='1m'):
    """Fetch real-time market data from Yahoo Finance (LEGAL)"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        # Get current price
        current_price = df['Close'].iloc[-1] if not df.empty else None
        
        # Get bid/ask approximation (since Yahoo doesn't provide real bid/ask)
        spread = TRADING_PAIRS.get(symbol.split('=')[0], {}).get('spread', 0)
        bid = current_price - (spread / 2) if current_price else None
        ask = current_price + (spread / 2) if current_price else None
        
        return {
            'df': df,
            'current_price': current_price,
            'bid': bid,
            'ask': ask,
            'timestamp': datetime.now()
        }
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None

def calculate_signals(df, pair_name):
    """Generate trading signals based on technical analysis"""
    if df is None or len(df) < 20:
        return None
    
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    volume = df['Volume'].values
    
    # Calculate indicators
    signals = {
        'timestamp': datetime.now(),
        'pair': pair_name,
        'price': close[-1],
        'signals': []
    }
    
    # 1. RSI Signal
    rsi = ta.momentum.rsi(pd.Series(close), window=14).iloc[-1]
    if rsi < 30:
        signals['signals'].append({'type': 'BUY', 'indicator': 'RSI', 'strength': 'STRONG', 'reason': f'RSI oversold at {rsi:.1f}'})
    elif rsi > 70:
        signals['signals'].append({'type': 'SELL', 'indicator': 'RSI', 'strength': 'STRONG', 'reason': f'RSI overbought at {rsi:.1f}'})
    
    # 2. Moving Average Crossover
    ma5 = ta.trend.sma_indicator(pd.Series(close), window=5).iloc[-1]
    ma20 = ta.trend.sma_indicator(pd.Series(close), window=20).iloc[-1]
    
    if ma5 > ma20 and close[-1] > ma5:
        signals['signals'].append({'type': 'BUY', 'indicator': 'MA_CROSS', 'strength': 'MEDIUM', 'reason': 'MA5 crossed above MA20'})
    elif ma5 < ma20 and close[-1] < ma5:
        signals['signals'].append({'type': 'SELL', 'indicator': 'MA_CROSS', 'strength': 'MEDIUM', 'reason': 'MA5 crossed below MA20'})
    
    # 3. MACD Signal
    macd = ta.trend.macd(pd.Series(close))
    macd_signal = ta.trend.macd_signal(pd.Series(close))
    
    if not macd.empty and not macd_signal.empty:
        if macd.iloc[-1] > macd_signal.iloc[-1] and macd.iloc[-2] <= macd_signal.iloc[-2]:
            signals['signals'].append({'type': 'BUY', 'indicator': 'MACD', 'strength': 'MEDIUM', 'reason': 'MACD bullish crossover'})
        elif macd.iloc[-1] < macd_signal.iloc[-1] and macd.iloc[-2] >= macd_signal.iloc[-2]:
            signals['signals'].append({'type': 'SELL', 'indicator': 'MACD', 'strength': 'MEDIUM', 'reason': 'MACD bearish crossover'})
    
    # 4. Bollinger Bands
    bb_upper = ta.volatility.bollinger_hband(pd.Series(close))
    bb_lower = ta.volatility.bollinger_lband(pd.Series(close))
    
    if not bb_upper.empty and not bb_lower.empty:
        if close[-1] <= bb_lower.iloc[-1]:
            signals['signals'].append({'type': 'BUY', 'indicator': 'BB', 'strength': 'STRONG', 'reason': 'Price touched lower Bollinger Band'})
        elif close[-1] >= bb_upper.iloc[-1]:
            signals['signals'].append({'type': 'SELL', 'indicator': 'BB', 'strength': 'STRONG', 'reason': 'Price touched upper Bollinger Band'})
    
    # 5. Volume Analysis
    avg_volume = np.mean(volume[-20:-1]) if len(volume) >= 20 else volume[-1]
    volume_ratio = volume[-1] / avg_volume if avg_volume > 0 else 1
    
    if volume_ratio > 1.5 and close[-1] > close[-2]:
        signals['signals'].append({'type': 'BUY', 'indicator': 'VOLUME', 'strength': 'MEDIUM', 'reason': f'High volume uptrend ({volume_ratio:.1f}x)'})
    elif volume_ratio > 1.5 and close[-1] < close[-2]:
        signals['signals'].append({'type': 'SELL', 'indicator': 'VOLUME', 'strength': 'MEDIUM', 'reason': f'High volume downtrend ({volume_ratio:.1f}x)'})
    
    # 6. Support/Resistance
    support = min(low[-20:])
    resistance = max(high[-20:])
    
    if close[-1] <= support * 1.002:
        signals['signals'].append({'type': 'BUY', 'indicator': 'SUPPORT', 'strength': 'MEDIUM', 'reason': 'Price near support level'})
    elif close[-1] >= resistance * 0.998:
        signals['signals'].append({'type': 'SELL', 'indicator': 'RESISTANCE', 'strength': 'MEDIUM', 'reason': 'Price near resistance level'})
    
    # Determine overall signal
    buy_signals = [s for s in signals['signals'] if s['type'] == 'BUY']
    sell_signals = [s for s in signals['signals'] if s['type'] == 'SELL']
    
    if len(buy_signals) >= 3:
        signals['overall'] = 'STRONG_BUY'
        signals['confidence'] = min(95, 70 + len(buy_signals) * 8)
        signals['direction'] = 'CALL'
        signals['action'] = 'BUY'
    elif len(buy_signals) >= 2:
        signals['overall'] = 'BUY'
        signals['confidence'] = 65 + len(buy_signals) * 5
        signals['direction'] = 'CALL'
        signals['action'] = 'BUY'
    elif len(sell_signals) >= 3:
        signals['overall'] = 'STRONG_SELL'
        signals['confidence'] = min(95, 70 + len(sell_signals) * 8)
        signals['direction'] = 'PUT'
        signals['action'] = 'SELL'
    elif len(sell_signals) >= 2:
        signals['overall'] = 'SELL'
        signals['confidence'] = 65 + len(sell_signals) * 5
        signals['direction'] = 'PUT'
        signals['action'] = 'SELL'
    else:
        signals['overall'] = 'NEUTRAL'
        signals['confidence'] = 50
        signals['direction'] = 'WAIT'
        signals['action'] = 'WAIT'
    
    return signals

async def generate_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate signal for a specific pair"""
    if not context.args:
        # Show available pairs
        pairs_list = "\n".join([f"• {pair}" for pair in TRADING_PAIRS.keys()])
        await update.message.reply_text(
            f"📊 **Available Trading Pairs:**\n\n{pairs_list}\n\n"
            f"Example: `/signal EURUSD`\n"
            f"Example: `/signal BTCUSD`",
            parse_mode='Markdown'
        )
        return
    
    pair = context.args[0].upper()
    
    if pair not in TRADING_PAIRS:
        await update.message.reply_text(f"❌ Pair {pair} not found. Use /signal to see available pairs.")
        return
    
    await update.message.reply_text(f"🔍 Analyzing {pair}... Please wait.")
    
    # Get real-time data
    symbol = TRADING_PAIRS[pair]['symbol']
    data = get_real_time_data(symbol)
    
    if not data or data['df'] is None:
        await update.message.reply_text(f"❌ Unable to fetch data for {pair}. Please try again.")
        return
    
    # Generate signals
    signals = calculate_signals(data['df'], pair)
    
    if not signals:
        await update.message.reply_text(f"❌ Unable to generate signals for {pair}. Insufficient data.")
        return
    
    # Format signal message
    if signals['direction'] == 'CALL':
        emoji = "🚀"
        direction_text = "📈 **CALL (BUY)** - Price expected to RISE"
        entry_action = "BUY"
    elif signals['direction'] == 'PUT':
        emoji = "⚠️"
        direction_text = "📉 **PUT (SELL)** - Price expected to FALL"
        entry_action = "SELL"
    else:
        emoji = "⏸️"
        direction_text = "⏸️ **NEUTRAL** - No clear signal"
        entry_action = "WAIT"
    
    # Get entry/exit levels
    current_price = signals['price']
    if signals['direction'] == 'CALL':
        tp1 = current_price * 1.005  # 0.5% profit
        tp2 = current_price * 1.01   # 1% profit
        sl = current_price * 0.997   # 0.3% loss
    elif signals['direction'] == 'PUT':
        tp1 = current_price * 0.995  # 0.5% profit
        tp2 = current_price * 0.99   # 1% profit
        sl = current_price * 1.003   # 0.3% loss
    else:
        tp1 = tp2 = sl = current_price
    
    signal_message = (
        f"{emoji} **TRADING SIGNAL - {pair}** {emoji}\n\n"
        f"{direction_text}\n"
        f"📊 **Confidence:** {signals['confidence']}%\n"
        f"💰 **Current Price:** {current_price:.5f}\n"
        f"⏰ **Time:** {datetime.now().strftime('%H:%M:%S')}\n\n"
        f"**Technical Analysis:**\n"
    )
    
    # Add signals
    for sig in signals['signals'][:5]:  # Show top 5 signals
        strength_emoji = "🔥" if sig['strength'] == 'STRONG' else "📊"
        signal_message += f"{strength_emoji} {sig['reason']}\n"
    
    # Add trade plan
    if signals['direction'] != 'WAIT':
        signal_message += (
            f"\n💡 **Trade Plan ({signals['direction']}):**\n"
            f"• **Entry:** {entry_action} at {current_price:.5f}\n"
            f"• **Take Profit 1:** {tp1:.5f} (+0.5%)\n"
            f"• **Take Profit 2:** {tp2:.5f} (+1.0%)\n"
            f"• **Stop Loss:** {sl:.5f} (-0.3%)\n"
            f"• **Risk/Reward:** 1:1.67\n"
            f"• **Expiry:** 2-5 minutes\n\n"
        )
    else:
        signal_message += (
            f"\n⏸️ **Wait for better setup:**\n"
            f"• No clear signal at this moment\n"
            f"• Monitor RSI and moving averages\n"
            f"• Wait for confirmation\n\n"
        )
    
    signal_message += (
        f"⚠️ **Risk Management:**\n"
        f"• Max risk per trade: 1-2%\n"
        f"• Use proper position sizing\n"
        f"• Set stop loss immediately\n"
        f"• Take partial profits at TP1\n\n"
        f"🤖 **Signal generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"#Trading #{pair} #{signals['direction']}"
    )
    
    # Save signal to log
    signals_log.append({
        'time': datetime.now().isoformat(),
        'pair': pair,
        'direction': signals['direction'],
        'confidence': signals['confidence'],
        'price': current_price
    })
    
    # Keep only last 100 signals
    if len(signals_log) > 100:
        signals_log.pop(0)
    
    await update.message.reply_text(signal_message, parse_mode='Markdown')

async def auto_signals_loop(context: ContextTypes.DEFAULT_TYPE):
    """Auto-generate signals for all pairs"""
    for pair, info in TRADING_PAIRS.items():
        try:
            # Fetch data
            data = get_real_time_data(info['symbol'])
            if not data or data['df'] is None:
                continue
            
            # Generate signals
            signals = calculate_signals(data['df'], pair)
            
            # Send auto-signal for high confidence setups
            if signals and signals['confidence'] >= 75 and signals['direction'] != 'WAIT':
                current_price = signals['price']
                
                if signals['direction'] == 'CALL':
                    direction_text = "CALL (BUY) ⬆️"
                    action = "BUY"
                    tp1 = current_price * 1.005
                    sl = current_price * 0.997
                else:
                    direction_text = "PUT (SELL) ⬇️"
                    action = "SELL"
                    tp1 = current_price * 0.995
                    sl = current_price * 1.003
                
                signal_message = (
                    f"🚨 **AUTO SIGNAL - HIGH PROBABILITY** 🚨\n\n"
                    f"📊 **Pair:** {pair}\n"
                    f"🎯 **Signal:** {direction_text}\n"
                    f"📈 **Confidence:** {signals['confidence']}%\n"
                    f"💰 **Price:** {current_price:.5f}\n"
                    f"⏰ **Time:** {datetime.now().strftime('%H:%M:%S')}\n\n"
                    f"**Key Indicators:**\n"
                )
                
                # Add top 3 signals
                for sig in signals['signals'][:3]:
                    signal_message += f"• {sig['reason']}\n"
                
                signal_message += (
                    f"\n💡 **Quick Trade:**\n"
                    f"• {action} at {current_price:.5f}\n"
                    f"• TP: {tp1:.5f} (+0.5%)\n"
                    f"• SL: {sl:.5f} (-0.3%)\n"
                    f"• Duration: 2-3 minutes\n\n"
                    f"⚠️ Trade responsibly!\n"
                    f"#AutoSignal #{pair}"
                )
                
                # Send to all subscribers
                for user_id in subscribers:
                    try:
                        await context.bot.send_message(
                            chat_id=user_id,
                            text=signal_message,
                            parse_mode='Markdown'
                        )
                    except Exception as e:
                        logger.error(f"Failed to send to {user_id}: {e}")
                
                # Send to channel if configured
                if CHANNEL_ID:
                    await context.bot.send_message(
                        chat_id=CHANNEL_ID,
                        text=signal_message,
                        parse_mode='Markdown'
                    )
                
                logger.info(f"Auto-signal sent for {pair}: {signals['direction']} ({signals['confidence']}%)")
                
                # Rate limiting - don't spam
                time.sleep(5)
        
        except Exception as e:
            logger.error(f"Error generating auto-signal for {pair}: {e}")
        
        # Small delay between pairs
        time.sleep(1)

async def start_auto_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start auto-signal generation"""
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("❌ Only admin can start auto-signals.")
        return
    
    # Check if job already exists
    job_exists = False
    for job in context.application.job_queue.jobs():
        if job.name == 'auto_signal_job':
            job_exists = True
            break
    
    if not job_exists:
        context.application.job_queue.run_repeating(
            auto_signals_loop,
            interval=120,  # Check every 2 minutes
            first=10,
            name='auto_signal_job'
        )
        await update.message.reply_text(
            "✅ **Auto-Signal System Started!**\n\n"
            "The bot will now:\n"
            "• Monitor all trading pairs\n"
            "• Analyze market conditions\n"
            "• Send signals when confidence is HIGH (75%+)\n"
            "• Run 24/7 automatically\n\n"
            "Subscribers will receive signals automatically!\n\n"
            "Use /stopauto to stop."
        )
    else:
        await update.message.reply_text("ℹ️ Auto-signals are already running.")

async def stop_auto_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Stop auto-signal generation"""
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("❌ Only admin can stop auto-signals.")
        return
    
    for job in context.application.job_queue.jobs():
        if job.name == 'auto_signal_job':
            job.schedule_removal()
            await update.message.reply_text("✅ Auto-signal generation stopped.")
            return
    
    await update.message.reply_text("ℹ️ Auto-signals were not running.")

async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Subscribe to trading signals"""
    user_id = update.effective_user.id
    if user_id not in subscribers:
        subscribers.add(user_id)
        save_subscribers()
        await update.message.reply_text(
            "✅ **Subscribed to Trading Signals!**\n\n"
            "You'll receive:\n"
            "• Real-time trading signals\n"
            "• High-probability setups (75%+ confidence)\n"
            "• Entry/exit levels\n"
            "• Risk management guidelines\n\n"
            "Use /unsubscribe to stop receiving signals.",
            parse_mode='Markdown'
        )
    else:
        await update.message.reply_text("ℹ️ You're already subscribed!")

async def unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Unsubscribe from signals"""
    user_id = update.effective_user.id
    if user_id in subscribers:
        subscribers.remove(user_id)
        save_subscribers()
        await update.message.reply_text("✅ Unsubscribed from signals.")
    else:
        await update.message.reply_text("ℹ️ You're not subscribed.")

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show signal statistics"""
    if not signals_log:
        await update.message.reply_text("No signals generated yet.")
        return
    
    # Calculate stats
    total = len(signals_log)
    call_signals = len([s for s in signals_log if s['direction'] == 'CALL'])
    put_signals = len([s for s in signals_log if s['direction'] == 'PUT'])
    avg_confidence = sum(s['confidence'] for s in signals_log) / total
    
    stats_text = (
        f"📊 **Signal Statistics**\n\n"
        f"Total Signals: {total}\n"
        f"CALL Signals: {call_signals} ({call_signals/total*100:.1f}%)\n"
        f"PUT Signals: {put_signals} ({put_signals/total*100:.1f}%)\n"
        f"Avg Confidence: {avg_confidence:.1f}%\n\n"
        f"**Recent Signals:**\n"
    )
    
    for sig in signals_log[-5:]:
        stats_text += f"• {sig['pair']}: {sig['direction']} ({sig['confidence']}%) at {sig['price']:.5f}\n"
    
    await update.message.reply_text(stats_text, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help command"""
    help_text = (
        "🤖 **Trading Signal Bot**\n\n"
        "**Commands:**\n"
        "/signal [pair] - Get signal for specific pair\n"
        "/subscribe - Receive auto-signals\n"
        "/unsubscribe - Stop auto-signals\n"
        "/stats - View signal statistics\n"
        "/autostart - Start auto-signals (Admin)\n"
        "/stopauto - Stop auto-signals (Admin)\n"
        "/help - Show this help\n\n"
        "**Available Pairs:**\n"
        "EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD,\n"
        "BTCUSD, ETHUSD, GOLD, SILVER, OIL\n\n"
        "**How it works:**\n"
        "1. Bot analyzes 6+ technical indicators\n"
        "2. Generates signals with confidence levels\n"
        "3. Provides entry/exit levels\n"
        "4. Auto-sends high-confidence signals (75%+)\n\n"
        "**For Pocket Option:**\n"
        "Use the signals on Pocket Option's digital options:\n"
        "• CALL = Price will go UP\n"
        "• PUT = Price will go DOWN\n"
        "• Duration: 1-5 minutes recommended"
    )
    await update.message.reply_text(help_text, parse_mode='Markdown')

def main():
    """Start the bot"""
    load_subscribers()
    
    app = Application.builder().token(TOKEN).build()
    
    # Add handlers
    app.add_handler(CommandHandler("start", help_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("signal", generate_signal))
    app.add_handler(CommandHandler("subscribe", subscribe))
    app.add_handler(CommandHandler("unsubscribe", unsubscribe))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("autostart", start_auto_signals))
    app.add_handler(CommandHandler("stopauto", stop_auto_signals))
    
    logger.info("🚀 Trading Signal Bot Started!")
    logger.info(f"📊 Loaded {len(subscribers)} subscribers")
    logger.info("📈 Monitoring all trading pairs 24/7")
    logger.info("⚡ Auto-signals enabled for high-confidence setups")
    
    app.run_polling()

if __name__ == "__main__":
    main()
