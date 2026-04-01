# Telegram Trading Signal Bot

A Telegram bot that provides real-time trading signals for Pocket Option, Forex, and Crypto.

## Features
- Real-time market analysis
- Trading signals for EURUSD, GBPUSD, BTCUSD, and more
- 1-5 minute timeframe signals
- Auto-signals with 75%+ confidence
- Subscriber management

## Commands
- `/signal [pair]` - Get trading signal
- `/subscribe` - Receive auto-signals
- `/unsubscribe` - Stop signals
- `/stats` - View statistics
- `/autostart` - Start auto-signals (admin)
- `/help` - Help menu

## Deployment
1. Create bot via @BotFather
2. Deploy on Railway
3. Set environment variables
4. Start with `/autostart`

## Environment Variables
- `TELEGRAM_BOT_TOKEN` - Your bot token
- `ADMIN_ID` - Your Telegram ID
- `CHANNEL_ID` - Optional channel ID
