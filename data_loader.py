# data_loader.py
import ccxt
import pandas as pd

class DataLoader:
    def __init__(self, exchange_name='upbit'):
        self.exchange = ccxt.upbit({
            'enableRateLimit': True,  # 요청 속도 제한
        })

    def get_historical_data(self, symbol='BTC/USDT', timeframe='1m', limit=100):
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        return data

# 사용 예시
if __name__ == '__main__':
    loader = DataLoader()
    symbol = 'BTC/USDT'
    data = loader.get_historical_data(symbol, '1m', limit=100)
    print(data.tail())
