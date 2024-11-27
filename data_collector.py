import pyupbit
import pandas as pd
import time

# 업비트 API 객체 생성 (본인의 키를 입력)
access_key = "your_access_key"
secret_key = "your_secret_key"
upbit = pyupbit.Upbit(access_key, secret_key)

# 데이터를 수집하는 함수
def collect_data():
    # 비트코인 가격 데이터 가져오기 (1분 봉)
    df = pyupbit.get_ohlcv("KRW-BTC", interval="minute1", count=200)

    # 이동평균 (MA3, MA5 계산)
    df['MA3'] = df['close'].rolling(window=3).mean()
    df['MA5'] = df['close'].rolling(window=5).mean()

    # RSI 계산 (14일 기준)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # CSV로 저장
    df.to_csv("historical_data.csv")

# 무한 루프에서 데이터를 3분마다 수집
while True:
    collect_data()
    time.sleep(180)  # 3분마다 실행
