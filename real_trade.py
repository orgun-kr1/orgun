import pyupbit
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
import time
import schedule
from dotenv import load_dotenv
import os
import pandas as pd

# .env 파일에서 API 키 로드
load_dotenv()
access_key = os.getenv("UPBIT_ACCESS_KEY")
secret_key = os.getenv("UPBIT_SECRET_KEY")
upbit = pyupbit.Upbit(access_key, secret_key)

# 거래 실행 함수
def execute_trade(action, available_balance):
    fee_rate = 0.0005  # 업비트 거래 수수료
    available_balance_after_fee = available_balance * (1 - fee_rate)
    
    if action == 'buy':
        if available_balance_after_fee >= 5000:  # 최소 5000원 이상일 경우만 구매
            buy_amount = available_balance_after_fee
            upbit.buy_market_order("KRW-BTC", buy_amount)
            print(f"Buy order executed, amount: {buy_amount} KRW (after fee)")
            return available_balance_after_fee
        else:
            return available_balance  # 매수 시도 안 함

    elif action == 'sell':
        btc_balance = upbit.get_balance("KRW-BTC")
        if btc_balance > 0:
            upbit.sell_market_order("KRW-BTC", btc_balance)
            print(f"Sell order executed, amount: {btc_balance} BTC")
            return 0  # 판매 후 KRW 잔고로 돌아옴
        else:
            return available_balance  # 판매 시도 안 함
    
    elif action == 'hold':
        print("Holding the position.")
        return available_balance
    
    else:
        return available_balance

# DuelingDQNAgent 모델 정의
class DuelingDQNAgent:
    def __init__(self, n_actions, n_features):
        self.n_actions = n_actions  # 행동의 수
        self.n_features = n_features  # 상태의 특성 수
        self.model = self._build_model()

    def _build_model(self):
        """Dueling DQN 모델 생성"""
        state_input = layers.Input(shape=(self.n_features,))
        x = layers.Dense(64, activation='relu')(state_input)
        x = layers.Dense(64, activation='relu')(x)

        value = layers.Dense(1, activation='linear')(x)  # 상태의 가치
        advantage = layers.Dense(self.n_actions, activation='linear')(x)  # 행동에 대한 우위

        # Keras 레이어 내에서 텐서 연산을 처리
        advantage_mean = layers.Lambda(lambda a: tf.reduce_mean(a, axis=1, keepdims=True))(advantage)
        q_values = layers.Add()([value, advantage - advantage_mean])

        model = models.Model(inputs=state_input, outputs=q_values)
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')
        return model

    def act(self, state, epsilon=0.1):
        """epsilon-greedy 정책으로 행동 선택"""
        if np.random.rand() <= epsilon:
            return np.random.choice(['buy', 'sell', 'hold'])
        state = np.array(state).reshape(1, -1)
        q_values = self.model.predict(state)
        return ['buy', 'sell', 'hold'][np.argmax(q_values[0])]

    def train(self, state, action, reward, next_state, gamma=0.95):
        """Q-learning 기반 학습"""
        state = np.array(state).reshape(1, -1)
        next_state = np.array(next_state).reshape(1, -1)

        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        action_index = ['buy', 'sell', 'hold'].index(action)
        target[0][action_index] = reward + gamma * np.max(target_next[0])

        self.model.fit(state, target, epochs=1, verbose=0)

    def save(self, filename):
        """모델 저장"""
        self.model.save(filename)

    def load(self, filename):
        """저장된 모델 로드"""
        self.model = models.load_model(filename)

# 손익 계산 함수
def calculate_profit(initial_balance, final_balance):
    return final_balance - initial_balance

# 수익률 계산 함수
def calculate_profit_percentage(initial_balance, final_balance):
    return ((final_balance - initial_balance) / initial_balance) * 100

# 데이터 수집 함수
def collect_data():
    df = pyupbit.get_ohlcv("KRW-BTC", interval="minute1", count=200)
    df['MA3'] = df['close'].rolling(window=3).mean()
    df['MA5'] = df['close'].rolling(window=5).mean()
    
    # MACD 추가
    df['12_EMA'] = df['close'].ewm(span=12, adjust=False).mean()
    df['26_EMA'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['12_EMA'] - df['26_EMA']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 볼린저 밴드 추가
    df['Middle_Band'] = df['close'].rolling(window=20).mean()
    df['Upper_Band'] = df['Middle_Band'] + 2 * df['close'].rolling(window=20).std()
    df['Lower_Band'] = df['Middle_Band'] - 2 * df['close'].rolling(window=20).std()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df.to_csv("historical_data.csv")

# 실시간 거래 함수
def real_time_trade(agent):  
    try:
        krw_balance = upbit.get_balance("KRW")
        if krw_balance < 5000:  # 잔고가 부족하면 매수하지 않음
            return
    except Exception as e:
        print(f"Error: {str(e)}")
        return

    initial_balance = krw_balance  
    print(f"Executing real-time trade... Balance: {initial_balance} KRW")
    
    # 모델 예측
    state = [0.0, 0.5, 0.8, 0.1, 0.5, 0.3, 0.7]  # 7개 특징
    action = agent.act(state)
    
    # 거래 실행
    final_balance = execute_trade(action, krw_balance)

    # 손익 계산 및 출력 (매수 시에는 손실을 학습하지 않음)
    if action != 'buy':  # 매수 외의 행동일 때만 손익 계산
        profit = calculate_profit(initial_balance, final_balance)
        profit_percentage = calculate_profit_percentage(initial_balance, final_balance)
        print(f"Profit/Loss: {profit} KRW")
        print(f"Profit Percentage: {profit_percentage}%")
    
    reward = profit if action != 'buy' else 0  # 매수 시 손실을 보지 않도록 reward를 0으로 설정
    next_state = [0.1, 0.6, 0.9, 0.2, 0.6, 0.4, 0.8]  # 7개 특징
    agent.train(state, action, reward, next_state)

# DuelingDQNAgent 객체 생성
agent = DuelingDQNAgent(n_actions=3, n_features=7)  # 7개의 특징을 사용

# 데이터 수집을 1초마다 실행
schedule.every(1).seconds.do(collect_data)

# 무한 루프를 돌며 주기적으로 실행
while True:
    schedule.run_pending()
    time.sleep(1)
