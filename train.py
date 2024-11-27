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
    
    # 수수료 제외 금액 계산
    available_balance_after_fee = available_balance * (1 - fee_rate)
    
    if action == 'buy':
        if available_balance_after_fee > 5000:  # 최소 5000원 이상일 경우만 구매
            buy_amount = available_balance_after_fee
            upbit.buy_market_order("KRW-BTC", buy_amount)
            print(f"[BUY] Buy order executed, amount: {buy_amount} KRW (after fee)")
            return available_balance_after_fee
        else:
            print("[BUY] Not enough balance to buy.")
            return 0
    
    elif action == 'sell':
        btc_balance = upbit.get_balance("KRW-BTC")
        if btc_balance > 0:
            upbit.sell_market_order("KRW-BTC", btc_balance)
            print(f"[SELL] Sell order executed, amount: {btc_balance} BTC")
            return btc_balance
        else:
            print("[SELL] No Bitcoin to sell.")
            return 0
    
    elif action == 'hold':
        print("[HOLD] Holding the position.")
        return available_balance
    
    else:
        print("[ERROR] No action taken.")
        return available_balance

# DQN 모델 정의
class DQNAgent:
    def __init__(self, n_actions, n_features, epsilon=1.0, epsilon_min=0.9, epsilon_decay=0.999):
        self.n_actions = n_actions
        self.n_features = n_features
        self.model = self._build_model()
        self.epsilon = epsilon  # 탐색률
        self.epsilon_min = epsilon_min  # 최소 탐색률
        self.epsilon_decay = epsilon_decay  # 탐색률 감소 비율

    def _build_model(self):
        model = models.Sequential([
            layers.Input(shape=(self.n_features,)),  # 첫 번째 레이어에서 Input 사용
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.n_actions, activation='linear')
        ])
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')
        return model

    def act(self, state):
        # epsilon-greedy 방식으로 액션 선택
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(['buy', 'sell', 'hold'])  # 탐색: 무작위로 선택
            print(f"[ACT] Action (explore): {action}")
        else:
            state = np.array(state).reshape(1, -1)
            q_values = self.model.predict(state)
            action = np.argmax(q_values[0])  # 최대 Q값에 해당하는 액션 선택
            action = ['buy', 'sell', 'hold'][action]  # 예측된 액션 선택
            print(f"[ACT] Action (exploit): {action}")
        
        # epsilon 값을 감소시키지 않고 일정 수준으로 유지 (탐색 비율을 높게 유지)
        if self.epsilon > self.epsilon_min:
            print(f"[TRAIN] Epsilon remains at {self.epsilon} for more exploration.")
        
        return action

    def train(self, state, action, reward, next_state):
        state = np.array(state).reshape(1, -1)
        next_state = np.array(next_state).reshape(1, -1)
        
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)
        
        action_index = ['buy', 'sell', 'hold'].index(action)
        target[0][action_index] = reward + 0.95 * np.max(target_next)  # Q-값 업데이트
        
        self.model.fit(state, target, epochs=1, verbose=0)
        
        # epsilon을 점차 줄이지 않고 일정 수준에서 유지
        if self.epsilon > self.epsilon_min:
            print(f"[TRAIN] Epsilon remains at {self.epsilon}.")

# 손익 계산 함수
def calculate_profit(initial_balance, final_balance):
    return final_balance - initial_balance

# 데이터 수집 함수
def collect_data():
    df = pyupbit.get_ohlcv("KRW-BTC", interval="minute1", count=200)
    df['MA3'] = df['close'].rolling(window=3).mean()
    df['MA5'] = df['close'].rolling(window=5).mean()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df.to_csv("historical_data.csv")
    print("[DATA COLLECT] Data collected and saved.")

# 실시간 거래 함수
def real_time_trade(agent):  # agent를 매개변수로 받도록 수정
    try:
        krw_balance = upbit.get_balance("KRW")
        if krw_balance < 5000:
            print("[TRADE] 잔고가 부족합니다. 거래를 진행할 수 없습니다.")
            return
    except Exception as e:
        print(f"[ERROR] Error: {str(e)}")
        return

    initial_balance = krw_balance  # 거래 전 잔고 기록
    print(f"[TRADE] Executing real-time trade... Balance: {initial_balance} KRW")
    
    # 모델 예측
    state = [0.0, 0.5, 0.8]  # 예시 상태 (이 부분은 실제 데이터에 맞게 업데이트 필요)
    action = agent.act(state)
    
    # 거래 실행
    final_balance = execute_trade(action, krw_balance)

    # 손익 계산 및 출력
    profit = calculate_profit(initial_balance, final_balance)
    print(f"[TRADE] Profit/Loss: {profit} KRW")

    # 학습 (실제 상황에서는 보상과 상태를 반영하여 학습)
    reward = profit  # 보상은 손익으로 설정
    next_state = [0.1, 0.6, 0.9]  # 예시 (다음 상태는 실제 데이터에 따라 달라짐)
    agent.train(state, action, reward, next_state)

# DQNAgent 객체 생성
agent = DQNAgent(n_actions=3, n_features=3)  # 예시로 3개의 행동과 3개의 특징 사용

# 데이터 수집을 3분마다 실행
schedule.every(3).minutes.do(collect_data)

# 실시간 거래를 3분마다 실행
schedule.every(3).minutes.do(real_time_trade, agent=agent)

# 무한 루프를 돌며 주기적으로 실행
while True:
    schedule.run_pending()
    time.sleep(1)
