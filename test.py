import pandas as pd
from q_learning import QLearningAgent  # Q-Learning 에이전트를 import 합니다.
from data_collector import collect_data  # 데이터를 수집하는 모듈을 import 합니다.

# 훈련된 Q-Learning 에이전트 초기화
actions = ["buy", "sell", "hold"]
agent = QLearningAgent(actions)

# 훈련한 Q-값 로드
# agent.load_q_table('q_table.csv')  # 훈련 후 저장된 Q-테이블을 로드하는 방법 (선택사항)

# 데이터 로드 (테스트할 데이터)
df = pd.read_csv("historical_data.csv")

# 테스트 파라미터 설정
initial_balance = 1000000  # 초기 자본 (예: 100만 원)
balance = initial_balance  # 현재 자본
btc_balance = 0  # 보유한 비트코인 양

# 테스트 시작
print("테스트 시작")
for i in range(1, len(df)):
    state = (df['close'][i-1], df['ma3'][i-1], df['ma5'][i-1], df['rsi'][i-1], df['macd'][i-1])
    action = agent.choose_action(state)  # 훈련된 에이전트로 행동 선택

    current_price = df['close'][i]

    if action == "buy" and balance >= current_price:  # 매수
        btc_balance += balance // current_price
        balance -= btc_balance * current_price
        print(f"매수: {current_price} 원, 잔액: {balance} 원, 비트코인 잔고: {btc_balance} BTC")

    elif action == "sell" and btc_balance > 0:  # 매도
        balance += btc_balance * current_price
        btc_balance = 0
        print(f"매도: {current_price} 원, 잔액: {balance} 원, 비트코인 잔고: {btc_balance} BTC")

    else:  # hold (기다림)
        print(f"기다림: 현재 가격 {current_price} 원, 잔액: {balance} 원, 비트코인 잔고: {btc_balance} BTC")

# 테스트 종료 후 최종 자산 출력
print(f"테스트 종료 후 최종 자산: {balance + btc_balance * current_price} 원")
