# main.py
from upbit_api import place_order
from q_learning import DQNAgent
from data_preprocessing import load_data, prepare_data
import numpy as np

def execute_trading():
    """자동 매매 함수"""
    # 데이터 로드 및 준비
    df = load_data("historical_data.csv")
    states, actions, rewards, next_states = prepare_data(df)

    # Q-learning 모델 초기화
    n_actions = 3  # 'buy', 'sell', 'hold' 3가지 행동
    n_features = len(states[0])  # 상태의 특성 수 (가격, MA, RSI 등)
    agent = DQNAgent(n_actions, n_features)

    for i in range(len(states)):
        state = states[i]
        action = agent.act(state)  # 행동 선택 (buy, sell, hold)
        reward = rewards[i]
        next_state = next_states[i]

        # Q-learning 학습
        agent.train(state, action, reward, next_state)

        # 선택된 행동을 실행
        if action == 'buy':
            order_response = place_order("buy", "KRW-BTC", price=50000000, volume=0.001)  # 예시 매수
            print("매수 주문 결과:", order_response)
        elif action == 'sell':
            order_response = place_order("sell", "KRW-BTC", price=60000000, volume=0.001)  # 예시 매도
            print("매도 주문 결과:", order_response)
        elif action == 'hold':
            print("홀딩 중...")

if __name__ == "__main__":
    execute_trading()
