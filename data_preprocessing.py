# data_preprocessing.py
import pandas as pd
import numpy as np

# CSV 파일에서 데이터 로드
def load_data(file_path="historical_data.csv"):
    df = pd.read_csv(file_path, index_col="timestamp", parse_dates=True)
    return df

# 상태(state)를 정의하는 함수 (가격, 이동평균, RSI, MACD 등을 포함)
def get_state(df, index):
    state = []
    state.append(df['close'].iloc[index])  # 현재 가격
    state.append(df['MA3'].iloc[index])  # 3일 이동 평균
    state.append(df['MA5'].iloc[index])  # 5일 이동 평균
    state.append(df['RSI'].iloc[index])  # RSI
    state.append(df['MACD'].iloc[index])  # MACD
    state.append(df['MACD_signal'].iloc[index])  # MACD 시그널 라인
    return np.array(state)

# 학습 데이터를 준비하는 함수
def prepare_data(df):
    states = []
    actions = []
    rewards = []
    next_states = []

    for i in range(len(df) - 1):  # 마지막 데이터는 다음 상태가 없으므로 제외
        state = get_state(df, i)
        next_state = get_state(df, i + 1)
        
        # 예시: 현재 시점에서 가격 상승/하락을 기준으로 보상 설정
        reward = df['close'].iloc[i + 1] - df['close'].iloc[i]
        
        # 상태, 행동, 보상, 다음 상태 저장
        states.append(state)
        rewards.append(reward)
        next_states.append(next_state)
        
        # 행동을 예시로 단순히 'buy', 'sell', 'hold'로 결정 (여기선 그냥 보상에 따라 결정)
        action = "buy" if reward > 0 else "sell"  # 간단한 예시: 가격 상승은 매수, 하락은 매도
        actions.append(action)

    return np.array(states), np.array(actions), np.array(rewards), np.array(next_states)
