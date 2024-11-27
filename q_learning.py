import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers

class DQNAgent:
    def __init__(self, n_actions, n_features):
        self.n_actions = n_actions  # 행동의 수
        self.n_features = n_features  # 상태의 특성 수
        self.model = self._build_model()  # 모델 빌드

    def _build_model(self):
        """Q-learning 모델을 구성합니다."""
        model = models.Sequential([
            layers.Input(shape=(self.n_features,)),  # 첫 번째 레이어에서 Input 객체 사용
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.n_actions, activation='linear')  # 행동의 수만큼 출력
        ])
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')  # 수정된 부분
        return model

    def act(self, state):
        """주어진 상태에서 가장 좋은 행동을 선택합니다."""
        state = np.array(state).reshape(1, -1)  # 상태를 모델에 입력할 수 있도록 reshape
        q_values = self.model.predict(state)  # Q값 예측
        print("Q-values:", q_values)  # Q-values 확인

        action = np.argmax(q_values[0])  # 가장 높은 Q값을 가진 행동 선택
        if action == 0:
            return 'buy'   # 매수
        elif action == 1:
            return 'sell'  # 매도
        else:
            return 'hold'  # 대기
    
    def train(self, state, action, reward, next_state):
        """Q-learning 알고리즘을 이용해 모델을 학습합니다."""
        state = np.array(state).reshape(1, -1)
        next_state = np.array(next_state).reshape(1, -1)

        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        # Q-learning 업데이트: Q(s, a) = r + γ * max(Q(s', a'))
        if action == 'buy':
            action_index = 0
        elif action == 'sell':
            action_index = 1
        else:
            action_index = 2

        # 목표 Q 값 계산
        target[0][action_index] = reward + 0.95 * np.max(target_next)  # 0.95는 감가율 γ

        self.model.fit(state, target, epochs=1, verbose=0)  # 모델 학습

    def save(self, filename):
        """모델을 저장합니다."""
        self.model.save(filename)

    def load(self, filename):
        """저장된 모델을 로드합니다."""
        self.model = models.load_model(filename)
