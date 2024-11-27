import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

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
