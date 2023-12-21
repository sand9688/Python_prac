import numpy as np
import pandas as pd
import yfinance as yf
import ta
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random

# 주식 데이터 불러오기 (여러 종목)
tickers = ["AAPL", "MSFT", "GOOGL"]  # 원하는 종목들의 심볼을 리스트로 지정
start_date = "2023-01-01"
end_date = "2023-01-31"
data = {}  # 종목별 데이터를 담을 딕셔너리 생성

for ticker in tickers:
    data[ticker] = yf.download(ticker, start=start_date, end=end_date)

# 보조 지표 계산 (이동평균선)
for ticker in tickers:
    data[ticker]["MA"] = ta.trend.SMAIndicator(data[ticker]["Close"], window=20).sma_indicator()

# 강화학습 모델 클래스
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # 할인율
        self.epsilon = 1.0   # 탐험 확률
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 강화학습 훈련
def train_dqn_agent(data_dict):
    state_size = 2    # 주식 종가 및 이동평균선으로 상태 공간 크기 설정
    action_size = 3   # 액션 공간 크기 (사기, 팔기, 보유)
    batch_size = 32
    agent = DQNAgent(state_size, action_size)

    for ticker in data_dict:
        for t in range(1, len(data_dict[ticker]) - 1):
            state = np.array([[data_dict[ticker]["Close"][t], data_dict[ticker]["MA"][t]]])
            next_state = np.array([[data_dict[ticker]["Close"][t + 1], data_dict[ticker]["MA"][t + 1]]])
            done = True if t == len(data_dict[ticker]) - 2 else False

            # 강화학습 모델 학습
            action = agent.act(state)
            reward = 0
            if action == 0:  # 사기
                reward = (data_dict[ticker]["Close"][t + 1] - data_dict[ticker]["Close"][t]) / data_dict[ticker]["Close"][t]
            elif action == 1:  # 팔기
                reward = (data_dict[ticker]["Close"][t] - data_dict[ticker]["Close"][t + 1]) / data_dict[ticker]["Close"][t]
            else:  # 보유
                reward = 0

            agent.remember(state, action, reward, next_state, done)
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

    return agent

# 주식 거래 시뮬레이션
def simulate_trading(agent, data_dict):
    state_size = 2
    action_size = 3
    total_profit = 0
    inventory = {}

    # 초기 보유량 설정
    for ticker in data_dict:
        inventory[ticker] = []

    for ticker in data_dict:
        for t in range(1, len(data_dict[ticker])):
            state = np.array([[data_dict[ticker]["Close"][t], data_dict[ticker]["MA"][t]]])
            action = agent.act(state)
            if action == 0:  # 사기
                inventory[ticker].append(data_dict[ticker]["Close"][t])
                print(f"Buy {ticker} at {data_dict[ticker].index[t]} - Price: {data_dict[ticker]['Close'][t]}")
            elif action == 1 and len(inventory[ticker]) > 0:  # 팔기
                buy_price = inventory[ticker].pop(0)
                profit = (data_dict[ticker]["Close"][t] - buy_price) / buy_price
                total_profit += profit
                print(f"Sell {ticker} at {data_dict[ticker].index[t]} - Price: {data_dict[ticker]['Close'][t]} - Profit: {profit}")

    return total_profit

# 주식 데이터 및 보조 지표 로드
# (데이터 전처리, 보조 지표 계산 등이 필요할 수 있습니다.)

# 강화학습 모델 훈련
trained_agent = train_dqn_agent(data)

# 훈련된 강화학습 모델로 주식 거래 시뮬레이션 수행
total_profit = simulate_trading(trained_agent, data)
print(f"Total Profit: {total_profit}")
