# pip install list : pickle, yfinance, sklearn, FinanceDataReader, ta, xgboost

import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import yfinance as yf
import FinanceDataReader as fdr
import ta
import numpy as np

# 상대 강도 지수 (RSI) 계산
def calculate_rsi(data, period=14):
    diff = data['Close'].diff()
    up, down = diff.copy(), diff.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    avg_gain = up.rolling(window=period).mean()
    avg_loss = abs(down.rolling(window=period).mean())
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 스토캐스틱 %K 계산
def calculate_stochastic_k(data, k_period=14):
    low_min = data['Low'].rolling(window=k_period).min()
    high_max = data['High'].rolling(window=k_period).max()
    stoch_k = ((data['Close'] - low_min) / (high_max - low_min)) * 100
    return stoch_k

# 커모디티 채널 인덱스 (CCI) 계산
def calculate_cci(data, cci_period=20):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    sma_typical_price = typical_price.rolling(window=cci_period).mean()
    mean_deviation = abs(typical_price - sma_typical_price).rolling(window=cci_period).mean()
    cci = (typical_price - sma_typical_price) / (0.015 * mean_deviation)
    return cci

# 애버리지 디렉셔널 인덱스 (ADX) 계산
def calculate_adx(data, period=14):
    tr = pd.DataFrame()
    tr['H-L'] = data['High'] - data['Low']
    tr['H-P'] = abs(data['High'] - data['Close'].shift(1))
    tr['L-P'] = abs(data['Low'] - data['Close'].shift(1))
    tr['TR'] = tr[['H-L', 'H-P', 'L-P']].max(axis=1)
    atr = tr['TR'].rolling(window=period).mean()
    tr['+DM'] = np.where((data['High'] - data['High'].shift(1)) > (data['Low'].shift(1) - data['Low']), (data['High'] - data['High'].shift(1)), 0)
    tr['-DM'] = np.where((data['Low'].shift(1) - data['Low']) > (data['High'] - data['High'].shift(1)), (data['Low'].shift(1) - data['Low']), 0)
    tr['+DI'] = (tr['+DM'].rolling(window=period).sum() / atr) * 100
    tr['-DI'] = (tr['-DM'].rolling(window=period).sum() / atr) * 100
    tr['DX'] = (abs(tr['+DI'] - tr['-DI']) / (tr['+DI'] + tr['-DI'])) * 100
    adx = tr['DX'].rolling(window=period).mean()
    return adx

# 오썸 오실레이터 계산
def calculate_awesome_oscillator(data, short_period=5, long_period=34):
    sma_short = data['Close'].rolling(window=short_period).mean()
    sma_long = data['Close'].rolling(window=long_period).mean()
    awesome_oscillator = sma_short - sma_long
    return awesome_oscillator

# 모멘텀 계산
def calculate_momentum(data, period=10):
    momentum = data['Close'].diff(period)
    return momentum

# MACD 계산
def calculate_macd(data, short_period=12, long_period=26):
    short_ema = data['Close'].ewm(span=short_period, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_period, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    return macd_histogram

# 스토캐스틱 RSI 패스트 계산
def calculate_stochastic_rsi_fast(data, rsi_period=14, stoch_period=14):
    rsi = calculate_rsi(data, period=rsi_period)
    stoch_rsi_fast = (rsi - rsi.rolling(window=stoch_period).min()) / (rsi.rolling(window=stoch_period).max() - rsi.rolling(window=stoch_period).min()) * 100
    return stoch_rsi_fast

# 윌리엄스 퍼센트 레인지 (W%R) 계산
def calculate_williams_percent_range(data, period=14):
    highest_high = data['High'].rolling(window=period).max()
    lowest_low = data['Low'].rolling(window=period).min()
    williams_percent_range = ((highest_high - data['Close']) / (highest_high - lowest_low)) * -100
    return williams_percent_range

# 불 베어 파워 계산
def calculate_bull_bear_power(data, period=13):
    bull_power = data['High'] - data['Low'].rolling(window=period).mean()
    bear_power = data['Low'] - data['High'].rolling(window=period).mean()
    return bull_power, bear_power

# 얼티미트 오실레이터 계산
def calculate_ultimate_oscillator(data, period1=7, period2=14, period3=28):
    tr = pd.DataFrame()
    tr['H-L'] = data['High'] - data['Low']
    tr['H-P'] = abs(data['High'] - data['Close'].shift(1))
    tr['L-P'] = abs(data['Low'] - data['Close'].shift(1))
    tr['BP'] = data['Close'] - data['Low']
    tr['TR'] = tr[['H-L', 'H-P', 'L-P']].max(axis=1)
    avg7 = tr['BP'].rolling(window=period1).sum() / tr['TR'].rolling(window=period1).sum()
    avg14 = tr['BP'].rolling(window=period2).sum() / tr['TR'].rolling(window=period2).sum()
    avg28 = tr['BP'].rolling(window=period3).sum() / tr['TR'].rolling(window=period3).sum()
    ultimate_oscillator = 100 * ((4 * avg7) + (2 * avg14) + avg28) / 7
    return ultimate_oscillator



# 이동 평균 계산 함수
def calculate_moving_average(data, window, type='simple'):
    if type == 'simple':
        return data['Close'].rolling(window=window).mean()
    elif type == 'exponential':
        return data['Close'].ewm(span=window, adjust=False).mean()
    elif type == 'volume_weighted':
        return (data['Close'] * data['Volume']).rolling(window=window).sum() / data['Volume'].rolling(window=window).sum()

# 일목 표준선 계산 함수
def calculate_ichimoku_baseline(data, short_period=9, long_period=26, baseline_period=52, lagging_period=26):
    short_conversion = (data['High'].rolling(window=short_period).max() + data['Low'].rolling(window=short_period).min()) / 2
    long_base = (data['High'].rolling(window=long_period).max() + data['Low'].rolling(window=long_period).min()) / 2
    ichimoku_baseline = (short_conversion + long_base) / 2
    leading_span_a = ((short_conversion + long_base) / 2).shift(lagging_period)
    leading_span_b = ((data['High'].rolling(window=baseline_period).max() + data['Low'].rolling(window=baseline_period).min()) / 2).shift(lagging_period)
    return ichimoku_baseline, leading_span_a, leading_span_b

#피봇
def calculate_pivot_points(data):
    high = data['High']
    low = data['Low']
    close = data['Close']

    pivot = (high + low + close) / 3
    support1 = 2 * pivot - high
    resistance1 = 2 * pivot - low
    support2 = pivot - (high - low)
    resistance2 = pivot + (high - low)

    return pivot, support1, resistance1, support2, resistance2


#피봇 우디
def calculate_woodie_pivot(data):

    high = data['High']
    low = data['Low']
    close = data['Close']

    pivot = (high + low + 2 * close) / 4
    support1 = 2 * pivot - high
    resistance1 = 2 * pivot - low
    support2 = pivot - (high - low)
    resistance2 = pivot + (high - low)
    return pivot, support1, resistance1, support2, resistance2


df_nasdaq = fdr.StockListing('NASDAQ')
nasdaq_list= df_nasdaq['Symbol'].tolist()

"""데이터 총 합산"""

result_df=pd.DataFrame()

len(nasdaq_list)

num=1
err_list=[]
for i in nasdaq_list:
  try:
    print(num,i)
    stock_data=yf.download(i,start = '2012-01-01')
    # 주어진 항목들에 대한 계산 함수 호출
    stock_data['symbol'] = i
    stock_data['RSI'] = calculate_rsi(stock_data)
    stock_data['Stochastic %K'] = calculate_stochastic_k(stock_data)
    stock_data['CCI'] = calculate_cci(stock_data)
    stock_data['ADX'] = calculate_adx(stock_data)
    stock_data['Awesome Oscillator'] = calculate_awesome_oscillator(stock_data)
    stock_data['Momentum'] = calculate_momentum(stock_data)
    stock_data['MACD'] = calculate_macd(stock_data)
    stock_data['Stochastic RSI Fast'] = calculate_stochastic_rsi_fast(stock_data)
    stock_data['Williams %R'] = calculate_williams_percent_range(stock_data)
    bull_power, bear_power = calculate_bull_bear_power(stock_data)
    stock_data['Bull Power'] = bull_power
    stock_data['Bear Power'] = bear_power
    stock_data['Ultimate Oscillator'] = calculate_ultimate_oscillator(stock_data)
    # 주어진 항목들에 대한 이동 평균 계산 함수 호출 및 데이터프레임에 저장
    for window in [10, 20, 30, 50, 100, 200]:
        stock_data['Simple Moving Average ({})'.format(window)] = calculate_moving_average(stock_data, window=window, type='simple')
        stock_data['Exponential Moving Average ({})'.format(window)] = calculate_moving_average(stock_data, window=window, type='exponential')
        stock_data['Volume Weighted Moving Average ({})'.format(window)] = calculate_moving_average(stock_data, window=window, type='volume_weighted')
    stock_data['Volume Weighted Moving Average (20)'] = calculate_moving_average(stock_data, window=20, type='volume_weighted')
    #일목 표준선
    stock_data['ichimoku_baseline'], stock_data['leading_span_a'], stock_data['leading_span_b'] = calculate_ichimoku_baseline(stock_data)
    #피봇 계산
    stock_data['pivot'], stock_data['support1'], stock_data['resistance1'], stock_data['support2'], stock_data['resistance2'] = calculate_pivot_points(stock_data)

    # 우디 피봇 포인트 계산
    stock_data['woddie_pivot'], stock_data['woddie_support1'], stock_data['woddie_resistance1'], stock_data['woddie_support2'], stock_data['woddie_resistance2'] = calculate_woodie_pivot(stock_data)
    stock_data['symbol_encoded'] = num
    per=[]
    for i in range(0,len(stock_data['Open'])-14):
        persent=[]
        for j in range(0,15):
            a = ((stock_data['Close'][i+j] - stock_data['Open'][i])/stock_data['Open'][i])*100
            persent.append(a)
        if max(persent) >= 7 and min(persent) > -5 :
            per.append(1)
        elif max(persent) >= 7 and min(persent) <= -5:
            per.append(2)
        elif max(persent) < 7  and min(persent) > -5:
            per.append(3)
        else:
            per.append(0)
    stock_data = stock_data.iloc[:-14]
    stock_data['label'] = per
    df = stock_data.dropna()
    result_df = pd.concat([result_df,df])
    num += 1
    print(result_df.shape)

  except IndexError as e:
    print(e)
    err_list.append(e)
    pass




# result_df.to_csv('rawdata7.csv')

nan_count_per_column = result_df.isnull().sum()
label_counts = result_df['label'].value_counts()
nan_count_per_column

print(nan_count_per_column)
print(label_counts)

# 데이터 준비
print('# 데이터 준비')
stock = result_df
del stock['symbol']

column = stock.columns
print(column[1:52])




X = stock[column[1:51]].values
y = stock['label']

# 데이터를 학습용과 테스트용으로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost 데이터셋 생성
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# XGBoost 모델 설정
params = {
    'objective': 'multi:softmax',  # 다중 분류 문제 설정
    'num_class': 4,  # 클래스의 수
    'max_depth': 10,   # 트리의 최대 깊이
    'eta': 0.005,      # 학습률
    'subsample': 0.8, # 데이터 샘플링 비율
    'colsample_bytree': 0.9, # 트리마다 사용할 특성 비율
    'eval_metric': 'mlogloss',  # 평가 지표3
}

# XGBoost 모델 훈련
num_round = 50000 # 훈련 라운드 수
bst = xgb.train(params, dtrain, num_round, evals=[(dtest, "test")], early_stopping_rounds=50)

# 테스트 데이터에 대한 예측 수행
y_pred = bst.predict(dtest)
y_pred_class = [int(x) for x in y_pred]

# 정확도 및 분류 리포트 출력
accuracy = accuracy_score(y_test, y_pred_class)
print(f"Accuracy: {accuracy:.2f}")

report = classification_report(y_test, y_pred_class)
print("Classification Report:\n", report)

# 모델 저장
model_filename = "5man_0.005_xgboost_multiclass_model.pkl"
with open(model_filename, 'wb') as model_file:
    pickle.dump(bst, model_file)
print(f"Model saved as {model_filename}")

# 저장된 모델 로드
with open(model_filename, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# 로드한 모델을 사용하여 예측 수행
loaded_y_pred = loaded_model.predict(dtest)
loaded_y_pred_class = [int(x) for x in loaded_y_pred]

# 로드한 모델의 정확도 확인
loaded_accuracy = accuracy_score(y_test, loaded_y_pred_class)
print(f"Loaded Model Accuracy: {loaded_accuracy:.2f}")

