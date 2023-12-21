import joblib
import yfinance as yf
import FinanceDataReader as fdr
import ta
from datetime import datetime, timedelta
import time
import pandas as pd
import numpy as np


start_time = time.time()


#입력데이터 준비

df_nasdaq = fdr.StockListing('NASDAQ')
nasdaq_list= df_nasdaq['Symbol'].tolist()

# 현재 날짜 가져오기
current_date = datetime.now()

# 2년 전 날짜 계산
two_year_ago = current_date - timedelta(days=730)

#모델설정
model = joblib.load('모델/best_decision_tree_model.joblib')

# decision_tree_model.joblib





#input_df만들기
input_df=pd.DataFrame()

#df_low


num=1
err_list=[]
input_list=[]
buy_list = []
buy_signal = 0

for i in nasdaq_list:
    try:
        df=yf.download(i,start = two_year_ago , end = '2023-08-01')

        df = df.reset_index()

         # 이동 평균선
        ma = [20,60,120,240,480]
        for days in ma:
            df['ma_'+str(days)] = df['Close'].rolling(window = days).mean()
        # 그외 각종 지표들
        H, L, C, V = df['High'], df['Low'], df['Close'], df['Volume']
        # stock_df['bol_high'] = ta.volatility.bollinger_hband(C)
        # stock_df['bol_low']  = ta.volatility.bollinger_lband(C)
        df['MFI'] = ta.volume.money_flow_index(
            high=H, low=L, close=C, volume=V, fillna=True)

        df['ADI'] = ta.volume.acc_dist_index(
            high=H, low=L, close=C, volume=V, fillna=True)

        df['OBV'] = ta.volume.on_balance_volume(close=C, volume=V, fillna=True)
        df['CMF'] = ta.volume.chaikin_money_flow(
            high=H, low=L, close=C, volume=V, fillna=True)

        df['FI'] = ta.volume.force_index(close=C, volume=V, fillna=True)
        df['EOM, EMV'] = ta.volume.ease_of_movement(
            high=H, low=L, volume=V, fillna=True)

        df['VPT'] = ta.volume.volume_price_trend(close=C, volume=V, fillna=True)
        df['NVI'] = ta.volume.negative_volume_index(close=C, volume=V, fillna=True)
        df['VMAP'] = ta.volume.volume_weighted_average_price(
            high=H, low=L, close=C, volume=V, fillna=True)

        # Volatility
        df['ATR'] = ta.volatility.average_true_range(
            high=H, low=L, close=C, fillna=True)
        df['BHB'] = ta.volatility.bollinger_hband(close=C, fillna=True)
        df['BLB'] = ta.volatility.bollinger_lband(close=C, fillna=True)
        df['KCH'] = ta.volatility.keltner_channel_hband(
            high=H, low=L, close=C, fillna=True)
        df['KCL'] = ta.volatility.keltner_channel_lband(
            high=H, low=L, close=C, fillna=True)
        df['KCM'] = ta.volatility.keltner_channel_mband(
            high=H, low=L, close=C, fillna=True)
        df['DCH'] = ta.volatility.donchian_channel_hband(
            high=H, low=L, close=C, fillna=True)
        df['DCL'] = ta.volatility.donchian_channel_lband(
            high=H, low=L, close=C, fillna=True)
        df['DCM'] = ta.volatility.donchian_channel_mband(
            high=H, low=L, close=C, fillna=True)
        df_nasdaq['UI'] = ta.volatility.ulcer_index(close=C, fillna=True)
        # Trend
        df['SMA'] = ta.trend.sma_indicator(close=C, fillna=True)
        df['EMA'] = ta.trend.ema_indicator(close=C, fillna=True)
        df['WMA'] = ta.trend.wma_indicator(close=C, fillna=True)
        df['MACD'] = ta.trend.macd(close=C, fillna=True)
        df['ADX'] = ta.trend.adx(high=H, low=L, close=C, fillna=True)
        df['-VI'] = ta.trend.vortex_indicator_neg(
            high=H, low=L, close=C, fillna=True)
        df['+VI'] = ta.trend.vortex_indicator_pos(
            high=H, low=L, close=C, fillna=True)
        df['TRIX'] = ta.trend.trix(close=C, fillna=True)
        df['MI'] = ta.trend.mass_index(high=H, low=L, fillna=True)
        df['CCI'] = ta.trend.cci(high=H, low=L, close=C, fillna=True)
        df['DPO'] = ta.trend.dpo(close=C, fillna=True)
        df['KST'] = ta.trend.kst(close=C, fillna=True)
        df['Ichimoku'] = ta.trend.ichimoku_a(high=H, low=L, fillna=True)
        df['Parabolic SAR'] = ta.trend.psar_down(
            high=H, low=L, close=C, fillna=True)
        df['STC'] = ta.trend.stc(close=C, fillna=True)
        # Momentum
        df['RSI'] = ta.momentum.rsi(close=C, fillna=True)
        df['SRSI'] = ta.momentum.stochrsi(close=C, fillna=True)
        df['TSI'] = ta.momentum.tsi(close=C, fillna=True)
        df['UO'] = ta.momentum.ultimate_oscillator(
            high=H, low=L, close=C, fillna=True)
        df['SR'] = ta.momentum.stoch(close=C, high=H, low=L, fillna=True)
        df['WR'] = ta.momentum.williams_r(high=H, low=L, close=C, fillna=True)
        df['AO'] = ta.momentum.awesome_oscillator(high=H, low=L, fillna=True)
        df['KAMA'] = ta.momentum.kama(close=C, fillna=True)
        df['ROC'] = ta.momentum.roc(close=C, fillna=True)
        df['PPO'] = ta.momentum.ppo(close=C, fillna=True)
        df['PVO'] = ta.momentum.pvo(volume=V, fillna=True)
        dfx=df.drop(['Date'], axis = 1 )
        last_row = dfx.iloc[-1]
        result=model.predict([last_row])
        if result == [1]:
          buy_signal += 1
          buy_list.append(i)
        print(num)
        print(i)
        num+=1


        print('추천 종목 객수: ',buy_signal)


    except IndexError as e:
        print(e)
        err_list.append(e)
        pass



end_time = time.time()

execution_time = end_time - start_time


print(buy_list)
buy_list2 = []
for i in buy_list:
    try:
        # 주식 데이터 가져오기
        ticker = i  # 예시로 'AAPL' (애플 주식) 사용
        start_date = '2022-08-01'
        end_date = '2023-08-01'

        stock_data = yf.download(ticker, start=start_date, end=end_date)

        # 1년 전 날짜 계산
        one_year_ago = pd.to_datetime(end_date) - pd.DateOffset(years=1)

        # 주식 데이터 필터링
        filtered_data = stock_data.loc[one_year_ago:].copy()  # 복사본 생성

        # 이동평균 계산
        window = 20  # 20일 이동평균선
        filtered_data['20MA'] = filtered_data['Close'].rolling(window=window).mean()

        # 엔벨롭 계산
        envelope_percentage = 0.20  # 20% 하단 엔벨롭
        filtered_data['LowerEnvelope'] = filtered_data['20MA'] * (1 - envelope_percentage)



        # 주식 시가와 하단 엔벨롭 비교하여 매수 신호 확인 및 조건식 적용
        filtered_data['BuySignal'] = np.where(
            (filtered_data['Open'] >= filtered_data['LowerEnvelope']) &
            (filtered_data['Open'] <= 0.5 * filtered_data['High'].max()),
            1, 0
        )

        df = filtered_data.reset_index()

        last_row = df.tail(1)
        column_value = last_row['BuySignal'].iloc[0]
        count_ones = (df['BuySignal'] == 1).sum()

        print(column_value)
        print(count_ones)
        print(i)
        if column_value == 1:
            buy_list2.append(i)
    except:
        print(i)


print(buy_list2)

with open('test_추천_list.txt', 'w+') as file:
    file.write('\n'.join(buy_list2))  # '\n' 대신 ', '를 사용하면 줄바꿈이 아닌 ', '를 기준으로



print(f"실행 시간: {execution_time:.5f} 초")