import threading
import time
import datetime
import requests
import json
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import FinanceDataReader as fdr
import ta
import schedule
import csv



# 주어진 모델 함수들

def Model_1 ():
    num=1
    buy_list = []
    buy_signal = 0
    #입력데이터 준비
    df_nasdaq = fdr.StockListing('NASDAQ')
    nasdaq_list= df_nasdaq['Symbol'].tolist()
    # 현재 날짜 가져오기
    current_date = datetime.now()
    # 2년 전 날짜 계산
    two_year_ago = current_date - timedelta(days=730)
    #모델설정
    model = joblib.load('best_decision_tree_model.joblib')
    #input_df만들기
    input_df=pd.DataFrame()

    for i in nasdaq_list:
        try:
            df=yf.download(i,start = two_year_ago)

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
            # last_row=last_row.reshape(-1, 1)
            result=model.predict([last_row])
            if result == [1]:
                buy_signal += 1
                buy_list.append(i)
            num+=1


            print('추천 종목 객수: ',buy_signal)
        except IndexError as e:
                print(e)
                pass
    model1_list=buy_list

    return model1_list

def Model_2():
    num=1
    buy_list = []
    buy_signal = 0

    #입력데이터 준비
    df_nasdaq = fdr.StockListing('NASDAQ')
    nasdaq_list= df_nasdaq['Symbol'].tolist()
    # 현재 날짜 가져오기
    current_date = datetime.now()
    # 2년 전 날짜 계산
    two_year_ago = current_date - timedelta(days=730)
    # 종목별 데이터 수집 및 데이터프레임 만들기
    dfs = []
    # 모델 설정
    model = joblib.load('new_decision_tree_model.pkl')
    for i in nasdaq_list:
        try :
            data = yf.download(i, start=two_year_ago, progress=False)


            # 1번: 주식 가격 데이터
            data['Stock Price'] = data['Open']  # Open 기준으로 주식 가격 데이터 설정

            # 2번: 거래량 데이터
            data['Volume'] = data['Volume']

            # 3번: 이동평균 데이터
            data['Moving Average'] = data['Close'].rolling(window=30).mean()

            # 4번: RSI 데이터
            def compute_rsi(data, window=14):
                diff = data.diff(1)
                gain = diff.where(diff > 0, 0)
                loss = -diff.where(diff < 0, 0)

                avg_gain = gain.rolling(window=window).mean()
                avg_loss = loss.rolling(window=window).mean()

                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                return rsi

            data['RSI'] = compute_rsi(data['Close'], window=14)

            # 5번: MACD 데이터
            exp12 = data['Close'].ewm(span=12, adjust=False).mean()
            exp26 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = exp12 - exp26
            data['Signal Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

            # 6번: 볼린저 밴드 데이터
            rolling_mean = data['Close'].rolling(window=20).mean()
            rolling_std = data['Close'].rolling(window=20).std()
            data['Upper Bollinger Band'] = rolling_mean + 2 * rolling_std
            data['Lower Bollinger Band'] = rolling_mean - 2 * rolling_std

            # 7번: 가격 변화율(이전 기준)
            data['Price Change (%)'] = (data['Close'] - data['Open']) / data['Open'] * 100

            # 8번: ATR (Average True Range) 데이터
            data['High-Low'] = data['High'] - data['Low']
            data['High-Close'] = abs(data['High'] - data['Close'].shift())
            data['Low-Close'] = abs(data['Low'] - data['Close'].shift())
            data['True Range'] = data[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
            data['ATR'] = data['True Range'].rolling(window=14).mean()
            data.drop(['High-Low', 'High-Close', 'Low-Close', 'True Range'], axis=1, inplace=True)

            # 9번: Bollinger Bands 데이터
            data['Middle Bollinger Band'] = rolling_mean
            data['Bollinger Band Width'] = (data['Upper Bollinger Band'] - data['Lower Bollinger Band']) / data['Middle Bollinger Band']

            # 10번: Historical Volatility 데이터
            data['Historical Volatility'] = data['Close'].pct_change().rolling(window=21).std() * 100

            nasdaq_data = data.fillna(0)



            nasdaq_data=nasdaq_data.reset_index()

            nasdaq_data.drop(['Date', 'Price Change (%)'], axis=1, inplace=True)


            last_row = [nasdaq_data.iloc[-1]]
            result=model.predict(last_row)

            if result == [1]:
                buy_list.append(i)
                buy_signal += 1

            num +=1

        except:
            print('pass')
    model2_list=buy_list
    return model2_list

## 토큰 만들기
def Token(APP_KEY, APP_SECRET):
    URL_BASE = "https://openapivts.koreainvestment.com:29443"
    headers = {"content-type": "application/json"}
    body = {
        "grant_type": "client_credentials",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET
    }
    PATH = "oauth2/tokenP"
    URL = f"{URL_BASE}/{PATH}"
    res = requests.post(URL, headers=headers, data=json.dumps(body))
    res_data = res.json()
    ACCESS_TOKEN = res_data['access_token']
    return ACCESS_TOKEN


## 헤쉬키
def hashkey(datas,APP_KEY,APP_SECRET):

    URL_BASE = "https://openapivts.koreainvestment.com:29443"
    PATH = "uapi/hashkey"
    URL = f"{URL_BASE}/{PATH}"
    headers = {
        'content-Type' : 'application/json',
        'appKey' : APP_KEY,
        'appSecret' : APP_SECRET,
        }
    res = requests.post(URL, headers=headers, data=json.dumps(datas))
    hashkey = res.json()["HASH"]

    return hashkey




## 잔고
def Account_Check(file_path):
    URL_BASE = "https://openapivts.koreainvestment.com:29443"
    PATH = 'uapi/overseas-stock/v1/trading/inquire-balance'
    URL = f"{URL_BASE}/{PATH}"
    with open(file_path, 'r') as file :
        content = file.read().splitlines()
    Account = content[0].split('-')[0]
    Account_Back = content[0].split('-')[1]
    APP_KEY = content[1]
    APP_SECRET = content[2]



    ACCESS_TOKEN = Token(APP_KEY,APP_SECRET)


    headers = {
        "content-type": "application/json",
        "authorization":f"Bearer {ACCESS_TOKEN}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "VTTS3012R"
    }

    params = {
        "CANO": Account,
        "ACNT_PRDT_CD":Account_Back,
        "OVRS_EXCG_CD": "NASD",
        "TR_CRCY_CD": "USD",
        "CTX_AREA_FK200": "",
        "CTX_AREA_NK200": ""
    }

    res = requests.get(URL, headers=headers, params=params)

    stock_data = res.json()



    return stock_data

#현재 가격
def Now_Price(file_path,Symbol):
    URL_BASE = "https://openapivts.koreainvestment.com:29443"
    PATH = "uapi/overseas-price/v1/quotations/price"
    URL = f"{URL_BASE}/{PATH}"
    with open(file_path, 'r') as file :
        content = file.read().splitlines()
    Account = content[0].split('-')[0]
    Account_Back = content[0].split('-')[1]
    APP_KEY = content[1]
    APP_SECRET = content[2]
    ACCESS_TOKEN = Token(APP_KEY,APP_SECRET)

    symbol = Symbol
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "HHDFS00000300"
    }

    params = {
        "AUTH": "",
        "EXCD": 'NAS',
        "SYMB": symbol
    }

    res = requests.get(URL, headers=headers, params=params)
    res_data = res.json()['output']

    return res_data['last']

#주문
def Order(file_path,Symbol,Order_num):
    #Order_num = 0: 매수 / 1: 매도
    with open(file_path, 'r') as file :
        content = file.read().splitlines()

    Account = content[0].split('-')[0]
    Account_Back = content[0].split('-')[1]
    APP_KEY = content[1]
    APP_SECRET = content[2]
    URL_BASE = "https://openapivts.koreainvestment.com:29443"
    PATH = "uapi/overseas-stock/v1/trading/order"
    URL = f"{URL_BASE}/{PATH}"

    symbol = Symbol
    ACCESS_TOKEN = Token(APP_KEY,APP_SECRET)
    price_data = Now_Price(file_path,symbol)
    price = str(price_data['last'])

    if Order_num == 0:
        #매수
        data = {
                    "CANO": Account,
                    "ACNT_PRDT_CD": Account_Back,
                    "OVRS_EXCG_CD": 'NAS',
                    "PDNO": symbol,
                    "ORD_QTY": str(10),# 갯수
                    "OVRS_ORD_UNPR": price,# 금액
                    "ORD_SVR_DVSN_CD": "0",
                    "ORD_DVSN": '00'
                }
        headers = {
                "content-type": "application/json",
                "authorization": f"Bearer {ACCESS_TOKEN}",
                "appKey": APP_KEY,
                "appSecret": APP_SECRET,
                "tr_id": 'VTTT1002U',
                "hashkey": hashkey(data,APP_KEY,APP_SECRET)
                }
        res = requests.post(URL, headers=headers, data=json.dumps(data))
        return res.json()


    elif Order_num == 1 :
        #매도
        data = {
                    "CANO": Account,
                    "ACNT_PRDT_CD": Account_Back,
                    "OVRS_EXCG_CD": 'NAS',
                    "PDNO": symbol,
                    "ORD_QTY": str(10),# 갯수
                    "OVRS_ORD_UNPR": price,# 금액
                    "ORD_SVR_DVSN_CD": "0",
                    'SLL_TYPE' : '00',
                    "ORD_DVSN": '00'
                }
        headers = {
                "content-type": "application/json",
                "authorization": f"Bearer {ACCESS_TOKEN}",
                "appKey": APP_KEY,
                "appSecret": APP_SECRET,
                "tr_id": 'VTTT1001U',
                "hashkey": hashkey(data,APP_KEY,APP_SECRET)
                }


        res = requests.post(URL, headers=headers, data=json.dumps(data))
        return res.json()

#미채결 확인
def Not_signed(file_path):
    with open(file_path, 'r') as file :
        content = file.read().splitlines()

    Account = content[0].split('-')[0]
    Account_Back = content[0].split('-')[1]
    APP_KEY = content[1]
    APP_SECRET = content[2]
    URL_BASE = "https://openapivts.koreainvestment.com:29443"
    ACCESS_TOKEN = Token(APP_KEY,APP_SECRET)

    PATH = "uapi/overseas-stock/v1/trading/inquire-nccs"
    URL = f"{URL_BASE}/{PATH}"

    headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {ACCESS_TOKEN}",
            "appKey": APP_KEY,
            "appSecret": APP_SECRET,
            "tr_id": 'VTTS3018R'
            }
    params = {
        "CANO": Account,
        "ACNT_PRDT_CD":Account_Back,
        "OVRS_EXCG_CD": "NASD",
        "SORT_SQN": "DS",
        "CTX_AREA_FK200": "",
        "CTX_AREA_NK200": ""
    }
    res = requests.get(URL, headers=headers, params=params)
    return res.json()

#정정(취소) 주문
def Cancel_Order(file_path,Not_signed_data,order_number):
    #order_number :01은 정정 02는 취소
    with open(file_path, 'r') as file :
        content = file.read().splitlines()

    Account = content[0].split('-')[0]
    Account_Back = content[0].split('-')[1]
    APP_KEY = content[1]
    APP_SECRET = content[2]
    URL_BASE = "https://openapivts.koreainvestment.com:29443"
    ACCESS_TOKEN = Token(APP_KEY,APP_SECRET)

    order_data = Not_signed_data
    price = price = str(Now_Price(file_path,order_data['pdno']))
    order_number = order_number

    PATH = "uapi/overseas-stock/v1/trading/order-rvsecncl"
    URL = f"{URL_BASE}/{PATH}"
    data = {
                "CANO": Account,
                "ACNT_PRDT_CD": Account_Back,
                "OVRS_EXCG_CD": 'NAS',
                "PDNO": order_data['pdno'],
                "ORGN_ODNO": order_data['odno'], #오더 번호
                "RVSE_CNCL_DVSN_CD": order_number,# 01은 정정
                "ORD_QTY": order_data['nccs_qty'],# 갯수
                "OVRS_ORD_UNPR": price,# 금액
                "ORD_SVR_DVSN_CD": "0",
            }

    headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {ACCESS_TOKEN}",
            "appKey": APP_KEY,
            "appSecret": APP_SECRET,
            "tr_id": 'VTTT1004U',
            "hashkey": hashkey(data)
            }


    res = requests.post(URL, headers=headers, data=json.dumps(data))
    return res.json()

# 계좌 체크 함수 (매도)
def Check_PER(file_path):
    file_path=file_path
    while True:
        #미체결 확인
        not_signed_list = []
        now = datetime.now()
        now =int(now.strftime('%H%M%S'))
        not_signed_data=Not_signed(file_path)
        for k in not_signed_data['output']:
            not_signed_list.append(k['pdno'])
        # 평가 손익율 확인
        stock_data = Account_Check(file_path)
        for i in stock_data['output1']:
            if i['ovrs_pdno'] not in not_signed_list:
                if i['evlu_pfls_rt'] > 7.5 or i['evlu_pfls_rt'] < -5:
                    Order(file_path,i['ovrs_pdno'],1)


            time.sleep(10)
        time.sleep(500)
        for k in not_signed_data['output']:
            # 현재 시간 가져오기
            now = datetime.now()
            # 시:분:초 형식으로 출력
            formatted_time = int(now.strftime('%H%M%S'))
            if formatted_time - int(k['ord_tmd']) >= 500:
                Cancel_Order(file_path,k,'01')
            time.sleep(10)
        if now  == 60000:
            break



        time.sleep(10)  # 10초에 한 번씩 계좌를 확인


# 추천 종목 함수
def Recommend(file_path):
    # model_1과 model_2 함수 실행
    result_model1 = Model_1()
    result_model2 = Model_2()
    common_elements = [x for x in result_model1 if x in result_model2]

    return common_elements

# 구매 함수
def Recommend_Buy(file_path,common_elements):
    not_signed_data=Not_signed(file_path)
    price_data=Now_Price(file_path,i)
    for i in common_elements:
        if float(price_data['last']) > 10 :
            Order(file_path,i,0)
            time.sleep(2)
    time.sleep(300)
    if len(not_signed_data['output']) > 0:
        for k in  not_signed_data['output']:
            # 현재 시간 가져오기
            now = datetime.now()
            # 시:분:초 형식으로 출력
            formatted_time = int(now.strftime('%H%M%S'))
            if int(formatted_time) - int(k['ord_tmd']) >= int('500'):
                Cancel_Order(file_path,k,'01')
                time.sleep(2)




# 메인 함수

def main():
    file_path = 'C:/Users/sand9/OneDrive/바탕 화면/project/시스템/api/모의투자.txt'
    buy_list=Recommend(file_path)
    Account_data=Account_Check(file_path)
    already_list = []
    for j in Account_data['output1']:
        already_list.append(j['ovrs_pdno'])
    for i in buy_list:
        if i in already_list:
            buy_list.remove(i)


    print(buy_list)
    print(len(buy_list))
    with open ('test.csv','w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(buy_list)






if __name__ == "__main__":
    main()