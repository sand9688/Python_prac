import requests
import json

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
def hashkey(datas):

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
def Balance(APP_KEY, APP_SECRET, Account, Account_Back):
    # Token 함수를 정의한 것으로 가정하고, 액세스 토큰을 얻습니다.
    ACCESS_TOKEN = Token(APP_KEY, APP_SECRET)
    URL_BASE = "https://openapivts.koreainvestment.com:29443"
    PATH = 'uapi/overseas-stock/v1/trading/inquire-balance'
    URL = f"{URL_BASE}/{PATH}"

    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "VTTS3012R"
    }

    params = {
        "CANO": Account,
        "ACNT_PRDT_CD": Account_Back,
        "OVRS_EXCG_CD": "NASD",
        "TR_CRCY_CD": "USD",
        "CTX_AREA_FK200": "",
        "CTX_AREA_NK200": ""
    }

    res = requests.get(URL, headers=headers, params=params)
    stock_data = res.json()
    for j in stock_data['output1']:
        print('해외 상품 코드 :',j['ovrs_pdno'])
        print('해외 종목 명 :',j['ovrs_item_name'])
        print('평가손익율 :',j['evlu_pfls_rt'], '%')
        print('수량 :',j['ord_psbl_qty'])
        print('외화매입금액 :',j['frcr_pchs_amt1'])
        print('해외주식 평가 금액:',j['ovrs_stck_evlu_amt'])


    return stock_data
## 현재가

def Price_last(symbol, ACCESS_TOKEN,APP_KEY,APP_SECRET):
    URL_BASE = "https://openapivts.koreainvestment.com:29443"
    PATH = "uapi/overseas-price/v1/quotations/price"
    URL = f"{URL_BASE}/{PATH}"
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
    price = str(res_data['last'])

    return price



## 매수
def Transaction_Buy(symbol, ACCESS_TOKEN, APP_KEY, APP_SECRET, Account, Account_Back):
    URL_BASE = "https://openapivts.koreainvestment.com:29443"
    PATH = "uapi/overseas-stock/v1/trading/order"
    URL = f"{URL_BASE}/{PATH}"
    price = Price_last(symbol, ACCESS_TOKEN,APP_KEY,APP_SECRET)  # Price_last() 함수로 가격을 가져오는 로직이 구현되어 있다고 가정합니다.

    # 매수 주문 데이터
    data = {
        "CANO": Account,
        "ACNT_PRDT_CD": Account_Back,
        "OVRS_EXCG_CD": 'NAS',
        "PDNO": symbol,
        "ORD_QTY": str(10),  # 갯수
        "OVRS_ORD_UNPR": price,  # 금액
        "ORD_SVR_DVSN_CD": "0",
        "ORD_DVSN": '00'
    }

    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": 'VTTT1002U',
        "hashkey": hashkey(data)  # hashkey() 함수가 정의되어 있다고 가정합니다.
    }

    res = requests.post(URL, headers=headers, data=json.dumps(data))
    print(res.json())
## 매도
def Transaction_Sell(symbol, ACCESS_TOKEN, APP_KEY, APP_SECRET, Account, Account_Back):
    URL_BASE = "https://openapivts.koreainvestment.com:29443"
    PATH = "uapi/overseas-stock/v1/trading/order"
    URL = f"{URL_BASE}/{PATH}"
    price = Price_last(symbol, ACCESS_TOKEN, APP_KEY, APP_SECRET)  # Price_last() 함수로 가격을 가져오는 로직이 구현되어 있다고 가정합니다.

    # 매도 주문 데이터
    data = {
        "CANO": Account,
        "ACNT_PRDT_CD": Account_Back,
        "OVRS_EXCG_CD": 'NAS',
        "PDNO": symbol,
        "ORD_QTY": str(10),  # 갯수
        "OVRS_ORD_UNPR": price,  # 금액
        "SLL_TYPE": '00',  # 매도 타입, '00'은 지정가 매도를 의미합니다.
        "ORD_SVR_DVSN_CD": "0",
        "ORD_DVSN": '00'
    }

    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": 'VTTT1002U',
        "hashkey": hashkey(data)  # hashkey() 함수가 정의되어 있다고 가정합니다.
    }

    res = requests.post(URL, headers=headers, data=json.dumps(data))
    print(res.json())

##취소
def Transaction_Cancle(symbol, ACCESS_TOKEN, APP_KEY, APP_SECRET, Account, Account_Back, ORDER_Number):
    URL_BASE = "https://openapivts.koreainvestment.com:29443"
    PATH = "uapi/overseas-stock/v1/trading/order-rvsecncl"
    URL = f"{URL_BASE}/{PATH}"
    price = Price_last(symbol, ACCESS_TOKEN, APP_KEY, APP_SECRET)  # Price_last() 함수로 가격을 가져오는 로직이 구현되어 있다고 가정합니다.

    # 주문 취소 데이터
    data = {
        "CANO": Account,
        "ACNT_PRDT_CD": Account_Back,
        "OVRS_EXCG_CD": 'NAS',
        "PDNO": symbol,
        "ORGN_ODNO": ORDER_Number,  # 오더 번호, 취소할 주문의 오더 번호를 지정해야 합니다.
        "RVSE_CNCL_DVSN_CD": "02",  # 01은 정정, 02는 취소를 의미합니다.
        "ORD_QTY": str(10),  # 갯수, 취소할 주문의 수량을 지정합니다.
        "OVRS_ORD_UNPR": price,  # 금액, 취소할 주문의 가격을 지정합니다.
        "ORD_SVR_DVSN_CD": "0",
    }

    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": 'VTTT1004U',
        "hashkey": hashkey(data)  # hashkey() 함수가 정의되어 있다고 가정합니다.
    }

    res = requests.post(URL, headers=headers, data=json.dumps(data))
    print(res.json())

## 정정
def Transaction_Correction(symbol, ACCESS_TOKEN, APP_KEY, APP_SECRET, Account, Account_Back, ORDER_Number):
    URL_BASE = "https://openapivts.koreainvestment.com:29443"
    PATH = "uapi/overseas-stock/v1/trading/order-rvsecncl"
    URL = f"{URL_BASE}/{PATH}"
    price = Price_last(symbol, ACCESS_TOKEN, APP_KEY, APP_SECRET)  # Price_last() 함수로 가격을 가져오는 로직이 구현되어 있다고 가정합니다.

    # 주문 정정 데이터
    data = {
        "CANO": Account,
        "ACNT_PRDT_CD": Account_Back,
        "OVRS_EXCG_CD": 'NAS',
        "PDNO": symbol,
        "ORGN_ODNO": ORDER_Number,  # 오더 번호, 취소할 주문의 오더 번호를 지정해야 합니다.
        "RVSE_CNCL_DVSN_CD": "01",  # 01은 정정, 02는 취소를 의미합니다.
        "ORD_QTY": str(10),  # 갯수, 취소할 주문의 수량을 지정합니다.
        "OVRS_ORD_UNPR": price,  # 금액, 취소할 주문의 가격을 지정합니다.
        "ORD_SVR_DVSN_CD": "0",
    }

    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": 'VTTT1004U',
        "hashkey": hashkey(data)  # hashkey() 함수가 정의되어 있다고 가정합니다.
    }

    res = requests.post(URL, headers=headers, data=json.dumps(data))
    print(res.json())

## 채결 내용
def Contract_Details(ACCESS_TOKEN, APP_KEY, APP_SECRET, Account, Account_Back, Start_day, Stop_day):
    URL_BASE = "https://openapivts.koreainvestment.com:29443"
    PATH = "uapi/overseas-stock/v1/trading/inquire-ccnl"
    URL = f"{URL_BASE}/{PATH}"

    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": 'VTTS3035R'
    }

    params = {
        "CANO": Account,
        "ACNT_PRDT_CD": Account_Back,
        "PDNO": "",
        "ORD_STRT_DT": Start_day,
        "ORD_END_DT": Stop_day,
        "SLL_BUY_DVSN": "00",
        "CCLD_NCCS_DVSN": "00",
        "OVRS_EXCG_CD": "%",
        "SORT_SQN": "DS",
        "ORD_DT": "",
        "ORD_GNO_BRNO": "02111",
        "ODNO": "",
        "CTX_AREA_NK200": "",
        "CTX_AREA_FK200": ""
    }

    res = requests.get(URL, headers=headers, params=params)
    contract_details_data = res.json()
    print(contract_details_data)

    return contract_details_data

## 미채결
def Non_Contract_Details(ACCESS_TOKEN, APP_KEY, APP_SECRET, Account, Account_Back):
    URL_BASE = "https://openapivts.koreainvestment.com:29443"
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
        "ACNT_PRDT_CD": Account_Back,
        "OVRS_EXCG_CD": "NASD",
        "SORT_SQN": "DS",
        "CTX_AREA_FK200": "",
        "CTX_AREA_NK200": ""
    }

    res = requests.get(URL, headers=headers, params=params)
    non_contract_data=res.json()
    print(non_contract_data)

    return non_contract_data




