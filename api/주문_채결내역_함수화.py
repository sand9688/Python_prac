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
    price = price = str(Now_Price(file_path,symbol))
    order_number = str(order_number)

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






file_path = 'C:/Users/sand9/OneDrive/바탕 화면/project/시스템/api/모의투자.txt'
symbol = 'AAPL'


# print(Order(file_path,symbol,1))
print(Not_signed(file_path))
