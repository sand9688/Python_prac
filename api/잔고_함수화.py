import requests
import json



file_path = 'C:/Users/sand9/OneDrive/바탕 화면/project/시스템/api/모의투자.txt'


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


stock_data = Account_Check(file_path)
stock_list = []
for j in stock_data['output1']:
    print('해외 상품 코드 :',j['ovrs_pdno'])
    print('해외 종목 명 :',j['ovrs_item_name'])
    print('해외 종목 명 :',j['ovrs_item_name'])
    print('외화기준 평가손익율 :',j['frcr_evlu_pfls_amt'], '%')
    print('평가손익율 :',j['evlu_pfls_rt'], '%')
    print('수량 :',j['ord_psbl_qty'])
    print('외화매입금액 :',j['frcr_pchs_amt1'])
    print('해외주식 평가 금액:',j['ovrs_stck_evlu_amt'])
    stock_list.append(j['ovrs_pdno'])

for i in stock_list:
    print(i)


