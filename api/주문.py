import requests
import json


file_path = 'C:/Users/sand9/OneDrive/바탕 화면/project/시스템/api/모의투자.txt'


with open(file_path, 'r') as file :
    content = file.read().splitlines()

Account = content[0].split('-')[0]
Account_Back = content[0].split('-')[1]
APP_KEY = content[1]
APP_SECRET = content[2]
URL_BASE = "https://openapivts.koreainvestment.com:29443"

headers = {"content-type":"application/json"}
body = {"grant_type":"client_credentials",
        "appkey":APP_KEY,
        "appsecret":APP_SECRET}

PATH = "oauth2/tokenP"

URL = f"{URL_BASE}/{PATH}"

res = requests.post(URL, headers=headers, data=json.dumps(body))
res_data = res.json()
ACCESS_TOKEN = res_data['access_token']


def hashkey(datas):
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



PATH = "uapi/overseas-stock/v1/trading/order"
URL = f"{URL_BASE}/{PATH}"
symbol = 'TQQQ'
#매수
data = {
            "CANO": Account,
            "ACNT_PRDT_CD": Account_Back,
            "OVRS_EXCG_CD": 'NAS',
            "PDNO": symbol,
            "ORD_QTY": str(10),# 갯수
            "OVRS_ORD_UNPR": str(43.4800),# 금액
            "ORD_SVR_DVSN_CD": "0",
            "ORD_DVSN": '00'
        }
headers = {
           "content-type": "application/json",
           "authorization": f"Bearer {ACCESS_TOKEN}",
           "appKey": APP_KEY,
           "appSecret": APP_SECRET,
           "tr_id": 'VTTT1002U',
           "hashkey": hashkey(data)
        }


# res = requests.post(URL, headers=headers, data=json.dumps(data))
# print(res.json())

#매도

data = {
            "CANO": Account,
            "ACNT_PRDT_CD": Account_Back,
            "OVRS_EXCG_CD": 'NAS',
            "PDNO": symbol,
            "ORD_QTY": str(10),# 갯수
            "OVRS_ORD_UNPR": str(45.2200),# 금액
            "ORD_SVR_DVSN_CD": "0",
            "ORD_DVSN": '00'
        }
headers = {
           "content-type": "application/json",
           "authorization": f"Bearer {ACCESS_TOKEN}",
           "appKey": APP_KEY,
           "appSecret": APP_SECRET,
           "tr_id": 'VTTT1001U',
           "hashkey": hashkey(data)
        }


res = requests.post(URL, headers=headers, data=json.dumps(data))
print(res.json())


#취소 주문
PATH = "uapi/overseas-stock/v1/trading/order-rvsecncl"
URL = f"{URL_BASE}/{PATH}"
data = {
            "CANO": Account,
            "ACNT_PRDT_CD": Account_Back,
            "OVRS_EXCG_CD": 'NAS',
            "PDNO": symbol,
            "ORGN_ODNO": "30135009", #오더 번호
            "RVSE_CNCL_DVSN_CD": "02",# 01은 정정
            "ORD_QTY": str(10),# 갯수
            "OVRS_ORD_UNPR": str(45.2200),# 금액
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
print(res.json())



