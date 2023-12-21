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

#미채결 내용

PATH = "uapi/overseas-stock/v1/trading/inquire-nccs"
URL = f"{URL_BASE}/{PATH}"
symbol = 'TQQQ'
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
print(res.json())


# 주문체결내역

PATH = "uapi/overseas-stock/v1/trading/inquire-ccnl"
URL = f"{URL_BASE}/{PATH}"
symbol = 'TQQQ'
headers = {
           "content-type": "application/json",
           "authorization": f"Bearer {ACCESS_TOKEN}",
           "appKey": APP_KEY,
           "appSecret": APP_SECRET,
           "tr_id": 'VTTS3035R'
        }
params = {
    "CANO": Account,
    "ACNT_PRDT_CD":"01",
    "PDNO": "",
    "ORD_STRT_DT": "20211027",
    "ORD_END_DT": "20230727",
    "SLL_BUY_DVSN": "00",
    "CCLD_NCCS_DVSN": "00",
    "OVRS_EXCG_CD": "%",
    "SORT_SQN": "DS",
    "ORD_DT": "",
    "ORD_GNO_BRNO":"02111",
    "ODNO": "",
    "CTX_AREA_NK200": "",
    "CTX_AREA_FK200": ""
}



res = requests.get(URL, headers=headers, params=params)
print(res.json())

