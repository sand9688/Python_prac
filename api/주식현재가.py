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

PATH = "uapi/overseas-price/v1/quotations/price"
URL = f"{URL_BASE}/{PATH}"
symbol = 'AAPL'
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
print('전일종가 :',res_data['base'])
print('현재가 :',res_data['last'])
print('등락율 :',res_data['rate'],"%")
print('거래대금 :',res_data['tamt'])
print('거래량 :',res_data['tvol'])
print('매수가능여부 :',res_data['ordy'])
