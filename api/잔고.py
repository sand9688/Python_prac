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

PATH = 'uapi/overseas-stock/v1/trading/inquire-balance'
URL = f"{URL_BASE}/{PATH}"

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
items=stock_data['output2'].items()
print(stock_data['output2'])
print(len(stock_data['output2']))
print(len(stock_data['output2'].keys()))
print(stock_data['output2'].items())
for key, value in stock_data['output2'].items():
  print(value)


already_list = []
for j in stock_data['output1']:
  print('해외 상품 코드 :',j['ovrs_pdno'])
  print('해외 종목 명 :',j['ovrs_item_name'])
  print('평가손익율 :',j['evlu_pfls_rt'], '%')
  print('수량 :',j['ord_psbl_qty'])
  print('외화매입금액 :',j['frcr_pchs_amt1'])
  print('해외주식 평가 금액:',j['ovrs_stck_evlu_amt'])
  already_list.append(j['ovrs_pdno'])


print(already_list)

# PATH = 'uapi/overseas-stock/v1/trading/inquire-present-balance'
# URL = f"{URL_BASE}/{PATH}"

# headers = {
#     "content-type": "application/json",
#     "authorization":f"Bearer {ACCESS_TOKEN}",
#     "appKey": APP_KEY,
#     "appSecret": APP_SECRET,
#     "tr_id": "VTRP6504R"
# }

# params = {
#     "CANO": Account,
#     "ACNT_PRDT_CD":Account_Back,
#     "WCRC_FRCR_DVSN_CD": "01",
#     "NATN_CD": "840",
#     "TR_MKET_CD": "01",
#     "INQR_DVSN_CD": "00"
# }


# res = requests.get(URL, headers=headers, params=params)

# stock_data = res.json()

# # print(stock_data['output3']['tot_dncl_amt'])


