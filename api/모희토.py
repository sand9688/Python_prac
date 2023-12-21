import requests
import json
import mojito

file_path = 'C:/Users/sand9/OneDrive/바탕 화면/project/시스템/api/모의투자.txt'


with open(file_path, 'r') as file :
    content = file.read().splitlines()
acc_no = content[0]
key = content[1]
secret = content[2]
broker = mojito.KoreaInvestment(
    api_key=key,
    api_secret=secret,
    acc_no= acc_no,
    exchange='나스닥',
    mock=True)
print(broker)
#현재가 조회

balance = broker.fetch_present_balance()
print('외화예수금 : ',balance['output2'][3]['frcr_dncl_amt_2'])
print('출금가능 외화금액 : ',balance['output2'][3]['frcr_drwg_psbl_amt_1'])
print('출금가능 원화금액 : ',balance['output2'][3]['frcr_evlu_amt2'])



price = broker.fetch_price("TQQQ")
buy_price=float(price['output']['last'])

# resp = broker.create_limit_buy_order(
#     symbol="TQQQ",
#     price=buy_price,
#     quantity=10
# )

sell_price=float(price['output']['last'])


resp = broker.create_limit_sell_order(
    symbol="TQQQ",
    price=sell_price,
    quantity=10
)



resp = broker.fetch_present_balance()
print(resp)
