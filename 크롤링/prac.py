import FinanceDataReader as fdr
import requests #pip install requests
from bs4 import BeautifulSoup # pip install beautlfulsoup4
import pandas as pd # pip install pandas
import time

#나스닥에 상장된 모든 기업 목록 가져오기
nasdaq_list = fdr.StockListing('NASDAQ')

corp_list = pd.DataFrame(nasdaq_list)
print(corp_list['Symbol'])

#corp_list 에 있는 Symbol 들을 나열해서 저장한뒤 가져와서 책으로 만들어야될까?
tickers = corp_list['Symbol']
headers= {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:87.0) Gecko/20100101 Firefox/87.0'}

for ticker in tickers:
    # 연간: https://stockanalysis.com/stocks/aapl/financials/quarterly/
    url = "https://stockanalysis.com/stocks/"+ticker+"/financials/"

    #분기: https://stockanalysis.com/stocks/aapl/financials/quarterly/
    url = "https://stockanalysis.com/stocks/"+ticker+"/financials/quarterly/"



    response = requests.get(url,allow_redirects=False, headers=headers)  # 웹 페이지 요청
    time.sleep(10)


    soup = BeautifulSoup(response.content, 'html.parser')  # HTML 파싱



    element_tables = soup.select("table[data-test='financials']")  # 원하는 테이블 요소 선택
    df = pd.read_html(str(element_tables))[0]  # 선택한 테이블을 데이터프레임으로 변환
    print(df)

    df = pd.read_html(str(element_tables))[0] #'0번 테이블 뽑기
    print(df)

    #df.to_csv(ticker+'.csv', index=False, encoding='euc-kr')
    # 엑셀 파일로 저장하기용
    # df.to_excel(ticker+'.xlsx', index=False, encoding='euc-kr')