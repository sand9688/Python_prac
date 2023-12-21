# selenium 4
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time

driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

driver.get('https://finance.yahoo.com/quote/AAPL/financials?p=AAPL')




element = driver.find_element(By.XPATH,'//*[@id="Col1-1-Financials-Proxy"]/section/div[3]/div[1]/div')      #('tableRanking')
print(element.text)
data1 = element.text

sentence_list = data1.split("\n")

print(sentence_list)