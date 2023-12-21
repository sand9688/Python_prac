import pandas as pd

print('# 데이터 불러오기')
stock = pd.read_csv('모델/rawdata4.csv')
del stock['Unnamed: 0']
column = stock.columns


column = stock.columns
print(column)

X_num = stock[column[0:55]].values
Y_num = stock['Per'].replace({0: 1, 1: 0})


# 상관계수 계산
correlation_matrix = stock[column[0:56]].corr()

# 'Per' 변수와의 상관계수
correlation_with_per = correlation_matrix['Per']

print("Per 변수와의 상관관계:")
print(correlation_with_per)