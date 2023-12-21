import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import random


# 데이터 불러오기
print('# 데이터 불러오기')
stock = pd.read_csv('rawdata4.csv')
del stock['Unnamed: 0']
column = stock.columns

# Feature와 Target 분리
X_num = stock[column[0:55]].values
Y_num = stock['Per'].replace({0: 1, 1: 0})

# 분류 데이터 생성
X_train, X_test, y_train, y_test = train_test_split(X_num, Y_num, test_size=0.2, random_state=42)

# 의사결정나무 분류 모델 생성
print('의사결정나무')
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)


# 모델 저장 함수
def save_model(model, filename):
    joblib.dump(model, filename)
    print("모델이 저장되었습니다.")

# 모델 저장
print('모델저장')
model_filename = "decision_tree_model.joblib"
save_model(classifier, model_filename)

# 테스트 데이터로 성능 평가
accuracy = classifier.score(X_test, y_test)
print("Test Accuracy:", accuracy)
