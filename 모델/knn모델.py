import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# 데이터 준비
print('# 데이터 불러오기')
stock = pd.read_csv('rawdata7_1.csv')
del stock['Unnamed: 0']
del stock['symbol']
encoder = LabelEncoder()
# stock['symbol_encoded'] = encoder.fit_transform(stock['symbol'])

# column_to_move = 'label'
# new_order = [col for col in stock.columns if col != column_to_move] + [column_to_move]
# stock = stock[new_order]
column = stock.columns
print(column[1:52])
# stock.to_csv('rawdata7_1.csv')



X = stock[column[1:51]].values
y = stock['label']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN 모델 생성
k = 10  # 이웃의 개수
weights = 'uniform'  # 가중치 설정
algorithm = 'ball_tree'  # 알고리즘 선택
metric = 'euclidean'  # 거리 메트릭 선택
knn_model = KNeighborsClassifier(n_neighbors=k, weights=weights, algorithm=algorithm, metric=metric)
# 모델 학습
print('모델학습')
knn_model.fit(X_train, y_train)

# 예측
y_pred = knn_model.predict(X_test)



# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 혼동 행렬
confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion_mat)

# 분류 보고서
classification_rep = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_rep)


# 모델 저장
model_filename = "knn_model.joblib"
joblib.dump(knn_model, model_filename)
print("Model saved as", model_filename)