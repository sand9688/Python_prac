import lightgbm as lgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# 데이터 준비
print('# 데이터 불러오기')
stock = pd.read_csv('rawdata7_1.csv')
del stock['Unnamed: 0']
del stock['symbol']
# stock['symbol_encoded'] = encoder.fit_transform(stock['symbol'])

# column_to_move = 'label'
# new_order = [col for col in stock.columns if col != column_to_move] + [column_to_move]
# stock = stock[new_order]
column = stock.columns
print(column[1:52])
# stock.to_csv('rawdata7_1.csv')


X = stock[column[1:51]].values
y = stock['label']



# 데이터를 훈련 세트와 테스트 세트로 나눕니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM 데이터셋 생성
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# LightGBM 모델 초기화
params = {
    'objective': 'multiclass',
    'num_class': num_classes,
    'boosting_type': 'gbdt',
    'metric': 'multi_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'early_stopping_rounds': 10,  # 조기 중단 반복 횟수 설정
    'eval_metric': 'multi_logloss',
    'verbose': 0
}

num_round = 20000  # 반복 횟수 설정

# LightGBM 모델 훈련
bst = lgb.train(params, train_data, num_round, valid_sets=[valid_data])

# 모델 평가
y_pred = bst.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)  # 확률 예측에서 클래스로 변환

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred_class)
print(f'Accuracy: {accuracy:.2f}')

# 다중 분류에 대한 세부 보고서 출력
print(classification_report(y_test, y_pred_class))


# 모델 저장
model_filename = "2man_0.05_lightboost_multiclass_model.pkl"
with open(model_filename, 'wb') as model_file:
    pickle.dump(bst, model_file)
print(f"Model saved as {model_filename}")

# 저장된 모델 로드
with open(model_filename, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# 로드한 모델을 사용하여 예측 수행
loaded_y_pred = loaded_model.predict(dtest)
loaded_y_pred_class = [int(x) for x in loaded_y_pred]

# 로드한 모델의 정확도 확인
loaded_accuracy = accuracy_score(y_test, loaded_y_pred_class)
print(f"Loaded Model Accuracy: {loaded_accuracy:.2f}")