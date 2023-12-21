import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd


# 데이터 준비
print('# 데이터 불러오기')
stock = pd.read_csv('rawdata7.csv')
del stock['symbol']

column = stock.columns
print(column[1:52])




X = stock[column[1:51]].values
y = stock['label']

# 데이터를 학습용과 테스트용으로 분할
X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(X_train,Y_train, test_size=0.2, random_state=42)


# 주식 데이터의 특성 수
num_features = 51

# 클래스 수 (예: 주가 상승, 하락, 변동 없음)
num_classes = 4



# 모델 생성
model = keras.Sequential([
    layers.Input(shape=(num_features,)),  # 입력 데이터의 형태를 지정
    layers.Reshape((num_features, 1)),  # 1D CNN을 위해 차원 변경
    layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# EarlyStopping 콜백 설정
early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=5, mode='auto', verbose=1)

# 모델 훈련
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_val, y_val), callbacks=[early_stopping_callback])

# 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

# 모델 예측
y_pred = model.predict(X_test)


# 모델을 저장할 경로 및 파일 이름 지정
model.save("10_08_CNN_Model.h5")

# 저장된 모델을 불러오기 위한 방법
loaded_model = tf.keras.models.load_model("10_08_CNN_Model.h5")