import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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

# 모델 작성
model = Sequential()

model.add(Dense(units=50, activation='relu', input=50))
model.add(Dense(units=100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=24, activation='relu'))
model.add(Dense(units=12, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=4, activation='softmax'))

model.compile(optimizer='adam',loss='spare_categorical_crossentropy',metrics=['accuercy'])

#학습률 전략
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)


#모델 학습
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_val, y_val), callbacks=[reduce_lr])


# 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

# 모델 예측
y_pred = model.predict(X_test)


# 모델을 저장할 경로 및 파일 이름 지정
model.save("10_11_deepModel.h5")

# 저장된 모델을 불러오기 위한 방법
loaded_model = tf.keras.models.load_model("10_11_deepModel.h5")