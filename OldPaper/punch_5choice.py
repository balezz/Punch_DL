# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 19:59:24 2019

@author: void_man
"""

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.utils import plot_model
import matplotlib.pyplot as plt
from keras import optimizers
#import pandas as pd


# загружаем данные с фичами
#dataset = np.loadtxt("E:\AI\straight_swing_right_train.txt", delimiter=",")
test_dataset = np.loadtxt("E:\AI\THRESHOLD_TEST.txt", delimiter=",")
dataset = np.loadtxt("E:\AI\THRESHOLD_ALL.txt", delimiter=",")

X = dataset[:,0:600]
Y = dataset[:,600]

X_test = test_dataset[:,0:600]
Y_test = test_dataset[:,600]

Z = test_dataset[:,0:600]
Z1 = test_dataset[:,600]

# Создаём модель!
model = Sequential()

model.add(Dense(1024, input_dim=600, activation='sigmoid'))
model.add(Dense(256, activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))

# выходной слой с 5 - прямой, боковой, апперкот, уракен, передвижения без ударов
model.add(Dense(5, activation='sigmoid'))
dot_img_file = "E:\AI\model_1.png"
plot_model(model, to_file=dot_img_file, show_shapes=True) 

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X, Y, validation_data=(X_test, Y_test),epochs=100, batch_size=10,  verbose=1)
print(history.history.keys())  
# Обучение и проверка точности значений
#============================рисуем эпохи======================================
plt.plot(history.history['accuracy'],  color='red')

plt.plot(history.history['val_accuracy'], color='black')

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')



plt.show()

# Обучение и проверка величины потерь

plt.plot(history.history['loss'], color='red')

plt.plot(history.history['val_loss'],  color='black')

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()

# Предсказание
print('\n# Evaluate on test data')
results = model.evaluate(X, Y, batch_size=128)
print('test loss, test acc:', results)

print ("for predict = ", Z1 )
predictions = model.predict(Z)
predict = []
#print("\nModel predicts\n ", predictions)
length = len(Z)
#print("\nlength Z = \n ", length)
coinsidence = ''
for i in range (length):
    predict = max(predictions[i])
    maxPredict = [k for k,j in enumerate (predictions[i]) if j==max(predictions[i])]
    if maxPredict==Z1[i]:
        coinsidence = 'ok'
    print(i," predict = ", predict, maxPredict, coinsidence)

  
  
















