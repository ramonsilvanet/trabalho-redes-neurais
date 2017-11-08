import pandas as pd
import numpy as np 
import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


colunas = ['ESCT','NDEP','RENDA', 'TIPOR', 'VBEM', 'NPARC', 'VPARC', 'TEL', 'IDADE', 'RESMS', 'ENTRADA', 'CLASSE']

data = pd.read_csv('dados/credtrain.txt', header=None, sep='\t', names=colunas)  
data.head()

dataset = data.values
P = dataset[:,0:11].astype(float)
T = dataset[:,11]

print(P.shape, T.shape)

test = pd.read_csv('dados/credtest.txt', header=None, sep='\t', names=colunas)  
dataset_test = test.values

P_test = dataset_test[:,0:11].astype(float)
T_test = dataset_test[:,11]

encoder = LabelEncoder()
encoder.fit(T)
encoded_T = encoder.transform(T)

def criar_modelo():
    model = Sequential()
    model.add(Dense(32, input_dim=11, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=criar_modelo, epochs=100, batch_size=300, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=50, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, P, encoded_T, cv=kfold)

print("Precisão: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))