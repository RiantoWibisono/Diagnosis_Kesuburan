# ========================================================================================================
# Diagnosis Kesuburan
# ========================================================================================================
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt 

df = pd.read_csv(
    'fertility.csv',
    names = ['Season', 'Age', 'ChDi', 'AcTr', 'SuIn', 'HiFe', 'FrAl', 'SmHa', 'HoSi', 'Diagnosis'],     
    header = 0
)

df = df.drop(50)     

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
df['ChDi'] = label.fit_transform(df['ChDi'])
df['AcTr'] = label.fit_transform(df['AcTr'])
df['SuIn'] = label.fit_transform(df['SuIn'])
df['HiFe'] = label.fit_transform(df['HiFe'])
df['FrAl'] = label.fit_transform(df['FrAl'])
df['SmHa'] = label.fit_transform(df['SmHa'])
df['Diagnosis'] = label.fit_transform(df['Diagnosis'])

df = df.drop(['Season'], axis = 1)
dfX = df.drop(['Diagnosis'], axis = 1)
dfY = df['Diagnosis']

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

coltrans = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [4, 5, 6])],           
    remainder = 'passthrough'                              
)    

dfX = np.array(coltrans.fit_transform(dfX))

from sklearn.model_selection import train_test_split
xtr, xts, ytr, yts = train_test_split(
    dfX,
    dfY,
    test_size = .1
)

from sklearn.linear_model import LogisticRegression
modelLog = LogisticRegression(solver = 'liblinear')
modelLog.fit(xtr,ytr)

from sklearn.ensemble import ExtraTreesClassifier
modelExtra = ExtraTreesClassifier(n_estimators=50)
modelExtra.fit(xtr, ytr)

from sklearn.neighbors import KNeighborsClassifier
modelKNN = KNeighborsClassifier(
    n_neighbors = 10       
)
modelKNN.fit(xtr, ytr)
 
# ------------------------------------------------------------
# Prediksi data kelima pasien wanita
# Keterangan: karena deskripsi untuk setiap pasien tidak seimbang (banyak data yang diperlukan tidak ada), maka untuk nilai-nilai berikut:
# 1. Untuk child diseases, accident or serious trauma, surgical intervention, dan high fevers in the last years dianggap tidak ada / tidak pernah 
# 2. Untuk no of hours spent sitting in a day, diambil nilai normal rata-rata yaitu 7 jam per hari, kecuali untuk bebi diambil 18 jam karena bebi tidak memiliki kaki
arin = [0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 29, 0, 0, 0, 5]
bebi = [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 31, 1, 1, 1, 18]
caca = [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 25, 1, 0, 0, 7]
dini = [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 28, 0, 1, 1, 7]
enno = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 42, 1, 0, 1, 8]

def fertile(x):
    if x[0] == 0:
        return 'ALTERED'
    elif x[0] == 1:
        return 'NORMAL'

print('Arin, prediksi kesuburan: {} (Logistic Regression)'.format(fertile(modelLog.predict([arin]))))
print('Arin, prediksi kesuburan: {} (Extreme Random Forest)'.format(fertile(modelExtra.predict([arin]))))
print('Arin, prediksi kesuburan: {} (K-Nearest Neighbors)'.format(fertile(modelKNN.predict([arin]))))
print(' ')
print('Bebi, prediksi kesuburan: {} (Logistic Regression)'.format(fertile(modelLog.predict([bebi]))))
print('Bebi, prediksi kesuburan: {} (Extreme Random Forest)'.format(fertile(modelExtra.predict([bebi]))))
print('Bebi, prediksi kesuburan: {} (K-Nearest Neighbors)'.format(fertile(modelKNN.predict([bebi]))))
print(' ')
print('Caca, prediksi kesuburan: {} (Logistic Regression)'.format(fertile(modelLog.predict([caca]))))
print('Caca, prediksi kesuburan: {} (Extreme Random Forest)'.format(fertile(modelExtra.predict([caca]))))
print('Caca, prediksi kesuburan: {} (K-Nearest Neighbors)'.format(fertile(modelKNN.predict([caca]))))
print(' ')
print('Dini, prediksi kesuburan: {} (Logistic Regression)'.format(fertile(modelLog.predict([dini]))))
print('Dini, prediksi kesuburan: {} (Extreme Random Forest)'.format(fertile(modelExtra.predict([dini]))))
print('Dini, prediksi kesuburan: {} (K-Nearest Neighbors)'.format(fertile(modelKNN.predict([dini]))))
print(' ')
print('Enno, prediksi kesuburan: {} (Logistic Regression)'.format(fertile(modelLog.predict([enno]))))
print('Enno, prediksi kesuburan: {} (Extreme Random Forest)'.format(fertile(modelExtra.predict([enno]))))
print('Enno, prediksi kesuburan: {} (K-Nearest Neighbors)'.format(fertile(modelKNN.predict([enno]))))
print(' ')