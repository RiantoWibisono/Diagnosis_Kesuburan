import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt 

df = pd.read_csv(
    'fertility.csv',
    names = ['Season', 'Age', 'ChDi', 'AcTr', 'SuIn', 'HiFe', 'FrAl', 'SmHa', 'HoSi', 'Diagnosis'],     
    header = 0
)

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
# print(xts[0].reshape(1, -1))
# print(modelLog.predict(xts[0].reshape(1, -1)))

from sklearn.ensemble import ExtraTreesClassifier
modelExtra = ExtraTreesClassifier(n_estimators=50)
modelExtra.fit(xtr, ytr)

from sklearn.neighbors import KNeighborsClassifier
modelKNN = KNeighborsClassifier(
    n_neighbors = 10       
)
modelKNN.fit(xtr, ytr)
 
# ============================================================================================================
# Prediksi data kelima pasien wanita
dfPasien = pd.read_csv(
    'pasien.csv',
    delimiter = ';',
    names = ['Age', 'ChDi', 'AcTr', 'SuIn', 'HiFe', 'FrAl', 'SmHa', 'HoSi'],     
    header = 0,
    index_col = 0,
    na_values = '-'
)

# print(dfPasien.isnull().sum())

# Membersihkan data input --> mengisi sel-sel yang tidak berisi data dengan nilai-nilai tertentu
dfPasien = dfPasien.fillna({    
    'ChDi' : 'no',      # dianggap tidak memiliki penyakit masa kecil / childhood diseases
    'AcTr' : 'no',      # dianggap tidak pernah mengalami kecelakaan / accident or serious trauma
    'SuIn' : 'no',      # dianggap tidak melakukan operasi bedah / surgical intervention
    'HiFe' : 'no',      # dianggap tidak memiliki demam di tahun lalu / high fevers in the last year
    'HoSi' : 7          # asumsi rata-rata waktu yang dihabiskan untuk duduk bagi kebanyakan orang / Number of hours spent sitting per day
})

# print(dfPasien)
# print(dfPasien.index[0])

# ============================================================================================================
# Label Encoder
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
dfPasien['ChDi'] = label.fit_transform(dfPasien['ChDi'])
print(label.classes_) # ['no' 'yes']
dfPasien['AcTr'] = label.fit_transform(dfPasien['AcTr'])
print(label.classes_)  # ['no' 'yes']
dfPasien['SuIn'] = label.fit_transform(dfPasien['SuIn'])
print(label.classes_)   # ['no' 'yes']
dfPasien['HiFe'] = label.fit_transform(dfPasien['HiFe'])
print(label.classes_)   # ['less than 3 months ago' 'no']
dfPasien['FrAl'] = label.fit_transform(dfPasien['FrAl'])
print(label.classes_)   # ['every day' 'hardly ever or never' 'several times a week']
dfPasien['SmHa'] = label.fit_transform(dfPasien['SmHa'])
print(label.classes_)   # ['daily' 'never']

print(dfPasien)
arrayPasien = []

#||<3mths >3mnths noFev||evryDay nvr 1/week svrl/day svrl/week||daily nvr occs||Age  ChDi AcTr SuIn HoSi
for i in range(len(dfPasien)): 
    if dfPasien['HiFe'][i] == 0:
        arrayPasien[i].append = [1, 0, 0]
    elif dfPasien['HiFe'][i] == 1:
        arrayPasien[i].append = [0, 0, 1]

print(arrayPasien)
























# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer

# coltrans = ColumnTransformer(
#     [('one_hot_encoder', OneHotEncoder(categories='auto'), [4, 5, 6])],           
#     remainder = 'passthrough'                              
# )    

# print(dfPasien)

# arrayPasien = np.array(coltrans.fit_transform(df))
# print(arrayPasien[0])
# print(arrayPasien[0])
# print(modelLog.predict([arrayPasien[0]]))
# ||<3mths >3mnths noFev||evryDay nvr 1/week svrl/day svrl/week||daily nvr occs||Age  ChDi AcTr SuIn HoSi

# arrayPasien = []
# for i in range(16):
#     if dfPasien[i]['HiFe'] = 0, 