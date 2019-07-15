# ========================================================================================================
# Diagnosis Kesuburan
# ========================================================================================================
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt 

df = pd.read_csv(
    'fertility.csv',
    names = ['Season', 'Age', 'ChDi', 'AcTr', 'SuIn', 'HiFe', 'FrAl', 'SmHa', 'HoSi', 'Diagnosis'],     # merupakan nama singkatan dari nama2 kolom yang terdapat di dataframe
    header = 0
)

# ------------------------------------------------------------
# Mencari tahu pilihan isi dari setiap kolom
# print(list(dict.fromkeys(df['Season'])))    # ['spring', 'fall', 'winter', 'summer']
# print(list(dict.fromkeys(df['Age'])))       # [30, 35, 27, 32, 36, 29, 33, 28, 31, 34]
# print(list(dict.fromkeys(df['ChDi']))) # ['no', 'yes']
# print(list(dict.fromkeys(df['AcTr'])))    # ['yes', 'no']
# print(list(dict.fromkeys(df['SuIn']))) # ['yes', 'no']
# print(list(dict.fromkeys(df['HiFe'])))  # ['more than 3 months ago', 'less than 3 months ago', 'no']
# print(list(dict.fromkeys(df['FrAl'])))  # ['once a week', 'hardly ever or never', 'several times a week', 'several times a day', 'every day']
# print(list(dict.fromkeys(df['SmHa']))) # ['occasional', 'daily', 'never']
# print(list(dict.fromkeys(df['HoSi']))) # [16, 6, 9, 7, 8, 5, 2, 11, 3, 342, 14, 18, 10, 1]        --> data dengan nilai '342' adalah sebuah outlier, maka harus dihapus
df = df.drop(50)        # membuang data index ke = 50 yang nilai HoSi -nya merupakan sebuah outlier
# print(list(dict.fromkeys(df['Diagnosis']))) # ['Normal', 'Altered']

# ------------------------------------------------------------
# Label Encoder
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()

df['ChDi'] = label.fit_transform(df['ChDi'])
# print(label.classes_)       # ['no' 'yes']

df['AcTr'] = label.fit_transform(df['AcTr'])
# print(label.classes_)       # ['no' 'yes']

df['SuIn'] = label.fit_transform(df['SuIn'])
# print(label.classes_)       # ['no' 'yes']

df['HiFe'] = label.fit_transform(df['HiFe'])
# print(label.classes_)       # ['less than 3 months ago' 'more than 3 months ago' 'no']

df['FrAl'] = label.fit_transform(df['FrAl'])
# print(label.classes_)       # ['every day' 'hardly ever or never' 'once a week' 'several times a day' 'several times a week']

df['SmHa'] = label.fit_transform(df['SmHa'])
# print(label.classes_)       # ['daily' 'never' 'occasional']

df['Diagnosis'] = label.fit_transform(df['Diagnosis'])
# print(label.classes_)       # ['Altered' 'Normal']

df = df.drop(['Season'], axis = 1)
dfX = df.drop(['Diagnosis'], axis = 1)
dfY = df['Diagnosis']

# ------------------------------------------------------------
# One Hot Encoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

coltrans = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [4, 5, 6])],           
    remainder = 'passthrough'                              
)    

dfX = np.array(coltrans.fit_transform(dfX))

'''
print(dfX[0])       
^ menghasilkan:
    [  0.      1.     0.       0.    0.    1.      0.       0.      0.   0.   1.   30.   0.   1.   1.   16.]     , dimana:
    ||<3mths >3mnths noFev||evryDay nvr 1/week svrl/day svrl/week||daily nvr occs||Age  ChDi AcTr SuIn HoSi
'''

# ------------------------------------------------------------
# Splitting dataset
from sklearn.model_selection import train_test_split
xtr, xts, ytr, yts = train_test_split(
    dfX,
    dfY,
    test_size = .1
)

# ------------------------------------------------------------
# Logistic Regression
from sklearn.linear_model import LogisticRegression
modelLog = LogisticRegression(solver = 'liblinear')
modelLog.fit(xtr,ytr)

# ------------------------------------------------------------
# Extreme Random Forest
from sklearn.ensemble import ExtraTreesClassifier
modelExtra = ExtraTreesClassifier(n_estimators=50)
modelExtra.fit(xtr, ytr)

# ------------------------------------------------------------
# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
modelKNN = KNeighborsClassifier(
    n_neighbors = 10        # n neighbors = sqrt(jumlah data) = sqrt(100) = 10
)
modelKNN.fit(xtr, ytr)
