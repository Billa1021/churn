

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

data = pd.read_excel(r'CHURNDATA.xlsx')

data['CUS_Month_Income'] = data.CUS_Month_Income.fillna(data['CUS_Month_Income'].mean())

data['CUS_Gender'] = data['CUS_Gender'].ffill()

data = data.drop('CIF',axis = 1)
data = data.drop('CUS_DOB',axis = 1)
data = data.drop('CUS_Target',axis=1)
data = data.drop('CUS_Customer_Since',axis = 1)
data = data.drop('# total debit transactions for S1',axis = 1)
data = data.drop('# total debit transactions for S2',axis = 1)
data = data.drop('# total debit transactions for S3',axis = 1)
data = data.drop('# total credit transactions for S1',axis = 1)
data = data.drop('# total credit transactions for S2',axis = 1)
data = data.drop('# total credit transactions for S3',axis = 1)
data = data.drop('total debit amount for S1',axis = 1)
data = data.drop('total debit amount for S2',axis = 1)
data = data.drop('total debit amount for S3',axis = 1)
data = data.drop('total credit amount for S1',axis = 1)
data = data.drop('total credit amount for S2',axis = 1)
data = data.drop('total credit amount for S3',axis = 1)
data = data.drop('total transactions',axis = 1)


label = LabelEncoder()
data['CUS_Gender'] = label.fit_transform(data['CUS_Gender'])
data['CUS_Marital_Status'] = label.fit_transform(data['CUS_Marital_Status'])
data['TAR_Desc'] = label.fit_transform(data['TAR_Desc'])
data['Status'] = label.fit_transform(data['Status'])

X = data
X = X.drop('Status',axis = 1)

y = pd.DataFrame(data['Status'])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


reg = LogisticRegression(class_weight= {0:1,1:3})

reg.fit(X_train,y_train)


# make predictions

expected = y_test
predicted = reg.predict(X_test)

# summarize the fit of the model
#Correction

metrics.classification_report(expected, predicted)
metrics.confusion_matrix(expected, predicted)

import pickle

pickle.dump(model_GYM, open("reg.pkl", "wb"))

model = pickle.load(open("reg.pkl", "rb"))

print(model.predict([[40,5.6,70]]))








