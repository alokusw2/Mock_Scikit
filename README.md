# Mock_Scikit

1 problem 1 : https://tinyurl.com/scikitMockInterviewQuestion


-----------------------------------------------

As a data scientist at a telecommunications company, you have been tasked with developing a classification model to predict customer churn (cancellation of subscriptions) based on their historical behavior and demographic information. The company wants to understand which customers are likely to churn in order to develop targeted customer retention programs.

Dataset: You have been provided with a dataset named "Telco-Customer-Churn.csv". Each row in the dataset represents a customer, and each column contains attributes related to the customer's services, account information, and demographic details. The dataset includes the following information:

------------------------------------------------

import pandas as pd
import numpy as np
from  sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import math


dataset = pd.read_csv('Telco-Customer-Churn.csv')

# Convert 'TotalCharges' to numeric, coercing errors which will turn non-numeric values into NaN
dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'], errors = 'coerce')

dataset['TotalCharges'].fillna(dataset['TotalCharges'].mean(), inplace = True)



# Fill missing values (NaN) in 'TotalCharges' with the mean
dataset['TotalCharges'].fillna(dataset['TotalCharges'].mean(), inplace=True)


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

colu = ['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','Churn']
for i in colu:
   dataset[i]=le.fit_transform(dataset[i])

#  dataset['gender'] = le.fit_transform(dataset['gender'])

# dataset.head()
# dataset.info()

# x= dataset.drop(['customerID','Churn'],axis=1)
# y=dataset['Churn']

x= dataset.iloc[:, 1:20]
y= dataset.iloc[:, 20:21]
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

log = LogisticRegression()
log.fit(x_train, y_train)
y_pred = log.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("logisticRegression accuracy:", accuracy)

knn = KNeighborsClassifier(n_neighbors=75)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("KNeighborsClassifier accuracy:", accuracy)


sv = SVC()
sv.fit(x_train, y_train)
y_pred = sv.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("SVC accuracy:", accuracy)

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("DecisionTreeClassifier accuracy:",  accuracy)

rf = RandomForestClassifier(n_estimators=100, random_state= 42)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("RandomForestClassifier accuracy:", accuracy)


# customerID    gender  SeniorCitizen   Partner Dependents      tenure  PhoneService    MultipleLines   InternetService OnlineSecurity  ...     DeviceProtection        TechSupport     StreamingTV     StreamingMovies Contract        PaperlessBilling        PaymentMethod   MonthlyCharges  TotalCharges    Churn
# 0     7590-VHVEG      Female  0       Yes     No      1       No      No phone service        DSL     No      ...     No      No      No      No      Month-to-month  Yes     Electronic check        29.85   29.85   No
# 1     5575-GNVDE      Male    0       No      No      34      Yes     No      DSL     Yes     ...     Yes     No      No      No      One year        No      Mailed check    56.95   1889.5  No
# 2     3668-QPYBK      Male    0       No      No      2       Yes     No      DSL     Yes     ...     No      No      No      No      Month-to-month  Yes     Mailed check    53.85   108.15  Yes
# 3     7795-CFOCW      Male    0       No      No      45      No      No phone service        DSL     Yes     ...     Yes     Yes     No      No      One year        No      Bank transfer (automatic)       42.30   1840.75 No
# 4     9237-HQITU      Female  0       No      No      2       Yes     No      Fiber optic     No      ...     No      No      No      No      Month-to-month  Yes     Electronic check        70.70   151.65  Yes
# 5 rows × 21 columns