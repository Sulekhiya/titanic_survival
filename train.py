'''
Titanic: Machine Learning from Disaster
This program trains a classifier to predict the survival
of a passenger aboard the Titanic
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier
training_dataset = pd.read_csv('train.csv')

training_dataset = training_dataset[['Name','Pclass','Sex','Age','SibSp','Fare','SibSp','Survived']]
'''
Drop all the missing values
'''
print(training_dataset['Mrs.' in  str(training_dataset[['Name']]]))
#training_dataset.dropna(inplace = True)
'''
y = np.array(training_dataset['Survived'])
y1 = np.array(training_dataset['Pclass'])
training_dataset.replace(['male','female'],[-1,1],inplace = True)
factors_dataset = training_dataset[['Fare','Sex','Age','SibSp','Pclass']]
X = np.array(factors_dataset)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
X_train, y_train = X, y
log = RandomForestClassifier(n_estimators=50, max_features='sqrt')
log.fit(X_train,y_train)
y_predict = log.predict(X_test)
#print(accuracy_score(y_test,y_predict))
test_dataset = pd.read_csv('test.csv') 
test_dataset = test_dataset[['Pclass','Sex','Age','SibSp','Fare','SibSp','Cost']]
test_dataset.fillna(64,inplace = True)
test_dataset.replace(['male','female'],[-1,1],inplace = True)
test_factors_ar = np.array(test_dataset,dtype = float)
print(test_factors_ar)
prediction = log.predict(test_factors_ar)
print(prediction.shape)
copy = 'PassengerId,Survived\n'
limit = 1000000
for _ in range(limit):
    copy = copy + (str(892+_)+','+str(prediction[_])) + '\n'
fd = open('submission.csv','a')
fd.write(copy)
fd.close()
'''
