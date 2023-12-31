###Random forest regression
https://shahedkaraki.github.io/Random-Forest-Regression/


import numpy as np # linear algebra
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

traindata=pd.read_csv('/kaggle/input/testandtrain-dataset/train.csv')
testdata=pd.read_csv('/kaggle/input/testandtrain-dataset/test.csv')


traindata.drop('id',axis=1,inplace=True)

id=testdata['id']
testdata.drop('id',axis=1,inplace=True)


x=traindata.drop("smoking",axis=1)
y=traindata["smoking"]

x_train, _, y_train, _= train_test_split(x, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=150, random_state=42)

rf_model.fit(x_train, y_train)

y_predict = rf_model.predict(testdata)

predict_data=pd.DataFrame({'id':id,'smoking':y_predict})

predict_data.to_csv('predictRFR4.csv',index=False)
