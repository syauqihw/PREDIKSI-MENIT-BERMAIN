import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle



dataset = pd.read_csv('MenitMain_Data.csv', sep=';')

dataset['Age'].fillna(0, inplace=True)
dataset['Min'].fillna(0, inplace=True)

X = dataset.iloc[:, :1]
y = dataset.iloc[:, -1]

regressor=LinearRegression()

regressor.fit(X,y)

pickle.dump(regressor, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[23]]))