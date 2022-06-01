from pandas import read_csv
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = read_csv("seattle-weather.csv")

df= df.drop('date',axis=1)

x = df.drop('weather',axis=1)
y = df['weather']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

logmodel = LogisticRegression(random_state = 0)
logmodel.fit(X_train, y_train)

prediction = logmodel.predict(X_test)

dump(logmodel, "SeattleModel.pkl")


