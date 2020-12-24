from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

data = pd.read_csv("insurance.csv")

# plt.style.use("fivethirtyeight")

x = data.iloc[:, :-1]
x_no_discrete = x.iloc[:, [0, 2, 3]].values
y = data.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x_no_discrete, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})

print("Mean Absolute Error: ", metrics.mean_absolute_error(y_test, y_pred))