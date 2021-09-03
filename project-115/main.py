import pandas as pd
import plotly_express as px
import numpy as np
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data.csv")

velocity_values = df["Velocity"].tolist()
escape_values = df["Escaped"].tolist()

velocity = np.array(velocity_values)
escape = np.array(escape_values)

m,c = np.polyfit(velocity,escape,1)
print(m,c)

y_list = []

for i in range(0,len(velocity_values)):
    y_value = m*velocity_values[i]+c
    y_list.append(y_value)

""" fig = px.scatter(df,x="Velocity", y="Escaped")
fig.update_layout(shapes=[dict(type="line", y0=min(
    y_list), y1=max(y_list), x0=min(velocity_values), x1=max(velocity_values))])
fig.show()  """

#Sigmoid Logistic regression
score_resized = np.reshape(velocity , (len(velocity), 1))
accepted_resized = np.reshape(escape, (len(escape), 1))
lr = LogisticRegression()
lr.fit(score_resized,accepted_resized)
print("m=" , lr.coef_)
print("c=" , lr.intercept_)

def calculate(x):
    y_value = 1 / 1 + np.exp(-x)
    return y_value

user_input = int(input("Type x value: \n"))
predicted_value_of_accepted = calculate(user_input * lr.coef_ + lr.intercept_)
print(predicted_value_of_accepted)