# Simple linear regression in Python
# Using data I found in the internet where one column states the height
# an the second column states the weight

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\caio_\Desktop\ai_and_python\linear-regression\weight-height.csv")
data_male = data[data["Gender"] == "Male"].drop("Gender", axis = 1)
data_female = data[data["Gender"] == "Female"].drop("Gender", axis = 1)

x = data_male['Height']
y = data_male['Weight']

def linear_regression(x, y):
    n = len(x)
    x_mean = x.mean()
    y_mean = y.mean()

    numerator = ((x - x_mean) * (y - y_mean)).sum()
    denominator = ((x - x_mean)**2).sum()

    slope = numerator/denominator

    intercept = y_mean - (slope * x_mean)

    return(numerator, slope)

slope, intercept = linear_regression(x, y)

def function_line(slope, intercept):
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color="yellow")
    plt.show()

function_line(slope, intercept)
plt.scatter(x, y, color="red")
plt.show()






