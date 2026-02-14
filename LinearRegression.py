import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


df = pd.read_csv("areaprice.csv")
#it is used to remove any hidden spaces
df.columns = df.columns.str.strip()
print(df)

plt.xlabel("area(sq.ft)")
plt.ylabel("price(Rupees)")
plt.scatter(df['area'],df['price'],color = "red",marker = "+")
plt.savefig('LR.png')

reg = LinearRegression()
reg.fit(df[['area']],df['price'])
print(reg.predict([[2650]]))
print(reg.coef_)
print(reg.intercept_)

y = 18.97941681 * 2650 + 15600.343053173252
print(y)
