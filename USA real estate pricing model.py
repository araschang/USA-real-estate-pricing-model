import pandas as pd
from sklearn.linear_model import LinearRegression

real_estate = pd.read_csv("/Users/araschang/Desktop/coding/PycharmProjects/learning/machine learning/realtor-data.csv")
print(real_estate.head())
for i in range(real_estate.shape[0]):
    if real_estate["pricepersize"][i] == '#DIV/0!':
        real_estate = real_estate.drop(i)
        print(real_estate.head())
    print(i)
lm = LinearRegression()
price = real_estate["pricepersize"].values.reshape(-1, 1)
bed = real_estate['bed'].values.reshape(-1, 1)

lm.fit(price, bed)

print("Coefficient: ", lm.coef_)
print("Intercept: ", lm.intercept_)
print("R square: ", lm.score(price, bed))
