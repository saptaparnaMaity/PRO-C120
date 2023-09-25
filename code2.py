import pandas as pd

df = pd.read_csv('income.csv')

print(df.head())
print(df.describe())

from sklearn.model_selection import train_test_split

X = df[["age", "hours-per-week", "education-num", "capital-gain", "capital-loss"]]
y = df["income"]

x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size=0.25, random_state=42)