import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target
df["species"] = df["species"].map(dict(zip(range(3), iris.target_names)))

print(df.head()


print("Shape:", df.shape)
print("\nInfo:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

print("\nClass balance:")
print(df["species"].value_counts())
