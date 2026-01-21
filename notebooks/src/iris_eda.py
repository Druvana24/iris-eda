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


# Histogram of sepal length
sns.histplot(df["sepal length (cm)"], kde=True)
plt.title("Sepal Length Distribution")
plt.show()

# Boxplot of petal length by species
sns.boxplot(x="species", y="petal length (cm)", data=df)
plt.title("Petal Length by Species")
plt.show()

# Pairplot of all features colored by species
sns.pairplot(df, hue="species")
plt.suptitle("Pairplot of Iris Features", y=1.02)
plt.show()

# Correlation heatmap for numeric features
plt.figure(figsize=(6, 4))
corr = df.drop(columns=["species"]).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Iris Features")
plt.tight_layout()
plt.show()
