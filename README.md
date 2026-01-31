# CodeAlpha_PulseAI-1
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load sample dataset (built-in)
from sklearn.datasets import load_iris
iris = load_iris()

# Create DataFrame
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

# -------------------------
# Basic Information
# -------------------------
print("First 5 rows:")
display(data.head())

print("\nDataset Info:")
data.info()

print("\nStatistical Summary:")
display(data.describe())

# -------------------------
# Check Missing Values
# -------------------------
print("\nMissing Values:")
print(data.isnull().sum())

# -------------------------
# Target Distribution
# -------------------------
plt.figure()
data['target'].value_counts().plot(kind='bar')
plt.title("Target Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# -------------------------
# Histograms
# -------------------------
data.hist(figsize=(10, 8))
plt.suptitle("Feature Distributions")
plt.show()

# -------------------------
# Boxplots (Outlier Detection)
# -------------------------
plt.figure(figsize=(10, 6))
sns.boxplot(data=data)
plt.title("Boxplot of Features")
plt.show()

# -------------------------
# Correlation Heatmap
# -------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# -------------------------
# Pairplot
# -------------------------
sns.pairplot(data, hue="target")
plt.show()
