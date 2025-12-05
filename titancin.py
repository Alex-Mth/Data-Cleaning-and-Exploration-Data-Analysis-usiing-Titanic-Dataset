# Titanic Data Cleaning and EDA (FINAL FIXED VERSION)
# ---------------------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)

# -------------------------------
# ✅ STEP 1: AUTO FILE DETECTION
# -------------------------------
print("\n--- SEARCHING FOR DATASET FILES ---")

possible_paths = [
    os.getcwd(),
    os.path.dirname(__file__),
    os.path.join(os.getcwd(), "data"),
    os.path.join(os.path.dirname(__file__), "data")
]

train_path = None
test_path = None

for path in possible_paths:
    if os.path.exists(os.path.join(path, "train.csv")):
        train_path = os.path.join(path, "train.csv")
    if os.path.exists(os.path.join(path, "test.csv")):
        test_path = os.path.join(path, "test.csv")

# ❌ If files still not found → STOP execution safely
if train_path is None or test_path is None:
    print("\n❌ ERROR: train.csv or test.csv not found!")
    print("\n➡️ Place both files in one of these locations:\n")
    for p in possible_paths:
        print(" -", p)
    print("\n➡️ Then run the script again.")
    exit()

print("\n✅ Train file found at:", train_path)
print("✅ Test file found at:", test_path)

# -------------------------------
# ✅ STEP 2: LOAD DATA
# -------------------------------
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# -------------------------------
# ✅ STEP 3: BASIC INFO
# -------------------------------
print("\n--- Dataset Info ---")
train.info()

print("\n--- First 5 Rows ---")
print(train.head())

# -------------------------------
# ✅ STEP 4: MISSING VALUES
# -------------------------------
print("\n--- Missing Values (Before Cleaning) ---")
print(train.isnull().sum())

# -------------------------------
# ✅ STEP 5: DATA CLEANING
# -------------------------------
train['Age'] = train['Age'].fillna(train['Age'].median())
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])

if 'Cabin' in train.columns:
    train = train.drop('Cabin', axis=1)

test['Age'] = test['Age'].fillna(test['Age'].median())
test['Fare'] = test['Fare'].fillna(test['Fare'].median())

if 'Cabin' in test.columns:
    test = test.drop('Cabin', axis=1)

# -------------------------------
# ✅ STEP 6: VERIFY CLEANING
# -------------------------------
print("\n--- Missing Values After Cleaning ---")
print(train.isnull().sum())

# -------------------------------
# ✅ STEP 7: SUMMARY STATISTICS
# -------------------------------
print("\n--- Summary Statistics ---")
print(train.describe())

# -------------------------------
# ✅ STEP 8: SURVIVAL COUNTS
# -------------------------------
print("\n--- Survival Counts ---")
print(train['Survived'].value_counts())

# -------------------------------
# ✅ STEP 9: EDA VISUALIZATIONS
# -------------------------------
sns.set_theme(style="whitegrid")

# 1️⃣ Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(data=train, x='Sex', hue='Survived')
plt.title('Survival Count by Gender')
plt.tight_layout()
plt.show()

# 2️⃣ Survival by Passenger Class
plt.figure(figsize=(6,4))
sns.countplot(data=train, x='Pclass', hue='Survived')
plt.title('Survival Count by Passenger Class')
plt.tight_layout()
plt.show()

# 3️⃣ Age Distribution
plt.figure(figsize=(6,4))
sns.histplot(train['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.tight_layout()
plt.show()

# 4️⃣ Age vs Survival
plt.figure(figsize=(6,4))
sns.boxplot(x='Survived', y='Age', data=train)
plt.title('Age vs Survival')
plt.tight_layout()
plt.show()

# 5️⃣ Correlation Heatmap
plt.figure(figsize=(8,6))
corr = train.select_dtypes(include=['number']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# -------------------------------
# ✅ STEP 10: KEY INSIGHTS
# -------------------------------
print("\n--- Key Insights ---")
print("1. Females had a significantly higher survival rate than males.")
print("2. Passengers in 1st class had the highest survival probability.")
print("3. Younger passengers had slightly better survival chances.")
print("4. Fare and passenger class strongly affect survival.")

print("\n✅ Titanic EDA Completed Successfully!")
