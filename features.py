import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

# 1. Φόρτωση Δεδομένων
print("=== Loading Data ===")
df = pd.read_csv("data/cleaned_data.csv")
print(f"Dataset Shape: {df.shape}")

# 2. Έλεγχος για NaN (Ελλιπείς Τιμές)
print("\n=== Checking for NaN Values ===")
nan_values = df.isnull().sum()
if nan_values.sum() == 0:
    print("Δεν βρέθηκαν κενές τιμές (No NaN values found).")
else:
    print(nan_values[nan_values > 0])

# 3. Έλεγχος για Outliers (Ακραίες Τιμές)
# Χρησιμοποιούμε boxplots για να δούμε αν υπάρχουν τιμές εκτός ορίων
plt.figure(figsize=(15, 8))
# Επιλέγουμε μερικά βασικά χαρακτηριστικά για το γράφημα (για να μην γεμίσει η οθόνη)
cols_to_plot = df.drop(columns=["Class", "PathOrder"]).columns[:10] 
df[cols_to_plot].boxplot()
plt.title("Boxplot for Outlier Detection (First 10 features)")
plt.xticks(rotation=45)
plt.show()

# 4. Κατανομή Κλάσεων (Class Balance)
print("\n=== Class Distribution ===")
print(df["Class"].value_counts(normalize=True)) # Ποσοστά ανά οδηγό
sns.countplot(x='Class', data=df)
plt.title("Distribution of Driver Classes")
plt.show()

# 5. Correlation Analysis (Συσχέτιση)
# Βλέπουμε ποια κανάλια "κινούνται" μαζί
print("\n=== Computing Correlation Matrix ===")
plt.figure(figsize=(12, 10))
features_only = df.drop(columns=["Class", "PathOrder"])
corr_matrix = features_only.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# 6. Feature Selection με Mutual Information
# Αυτό απαντάει άμεσα στην ερώτηση του καθηγητή: "Με βάση ποια μέθοδο;"
print("\n=== Calculating Mutual Information Score ===")
X = features_only
y = df["Class"]

# Κωδικοποίηση του target για το Mutual Info
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Υπολογισμός MI
mi_scores = mutual_info_classif(X, y_encoded, random_state=42)
mi_results = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

print("\nTop 10 Σημαντικότερα Κανάλια (βάσει Mutual Information):")
print(mi_results.head(10))

# Γράφημα Σημαντικότητας
plt.figure(figsize=(10, 6))
mi_results.head(15).plot(kind='barh')
plt.title("Top 15 Features by Mutual Information")
plt.xlabel("Mutual Information Score")
plt.gca().invert_yaxis()
plt.show()

