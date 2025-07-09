import pandas as pd

# Only run this block ONCE to create the sample dataset
sample_data = {
    'amount': [100, 2500, 75, 20000, 120, 50000, 80, 6000, 95, 100000],
    'time': [10, 200, 5, 600, 15, 800, 7, 300, 8, 1000],
    'is_fraud': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 1 = Fraud, 0 = Legit
}
df = pd.DataFrame(sample_data)
df.to_csv("creditcard.csv", index=False)
print("✅ Sample dataset created: creditcard.csv")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load the dataset
data = pd.read_csv("creditcard.csv")

# Step 2: Separate features and labels
X = data[['amount', 'time']]
y = data['is_fraud']

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Step 6: Predict and evaluate
y_pred = model.predict(X_test_scaled)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
