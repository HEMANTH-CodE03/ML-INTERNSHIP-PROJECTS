import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# Step 1: Load the CSV file
data = pd.read_csv("sms_spam.csv")

# Step 2: Separate input and output
X = data['message']
y = data['label']

# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Convert text into numbers using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Train the model using Naive Bayes
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test_tfidf)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# Step 7: Predict new messages
def predict_message(text):
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]

# Step 8: Test with custom messages
test_messages = [
    "Free prize if you call now!",
    "Hey! Are we still meeting at 5?",
    "Congratulations, you’ve won a free vacation!",
    "I’ll bring the files to class tomorrow."
]

print("\n🔎 Test Message Predictions:")
for msg in test_messages:
    print(f"Message: '{msg}' => Predicted: {predict_message(msg)}")
