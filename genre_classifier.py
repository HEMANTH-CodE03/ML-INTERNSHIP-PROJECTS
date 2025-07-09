import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

# Step 1: Load your dataset
data = pd.read_csv("movies.csv")

# Step 2: Preprocess genres into list format
data['genres'] = data['genres'].apply(lambda x: x.split('|'))

# Step 3: Encode the labels
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(data['genres'])

# Step 4: Split into training and testing
X_train_raw, X_test_raw, y_train, y_test = train_test_split(data['plot'], y, test_size=0.3, random_state=42)

# Step 5: Convert text to numeric using CountVectorizer
vectorizer = CountVectorizer(max_features=5000, stop_words='english')
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

# Step 6: Train the model
model = OneVsRestClassifier(MultinomialNB())
model.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=mlb.classes_, zero_division=0))

# Step 8: Test with new plots
def predict_genre(plot_text, threshold=0.3):
    plot_vec = vectorizer.transform([plot_text])
    probs = model.predict_proba(plot_vec)
    genres = []

    for i, p in enumerate(probs[0]):
        if p >= threshold:
            genres.append(mlb.classes_[i])
    
    return [tuple(genres)]

# Test cases
test_plots = [
    "A wizard learns magic and fights a dark lord."
    "A vampire falls in love with a human.",
    "Aliens attack Earth and a soldier saves the world.",
    "A detective investigates a kidnapping case.",
    "A young girl joins a magic school."
]

for plot in test_plots:
    print(f"\nTest: {plot}")
    print("Predicted genres:", predict_genre(plot))
