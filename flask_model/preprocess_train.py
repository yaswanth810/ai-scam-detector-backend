import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib  # To save the trained model

# Download NLTK stopwords if not already installed
nltk.download("stopwords")

# Load cleaned dataset
df = pd.read_csv("cleaned_dataset.csv")

# Text Cleaning Function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    words = text.split()  # Tokenize
    words = [word for word in words if word not in stopwords.words("english")]  # Remove stopwords
    return " ".join(words)

# Apply text cleaning
df["clean_text"] = df["text"].apply(clean_text)

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Limit features to 5000 for efficiency
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]  # Spam = 1, Ham = 0

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save Model and Vectorizer
joblib.dump(model, "scam_detector_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("\nModel and vectorizer saved successfully!")
