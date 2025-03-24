import pandas as pd
import numpy as np
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ✅ Ensure stopwords are available
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# ✅ Function to Clean Text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words)

# ✅ Function to Extract Extra Features
def extract_features(text):
    spam_keywords = ["win", "prize", "free", "money", "click", "congratulations", "offer"]
    contains_link = 1 if "http" in text or "www." in text else 0
    spam_word_count = sum(1 for word in text.split() if word in spam_keywords)
    return [spam_word_count, contains_link]

# ✅ Load Dataset
try:
    df = pd.read_csv("scam_dataset.csv", encoding="latin1")  # Fix encoding issue
    print("✅ Dataset Loaded Successfully!")
except Exception as e:
    print(f"❌ Error Loading Dataset: {e}")
    exit()

# ✅ Print available columns to verify
print("✅ Dataset Columns:", df.columns.tolist())

# ✅ Force correct column selection for your dataset
text_column = "v2"  # v2 contains the message text
label_column = "v1"  # v1 contains the label (ham/spam)

# ✅ Check if columns exist
if text_column not in df.columns or label_column not in df.columns:
    print("❌ Required columns not found in dataset!")
    exit()

# ✅ Clean text and process labels
df["cleaned_text"] = df[text_column].astype(str).apply(clean_text)

# ✅ Convert labels: "spam" → 1, "ham" → 0
df[label_column] = df[label_column].map({"spam": 1, "ham": 0})

# ✅ Drop any missing values
df = df.dropna(subset=["cleaned_text", label_column])

# ✅ Apply Cleaning & Feature Extraction
extra_features = df["cleaned_text"].apply(lambda x: extract_features(x))

# ✅ Convert Text into Numerical Features (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)  # Ensure fixed feature count
X_tfidf = vectorizer.fit_transform(df["cleaned_text"])

# ✅ Combine TF-IDF + Extra Features
X = np.hstack([X_tfidf.toarray(), np.array(extra_features.tolist())])
y = df[label_column].values  # Label column (1 = Scam, 0 = Safe)

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"✅ Model Accuracy: {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F1 Score: {f1:.4f}")

# ✅ Save Model & Vectorizer
joblib.dump(model, "scam_detector_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("✅ Model and vectorizer saved successfully!")
