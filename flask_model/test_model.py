import joblib

# Load Model and Vectorizer
model = joblib.load("scam_detector_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Sample Scam Message
sample_text = "CONGRATULATIONS you got 1000$ click on the link"

# Convert to Vector
text_vector = vectorizer.transform([sample_text])

# Predict
prediction = model.predict(text_vector)[0]

# Output Result
result = "⚠️ Scam Detected" if prediction == 1 else "✅ Safe"
print(f"Prediction for '{sample_text}': {result}")
