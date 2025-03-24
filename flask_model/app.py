from flask import Flask, request, jsonify
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from flask_cors import CORS
import numpy as np

# ✅ Ensure stopwords are available
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# ✅ Initialize Flask App
app = Flask(__name__)
CORS(app, resources={r"/check": {"origins": "*"}})  # ✅ Allow frontend access

# ✅ Function to Load Model and Vectorizer
def load_model():
    global model, vectorizer
    try:
        model = joblib.load("scam_detector_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        print("✅ Model Loaded Successfully!")
    except Exception as e:
        print(f"❌ Error loading model/vectorizer: {e}")
        exit()

# ✅ Load model at startup
load_model()

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

# ✅ API Endpoint for Scam Detection
@app.route("/check", methods=["POST"])
def check_scam():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # ✅ Clean & Process Text
        cleaned_text = clean_text(text)
        print(f"🔍 Processed Text: {cleaned_text}")  # Debugging

        # ✅ Convert Text into Numerical Features
        text_vector = vectorizer.transform([cleaned_text]).toarray()
        extra_features = np.array(extract_features(cleaned_text)).reshape(1, -1)
        
        # ✅ Ensure Feature Consistency
        full_features = np.hstack([text_vector, extra_features])
        print(f"✅ Input Features at Prediction: {full_features.shape[1]}")  # Debugging

        # ✅ Predict & Adjust Decision Threshold
        scam_probability = model.predict_proba(full_features)[0][1]  # Probability of scam
        prediction = 1 if scam_probability > 0.4 else 0  # Lower threshold from 0.5 to 0.4
        print(f"✅ Scam Probability: {scam_probability:.4f}")  # Debugging

        # ✅ Return Response
        result = "⚠️ Scam Detected" if prediction == 1 else "✅ Safe"
        return jsonify({"result": result})

    except Exception as e:
        print(f"❌ Server Error: {str(e)}")  # Debugging
        return jsonify({"error": f"Server Error: {str(e)}"}), 500

# ✅ Endpoint to Reload Model
@app.route("/reload", methods=["GET"])
def reload_model():
    try:
        load_model()
        return jsonify({"message": "✅ Model reloaded successfully!"})
    except Exception as e:
        return jsonify({"error": f"❌ Error reloading model: {str(e)}"}), 500

# ✅ Run the Flask App
if __name__ == "__main__":
    app.run(debug=True)
