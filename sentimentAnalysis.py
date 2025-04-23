import streamlit as st
import joblib
from deep_translator import GoogleTranslator

# Load separately saved vectorizer and model
vectorizer = joblib.load("tfidf_vectorizer.pkl")  # Make sure this file exists
model = joblib.load("svm_model.pkl")              # Should be a classifier like LogisticRegression

# Function to translate and predict sentiment
def analyze_sentiment(review):
    if review:
        # Translate review to Malay
        translated_review = GoogleTranslator(source='auto', target='ms').translate(review)

        # Transform the review using the vectorizer
        review_vectorized = vectorizer.transform([translated_review])  # Output is 2D

        # Predict sentiment
        sentiment = model.predict(review_vectorized)[0]
        return sentiment, translated_review
    return None, None

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("📝 Sentiment Analysis App")
st.write("This app translates your review to **Malay** and predicts its sentiment.")

user_input = st.text_area("Enter your review below:")

if st.button("Analyze"):
    sentiment, translated = analyze_sentiment(user_input)
    if sentiment:
        st.markdown("### 🗣️ Translated Review")
        st.code(translated, language='markdown')

        st.markdown("### 📊 Sentiment Prediction")
        if sentiment == 'Positive':
            st.success(f"It is a **{sentiment}** review.")
        elif sentiment == 'Neutral':
            st.info(f"It is a **{sentiment}** review.")
        elif sentiment == 'Negative':
            st.error(f"It is a **{sentiment}** review.")
        else:
            st.warning(f"Unexpected prediction: {sentiment}")
    else:
        st.warning("Please enter a review.")
