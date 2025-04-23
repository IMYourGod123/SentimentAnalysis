# sentimentAnalysis.py
import streamlit as st
import joblib
from deep_translator import GoogleTranslator

# Load the saved pipeline (includes both TF-IDF and classifier)
model = joblib.load("svm_pipeline_model.pkl")

# Function to translate and predict sentiment
def analyze_sentiment(review):
    if review:
        # Translate to Malay
        translated_review = GoogleTranslator(source='auto', target='ms').translate(review)

        # Predict using the pipeline
        sentiment = model.predict([translated_review])[0]
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
