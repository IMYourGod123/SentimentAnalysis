import streamlit as st
import joblib
from deep_translator import GoogleTranslator

# Load the pre-trained pipeline model
model = joblib.load("svm_model.pkl")  # This includes both vectorizer and classifier

# Sentiment analysis function
def analyze_sentiment(review):
    if review:
        # Translate review to Malay
        translated_review = GoogleTranslator(source='auto', target='ms').translate(review)

        # Predict using the loaded pipeline
        sentiment = model.predict([translated_review])[0]
        return sentiment, translated_review
    return None, None

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("📝 Sentiment Analysis App")
st.write("This app translates your review to Malay and predicts its sentiment.")

user_input = st.text_area("Enter your review below:")

if st.button("Analyze"):
    sentiment, translated = analyze_sentiment(user_input)
    if sentiment:
        st.markdown(f"**Translated Review:** `{translated}`")
        if sentiment == 'Positive':
            st.success(f"It is a **{sentiment}** review.")
        elif sentiment == 'Neutral':
            st.info(f"It is a **{sentiment}** review.")
        elif sentiment == 'Negative':
            st.error(f"It is a **{sentiment}** review.")
    else:
        st.warning("Please enter a review.")
