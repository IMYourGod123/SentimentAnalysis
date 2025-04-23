# sentimentAnalysis.py
import streamlit as st
import joblib
import pandas as pd
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

# Display demo sentiment comparison table
demo_data = [
    {"Review": "saya not feeling lucky today", "Actual Sentiment": "Negative", "SVM": "Positive"},
    {"Review": "that parcel buruk and x betul", "Actual Sentiment": "Negative", "SVM":"Negative"},
    {"Review": "baju ni okay je pakai", "Actual Sentiment": "Neutral", "SVM":"Neutral"},
    {"Review": "I like this baju melayu. Customer service sangat baik, dan delivery pun cepat. tq", "Actual Sentiment": "Positive", "SVM":"Positive"},
    {"Review": "baju ni okay jer dipakai, kualiti match dgn harga", "Actual Sentiment": "Neutral", "SVM":"Positive"},
    {"Review": "Kain baju ni teruk ... penghantaran lambat, tak kan repeat order lagi ...", "Actual Sentiment": "Negative", "SVM":"Negative"},
    {"Review": "baju ni cepat sampai dan baik", "Actual Sentiment": "Positive", "SVM":"Positive"},
    {"Review": "baju ni kotor dan dont have quality", "Actual Sentiment": "Negative", "SVM":"Positive"},
    {"Review": "sizing tidak betul aku rasa sedih", "Actual Sentiment": "Negative", "SVM":"Negative"},
    {"Review": "delivery laju kain cantik thanks seller", "Actual Sentiment": "Positive", "SVM":"Positive"},

]

# Create a DataFrame and add predicted sentiment
df = pd.DataFrame(demo_data)
df["Predicted Sentiment"] = df["Review"].apply(
    lambda r: model.predict([GoogleTranslator(source='auto', target='ms').translate(r)])[0]
)

# Display demo table
st.markdown("### 🧾 Demo Sentiment Table")
st.dataframe(df)

# User input section
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
