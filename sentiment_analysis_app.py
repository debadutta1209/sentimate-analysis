from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Streamlit App Header
st.title('Real-Time Sentiment Analysis for Customer Feedback')

# Plot Style
plt.style.use('ggplot')

# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')

# Initialize VADER Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Load or Download RoBERTa Model
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    return tokenizer, model

tokenizer, model = load_model()

# Function to compute sentiment scores using RoBERTa
def polarity_scores_roberta(text):
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }

# Function to classify sentiment based on VADER and RoBERTa results
def classify_sentiment(vader_result, roberta_result):
    if vader_result['compound'] >= 0.05 or roberta_result['roberta_pos'] > max(roberta_result['roberta_neg'], roberta_result['roberta_neu']):
        return "Positive"
    elif vader_result['compound'] <= -0.05 or roberta_result['roberta_neg'] > max(roberta_result['roberta_pos'], roberta_result['roberta_neu']):
        return "Negative"
    else:
        return "Neutral"

# Sidebar for Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose Mode", ["Single Feedback", "Bulk Feedback (CSV)"])

if app_mode == "Single Feedback":
    st.header("Analyze Single Feedback")
    text = st.text_area("Enter customer feedback:")
    if text:
        # Clean Text
        cleaned_text = cleantext.clean(text, clean_all=False, extra_spaces=True, stopwords=True, lowercase=True, numbers=True, punct=True)

        # Analyze with TextBlob
        blob = TextBlob(cleaned_text)
        polarity = round(blob.sentiment.polarity, 2)
        subjectivity = round(blob.sentiment.subjectivity, 2)

        # Analyze with VADER
        vader_result = sia.polarity_scores(cleaned_text)

        # Analyze with RoBERTa
        roberta_result = polarity_scores_roberta(cleaned_text)

        # Final Sentiment Classification
        sentiment = classify_sentiment(vader_result, roberta_result)

        # Display Results
        st.subheader("Results")
        st.write("**Cleaned Text:**", cleaned_text)
        st.write("**TextBlob Polarity:**", polarity)
        st.write("**TextBlob Subjectivity:**", subjectivity)
        st.write("**VADER Sentiment:**", vader_result)
        st.write("**RoBERTa Sentiment:**", roberta_result)
        st.write("**Overall Sentiment:**", sentiment)

elif app_mode == "Bulk Feedback (CSV)":
    st.header("Analyze Bulk Feedback from CSV")
    uploaded_file = st.file_uploader("Upload a CSV file with a 'Text' column", type=["csv"])

    if uploaded_file:
        try:
            # Load the dataset with a fallback encoding
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')  # Default encoding
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding='latin-1')  # Fallback encoding

            if 'Text' not in df.columns:
                st.error("The uploaded CSV must contain a 'Text' column.")
            else:
                # Process each feedback
                results = []
                for text in df['Text']:
                    cleaned_text = cleantext.clean(text, clean_all=False, extra_spaces=True, stopwords=True, lowercase=True, numbers=True, punct=True)
                    vader_result = sia.polarity_scores(cleaned_text)
                    roberta_result = polarity_scores_roberta(cleaned_text)
                    sentiment = classify_sentiment(vader_result, roberta_result)
                    results.append({'Text': text, 'Sentiment': sentiment})

                # Convert results to DataFrame
                results_df = pd.DataFrame(results)

                # Display and Download Results
                st.subheader("Results")
                st.write(results_df)

                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results", data=csv, file_name="sentiment_results.csv", mime="text/csv")

                # Sentiment Distribution Visualization
                st.subheader("Sentiment Distribution")
                sentiment_counts = results_df['Sentiment'].value_counts()

                # Ensure all sentiment categories are represented
                sentiment_categories = ['Positive', 'Negative', 'Neutral']
                sentiment_counts = sentiment_counts.reindex(sentiment_categories, fill_value=0)

                st.bar_chart(sentiment_counts)

        except Exception as e:
            st.error(f"Error processing the file: {e}")
