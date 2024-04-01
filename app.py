# pip install -U streamlit
# streamlit run app.py
    
import streamlit as st
import pandas as pd
from textblob import TextBlob
import pickle
import time
import joblib


# Function to perform sentiment analysis
def analyze_sentiment(tweet):
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    elif  analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Irrelevent'
# Load your model
def load_model():
    # Load your trained model here (e.g., from a .pkl file)
    model = joblib.load('sentiment_ana.pkl')
    return model

# Load your dataset
@st.cache_data 
def load_data():
    # Load your dataset here (e.g., from a CSV file)
    df = pd.read_csv('predicted_results.csv')
    return df

# Streamlit app
def main():
    st.title('Twitter Sentiment Analysis')
    
    # Load the trained model
    model = load_model()

    # Load the dataset
    df = load_data()

    # Select a tweet from the dataset
    selected_index = st.selectbox('Select a tweet index:', df.index)
    selected_tweet = df.loc[selected_index, 'text']

    # Display the selected tweet
    st.subheader('Selected Tweet:')
    st.write(selected_tweet)

    # Perform sentiment analysis
    sentiment = analyze_sentiment(selected_tweet)
    #sentiment=model.predict([selected_tweet])
    
    st.subheader('Sentiment Analysis(True Lable):')
    st.write(sentiment)

# Run the app
if __name__ == '__main__':
    main()

# load the model
model = pickle.load(open('sentiment_ana.pkl', 'rb'))

tweet = st.text_input('Enter your Tweet')

submit = st.button('Analyze Sentiment')

if submit:
    start = time.time()
    prediction = model.predict([tweet])
    end = time.time()
    st.write('Prediction time taken: ', round(end-start, 2), 'seconds')
    
    print(prediction[0])
    st.write(prediction[0])
    


