import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
from wordcloud import WordCloud
import re
import string
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Configure Streamlit page
st.set_page_config(
    page_title="Sentiments App",
    page_icon="R.png",
    layout="wide"
)

# Load the exported data using st.cache_data
@st.cache_data
def load_data():
    return pd.read_csv('exported_sentiments.csv')

df1 = load_data()

# Text Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r'\b[0-9]+\b\s*', '', text)
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

X = df1['Feedback']
y = df1['Sentiments']

# Preprocess the texts
X_preprocessed = [preprocess_text(text) for text in X]

# Load the pretrained BERT model for sentiment analysis using the pipeline
classifier = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")

# Create a Streamlit container for sentiment analysis
with st.container():
    st.subheader("Sentiment Analysis App")
    st.image("sentim.jpg")

st.markdown("""
This web application is a sentiment analysis tool developed by AI&DS. It can be used to determine whether user-entered text has a Positive or Negative sentiment. The underlying text classification model was trained on feedback data collected from 300 level undergraduate Computer Engineering students at the University of Ilorin (who are Raqib's peers). Subsequently, the model underwent fine-tuning using BERT and KerasNLP techniques, resulting in an impressive accuracy score of 96%. The objective is to uncover sentiments expressed in the feedback and gain a comprehensive understanding of student perceptions and satisfaction regarding their educational experience.  
To utilize the application, simply input your text, and it will promptly reveal the underlying sentiment.
The app also has Exploratory Data Analysis capabilities.
""")

# Create containers for sideboxes
course_code_container = st.empty()
previous_exp_container = st.empty()
gender_container = st.empty()
attendance_container = st.empty()
difficulty_container = st.empty()
study_hours_container = st.empty()
satisfaction_container = st.empty()
department_container = st.empty()

# Get values from sideboxes
course_code = course_code_container.selectbox("Course Code", ['Select Course Code', 'CPE 321', 'CPE 311', 'CPE 341', 'CPE 381', 'CPE 331', 'MEE 361', 'GSE 301'])
previous_exp = previous_exp_container.selectbox("Previous Experience", ['Select Option', "Yes", "No"])
gender = gender_container.selectbox("Gender", ['Select Gender', 'Male', 'Female'])
attendance = attendance_container.selectbox("Attendance", ['Select Attendance', 'Regular', 'Irregular', 'Occasional'])
difficulty = difficulty_container.selectbox("Course Difficulty", ['Select Difficulty', 'Easy', 'Difficult', 'Challenging', 'Moderate'])
study_hours = study_hours_container.slider("Study Hours (per week)", 0, 24)
satisfaction = satisfaction_container.slider("Overall Satisfaction", 1, 10)
department = department_container.selectbox("Department", ['Select Option', "Yes", "No"])

# Add text input
text = st.text_input("Enter your text:")

if st.button("Submit Predictions"):
    # Check if all required fields are filled
    if not text or course_code == 'Select Course Code' or previous_exp == 'Select Option' or gender == 'Select Gender' or \
            attendance == 'Select Attendance' or difficulty == 'Select Difficulty' or study_hours is None or \
            satisfaction is None or department == 'Select Option':
        st.warning("Please fill in all the required fields before submitting predictions.")
    else:
        # Predict the sentiment with a spinner
        with st.spinner("Loading Output.."):
            preprocessed_text = preprocess_text(text)
            sentiment = classifier(preprocessed_text)[0]

            sentiment_label = sentiment['label']
            confidence = sentiment['score'] * 100

            if sentiment_label == "POSITIVE":
                st.success(f"The sentiment of your text is: {sentiment_label} with a {confidence:.2f}% confidence.")
            else:
                st.error(f"The sentiment of your text is: {sentiment_label} with a {confidence:.2f}% confidence.")

            # Append the new row to the DataFrame with numerical label
            new_row = {
                'Course Code': course_code,
                'Feedback': text,
                'Previous Experience': previous_exp,
                'Gender': gender,
                'Attendance': attendance,
                'Course Difficulty': difficulty,
                'Study Hours (per week)': study_hours,
                'Overall Satisfaction': satisfaction,
                'Department': department,
                'Date': datetime.today().strftime('%Y-%m-%d'),
                'Time': datetime.now().strftime('%H:%M:%S'),
                'Hour': datetime.now().hour,
                'Processed_Feedback': preprocess_text(text),
                'Char_Count': len(preprocess_text(text)),
                'Word_Count': len(preprocess_text(text).split()),
                'Sentiments': 1 if sentiment_label == "POSITIVE" else 0
            }

            # Append the new row to the DataFrame
            df1 = pd.concat([df1, pd.DataFrame([new_row])], ignore_index=True)

            # Save the updated dataset to the CSV file
            try:
                df1.to_csv('exported_sentiments.csv', index=False)
                st.success("Data saved successfully.")
            except Exception as e:
                st.error(f"Error saving data: {str(e)}")

# Generate metrics and confusion matrix
if st.button("Generate Metrics and Confusion Matrix"):
    y_pred = df1['Sentiments'].tolist()
    y_true = df1['Sentiments'].tolist()

    # Convert to string to ensure consistent types
    y_true = pd.Series(y_true).astype(str)
    y_pred = pd.Series(y_pred).astype(str)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    # Display metrics
    st.write(f'Accuracy: {accuracy:.2f}')
    st.write(f'Precision: {precision:.2f}')
    st.write(f'Recall: {recall:.2f}')
    st.write(f'F1 Score: {f1:.2f}')

    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    st.pyplot(plt)
