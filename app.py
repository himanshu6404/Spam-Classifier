import streamlit as st
import pickle
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Safe download check (won't re-download every time)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

ps = PorterStemmer()

# Preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load vectorizer and models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

models = {
    "Logistic Regression": pickle.load(open("logistic_model.pkl", 'rb')),
    "Support Vector Classifier": pickle.load(open("svm_model.pkl", 'rb')),
    "K-Nearest Neighbors": pickle.load(open("knn_model.pkl", 'rb'))
}

# Streamlit UI
st.title("📩 Email/SMS Spam Classifier")
st.markdown("A simple NLP-based spam detector using Logistic Regression, SVM, or KNN.")

input_sms = st.text_area("Enter your message")

model_choice = st.selectbox("Choose a model", list(models.keys()))

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict
        model = models[model_choice]
        result = model.predict(vector_input)[0]

        # 4. Display
        if result == 1:
            st.error("❌ Spam")
        else:
            st.success("✅ Not Spam")
