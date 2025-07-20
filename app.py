import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
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

# Load vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

# Load all models
models = {
    "Logistic Regression": pickle.load(open("logistic_model.pkl", 'rb')),
    "Support Vector Classifier": pickle.load(open("svm_model.pkl", 'rb')),
    "K-Nearest Neighbors": pickle.load(open("knn_model.pkl", 'rb'))
}

# Streamlit UI
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

# Dropdown to select model
model_choice = st.selectbox("Choose a model", list(models.keys()))

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message.")
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
            st.header("Spam")
        else:
            st.header("Not Spam")
