import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

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

# Set page config
st.set_page_config(page_title="Spam Classifier", page_icon="📩")

# Title
st.title("📩 Spam Message Classifier")
st.write("Classify SMS messages as **Spam** or **Ham (Not Spam)** using different ML models.")

# Load vectorizer and models
@st.cache_resource
def load_assets():
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    lrc_model = pickle.load(open('logistic_model.pkl', 'rb'))
    svc_model = pickle.load(open('svm_model.pkl', 'rb'))
    knn_model = pickle.load(open('knn_model.pkl', 'rb'))
    return tfidf, {
        "Logistic Regression": lrc_model,
        "Support Vector Classifier": svc_model,
        "K-Nearest Neighbors": knn_model
    }

# Load all assets
tfidf, models = load_assets()

# Input text box
user_input = st.text_area("✍️ Enter your message below:", height=150)

# Model selector
selected_model_name = st.selectbox("🔍 Choose a model to use:", list(models.keys()))
selected_model = models[selected_model_name]

# Predict button
if st.button("🚀 Classify Message"):
    if not user_input.strip():
        st.warning("Please enter a message before classifying.")
    else:
        # Transform the input text
        transformed_input = transform_text(user_input)

        # Vectorize the input text
        vector_input = tfidf.transform([transformed_input])

        # SVC requires dense array
        if selected_model_name == "Support Vector Classifier":
            vector_input = vector_input.toarray()


        # Make prediction
        prediction = selected_model.predict(vector_input)[0]

        # Display result
        if prediction == 1:
            st.error("❌ This message is classified as **SPAM**.")
        else:
            st.success("✅ This message is classified as **HAM** (Not Spam).")

        # Show model used
        st.markdown(f"🔧 Model used: **{selected_model_name}**")


# Footer
st.markdown("---")