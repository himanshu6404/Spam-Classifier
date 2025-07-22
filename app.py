import streamlit as st
import nltk
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure necessary NLTK data is downloaded
nltk_dependencies = ['punkt', 'stopwords']
for dep in nltk_dependencies:
    try:
        nltk.data.find(f'tokenizers/{dep}' if dep == 'punkt' else f'corpora/{dep}')
    except LookupError:
        nltk.download(dep)

# Initialize stemmer and stopwords
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Text preprocessing function
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
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Set page config
st.set_page_config(page_title="Spam Classifier", page_icon="📩")

# Title and description
st.title("📩 Spam Message Classifier")
st.write("Classify SMS messages as **Spam** or **Ham (Not Spam)** using different ML models.")

# Load vectorizer and models
@st.cache_resource
def load_assets():
    with open("vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("logistic_model.pkl", "rb") as f:
        lrc_model = pickle.load(f)
    with open("svm_model.pkl", "rb") as f:
        svc_model = pickle.load(f)
    with open("knn_model.pkl", "rb") as f:
        knn_model = pickle.load(f)
    
    models = {
        "Logistic Regression": lrc_model,
        "Support Vector Classifier": svc_model,
        "K-Nearest Neighbors": knn_model
    }
    return tfidf, models

# Load all models and vectorizer
tfidf, models = load_assets()

# User input
user_input = st.text_area("✍️ Enter your message below:", height=150)

# Model selection
selected_model_name = st.selectbox("🔍 Choose a model to use:", list(models.keys()))
selected_model = models[selected_model_name]

# Predict button
if st.button("🚀 Classify Message"):
    if not user_input.strip():
        st.warning("Please enter a message before classifying.")
    else:
        # Preprocess input
        transformed_input = transform_text(user_input)

        # Vectorize input
        vector_input = tfidf.transform([transformed_input])

        # Convert to dense if using SVM
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
st.markdown("Made with ❤️ using Streamlit")
