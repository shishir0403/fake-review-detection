import streamlit as st
import pickle
import re

# ======================
# Load model and vectorizer
# ======================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ======================
# Preprocessing (SAME as training)
# ======================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# ======================
# UI
# ======================
st.set_page_config(page_title="Fake Review Detector", page_icon="🕵️")

st.title("🕵️ Fake Review Detection")
st.write("Detect whether a review is Computer-Generated (Fake) or Human-Written (Real)")

user_input = st.text_area("Enter Review:")

# ======================
# Prediction
# ======================
if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter a review")
    
    else:
        # Preprocess
        cleaned = clean_text(user_input)

        # Vectorize
        vect = vectorizer.transform([cleaned])

        # Predict
        prediction = model.predict(vect)[0]

        # Output
        if prediction == 1:
            st.success("✅ Human-Written Review (Real)")
        else:
            st.error("❌ Computer-Generated Review (Fake)")

        # Optional debug
        st.write("Prediction Value:", prediction)