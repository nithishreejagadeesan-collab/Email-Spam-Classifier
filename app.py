import streamlit as st
import joblib

# Load Model
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Page Title
st.set_page_config(page_title="Email Spam Classifier", page_icon="📧")

st.title("📧 Email Spam Classifier")
st.write("Enter an Email or SMS message to check whether it is Spam or Ham.")

# Input Box
message = st.text_area("Enter Message")

# Prediction Button
if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        transformed_message = vectorizer.transform([message])
        prediction = model.predict(transformed_message)

        if prediction[0] == 1:
            st.error("🚫 Spam Message")
        else:
            st.success("✅ Ham (Not Spam)")