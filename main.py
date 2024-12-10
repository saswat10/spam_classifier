import streamlit as st
import pickle
from util import transform_text


tfdif = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")


if st.button("Predict"):
  # Preprocess
  transformed_sms = transform_text(input_sms)

  # Vectorize
  vector_input = tfdif.transform([transformed_sms])

  # Predict
  result = model.predict(vector_input)[0]

  # Display
  if result == 1:
    st.header("Spam")
  else: 
    st.header("Not Spam")
