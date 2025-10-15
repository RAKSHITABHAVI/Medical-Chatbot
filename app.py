import streamlit as st
import pandas as pd
from fuzzywuzzy import process
import datetime


faq_df = pd.read_csv("medical_faq.csv")


def ask_medical_bot(question):
    questions = faq_df['question'].tolist()
    match, score = process.extractOne(question, questions)
    if score > 50:
        return faq_df[faq_df['question'] == match]['answer'].values[0]
    else:
        return "Sorry, I don't know the answer. Please consult a doctor for professional advice."

def symptom_checker(symptoms):
    symptoms = symptoms.lower()
    if "fever" in symptoms and "cough" in symptoms:
        return "You might have a cold or flu. Rest and stay hydrated."
    elif "headache" in symptoms and "dizziness" in symptoms:
        return "Could be dehydration or stress. Drink water and rest."
    else:
        return "Please provide more details or consult a doctor."

def save_chat(user_input, bot_response):
    df = pd.DataFrame([[datetime.datetime.now(), user_input, bot_response]],
                      columns=["time","question","answer"])
    df.to_csv("chat_history.csv", mode='a', index=False, header=False)


st.title(" Medical Chatbot")
st.write("Disclaimer: This chatbot provides general information only and is not a substitute for professional medical advice.")

user_input = st.text_input("Ask your question or describe your symptoms:")

if st.button("Send"):
    # You can switch between FAQ bot or symptom checker
    bot_response = ask_medical_bot(user_input)
    # bot_response = symptom_checker(user_input)  # Optional: use symptom checker
    st.write("🩺 Bot:", bot_response)
    save_chat(user_input, bot_response)


try:
    df_history = pd.read_csv("chat_history.csv", names=["time","question","answer"])
    st.write("Total interactions:", df_history.shape[0])
    st.bar_chart(df_history['question'].value_counts().head(5))
except FileNotFoundError:
    st.write("No interactions yet.")
