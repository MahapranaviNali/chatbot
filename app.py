import os
import json
import datetime
import csv
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk

nltk.download("punkt")

# Load intents file with error handling
def load_intents(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        st.error("Error: 'intents.json' file not found.")
        return []
    except json.JSONDecodeError as e:
        st.error(f"Error decoding 'intents.json': {e}")
        return []

# Load intents
file_path = "./intents.json"
intents = load_intents(file_path)

# Machine Learning Model
def train_model(intents):
    vectorizer = TfidfVectorizer(ngram_range=(1, 4))
    clf = LogisticRegression(random_state=0, max_iter=10000)
    tags, patterns = [], []

    for intent in intents:
        for pattern in intent["patterns"]:
            tags.append(intent["tag"])
            patterns.append(pattern)

    if patterns:
        x = vectorizer.fit_transform(patterns)
        y = tags
        clf.fit(x, y)
    return vectorizer, clf

vectorizer, clf = train_model(intents)

# Chatbot function
def chatbot(input_text):
    try:
        input_text_transformed = vectorizer.transform([input_text])
        tag = clf.predict(input_text_transformed)[0]
        for intent in intents:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
        return "Sorry, I don't understand that."
    except Exception as e:
        return f"Error processing your request: {e}"

# Streamlit App
def main():
    st.title("Chatbot for Andhra Pradesh Restaurants")
    st.sidebar.title("Menu")
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Options", menu)

    # Conversation logging
    log_file = "chat_log.csv"
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["User Input", "Chatbot Response", "Timestamp"])

    if choice == "Home":
        st.header("Restaurant Finder Chatbot")
        user_input = st.text_input("You:", placeholder="Type your message here...")
        if user_input:
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=150)
            # Log conversation
            with open(log_file, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([user_input, response, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

    elif choice == "Conversation History":
        st.header("Conversation History")
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    st.write(f"**User:** {row[0]}")
                    st.write(f"**Chatbot:** {row[1]}")
                    st.write(f"**Timestamp:** {row[2]}")
        else:
            st.write("No conversation history available.")

    elif choice == "About":
        st.write("This chatbot helps you find restaurants in Andhra Pradesh.")
        st.write("Designed using Streamlit and machine learning.")

if __name__ == "__main__":
    main()
