import os
import json
import datetime
import csv
import random
import nltk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('punkt')

# Load intents
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Define chatbot functions
def get_restaurant_details(name):
    restaurant_details = {
        "Sri Venkateswara Annapoorna Restaurant": {
            "location": "Guntur",
            "speciality": "Andhra-style meals",
            "rating": "4.5",
            "contact": "123-456-7890",
            "hours": "9 AM - 10 PM",
        },
        "Boardwalk Restaurant": {
            "location": "Visakhapatnam",
            "speciality": "Seafood specialties",
            "rating": "4.8",
            "contact": "987-654-3210",
            "hours": "11 AM - 11 PM",
        },
    }
    return restaurant_details.get(name, "Sorry, I don't have information about this restaurant.")

def chatbot(input_text):
    input_text_transformed = vectorizer.transform([input_text])
    tag = clf.predict(input_text_transformed)[0]
    if tag == "restaurant_details":
        restaurant_name = input_text.split("about ")[-1].strip("?")
        details = get_restaurant_details(restaurant_name)
        return details if isinstance(details, str) else (
            f"{restaurant_name} Details:\n"
            f"Location: {details['location']}\n"
            f"Speciality: {details['speciality']}\n"
            f"Rating: {details['rating']}\n"
            f"Contact: {details['contact']}\n"
            f"Hours: {details['hours']}"
        )
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# Main Streamlit App
def main():
    st.title("Chatbot for Andhra Pradesh Restaurants")
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome! Start a conversation to find restaurants or get details.")
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        user_input = st.text_input("You:")
        if user_input:
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=150)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

    elif choice == "Conversation History":
        st.header("Conversation History")
        if os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
        else:
            st.write("No conversation history found.")

    elif choice == "About":
        st.write("This chatbot provides restaurant recommendations and details in Andhra Pradesh.")

if __name__ == '__main__':
    main()
