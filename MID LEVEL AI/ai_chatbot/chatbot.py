import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load intents
with open("intents.json") as file:
    data = json.load(file)

patterns = []
tags = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])

# TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

print("Chatbot Ready (type quit to exit)\n")

while True:
    user = input("You: ")

    if user == "quit":
        break

    user_vec = vectorizer.transform([user])
    similarity = cosine_similarity(user_vec, X)

    index = similarity.argmax()
    tag = tags[index]

    for intent in data["intents"]:
        if intent["tag"] == tag:
            print("Bot:", random.choice(intent["responses"]))