from flask import Flask, render_template, request
from transformers import pipeline
from pymongo import MongoClient
from datetime import datetime
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os

app = Flask(__name__)

print("Loading AI model...")

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

print("AI Model Loaded Successfully!")

client = MongoClient("mongodb://localhost:27017/")

db = client["sentimentDB"]

collection = db["sentimentResults"]

def create_chart():

    data = list(collection.find())

    if len(data) == 0:
        return

    labels = []

    for item in data:
        labels.append(item["label"])

    positive_count = labels.count("POSITIVE")

    negative_count = labels.count("NEGATIVE")

    plt.figure(figsize=(4,4))

    plt.pie(
        [positive_count, negative_count],
        labels=["POSITIVE", "NEGATIVE"],
        autopct="%1.1f%%"
    )

    plt.title("Sentiment Report")

    plt.savefig("static/chart.png")

    plt.close()

@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        user_text = request.form["text"]

        bad_words = [
            "fuck",
            "hate",
            "idiot",
            "stupid",
            "worst",
            "bad"
        ]

        if any(word in user_text.lower() for word in bad_words):

            label = "NEGATIVE"

            score = 99.0

        else:

            prediction = sentiment_pipeline(user_text)[0]

            label = prediction["label"]

            score = round(prediction["score"] * 100, 2)

        result = {
            "text": user_text,
            "label": label,
            "score": score,
            "time": datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        }

        collection.insert_one(result)

        create_chart()

    history = list(
        collection.find().sort("_id", -1).limit(5)
    )

    return render_template(
        "index.html",
        result=result,
        history=history
    )

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
