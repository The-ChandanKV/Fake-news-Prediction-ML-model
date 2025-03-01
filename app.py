from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('stopwords')

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model/fake_news_model.pkl', 'rb'))
vectorizer = pickle.load(open('model/tfidf_vectorizer.pkl', 'rb'))

@app.route("/")
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        news_text = request.form["news_text"]
        transformed_text = vectorizer.transform([news_text])
        prediction = model.predict(transformed_text)[0]

        # Convert 0/1 prediction to meaningful text
        result_text = "Real News" if prediction == 1 else "Fake News"

        return render_template("index.html", prediction=result_text)



# Text preprocessing
def preprocess_text(content):
    port_stem = PorterStemmer()
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()
    content = [port_stem.stem(word) for word in content if not word in stopwords.words('english')]
    return ' '.join(content)




if __name__ == '__main__':
    app.run(debug=True)