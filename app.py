
from flask import Flask, render_template, request
import joblib
import re
import string

app = Flask(__name__)

# Load saved model and vectorizer
model = joblib.load("job_fraud_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub("\d+", "", text)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    title = request.form['title']
    company = request.form['company_profile']
    description = request.form['description']
    location = request.form['location']
    
    text_input = f"{title} {company} {description} {location}"
    text_input = preprocess_text(text_input)
    text_vectorized = vectorizer.transform([text_input])
    
    prediction = model.predict(text_vectorized)[0]
    result = "Fake Job Posting" if prediction == 1 else "Real Job Posting"

    return render_template('result.html', title=title, company=company, description=description, location=location, result=result)

if __name__ == '__main__':
    app.run(debug=True)
