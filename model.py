import pandas as pd
import numpy as np
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("fake_job_postings.csv")

# Fill missing values
df.fillna("", inplace=True)

# Feature Engineering
df['text'] = df['title'] + " " + df['company_profile'] + " " + df['description'] + " " + df['location'] 

def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub("\d+", "", text)
    return text

df['cleaned_text'] = df['text'].apply(preprocess_text)

# Label Encoding
df['fraudulent'] = df['fraudulent'].astype(int)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['fraudulent'], test_size=0.2, random_state=42)

# Vectorization using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, y_train)

# Model Evaluation
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(model, "job_fraud_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
