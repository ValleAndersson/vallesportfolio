import re
import time
import sys
import warnings
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV

# Suppress warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Load data
file_path = r"C:\workspace\vallesportfolio\ML\Book1tillML.csv"
data_raw = pd.read_csv(file_path)

data_raw = data_raw.sample(frac=1)  # Shuffle data

# Identify category columns
categories = list(data_raw.columns.values)[2:]

# Text cleaning
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('swedish'))
stemmer = SnowballStemmer("swedish")

def clean_text(sentence):
    words = [word for word in nltk.word_tokenize(sentence) if word not in stop_words]
    stemmed_words = [stemmer.stem(word) for word in words]
    return " ".join(stemmed_words)

data_raw['Heading'] = (
    data_raw['Heading']
    .str.lower()
    .str.replace(r'[^\w\s]', '', regex=True)
    .str.replace(r'\d+', '', regex=True)
    .str.replace(r'<.*?>', '', regex=True)
    .apply(clean_text)
)

# Train/Test split
train, test = train_test_split(data_raw, random_state=42, test_size=0.30, shuffle=True)
train_text, test_text = train['Heading'], test['Heading']
y_train, y_test = train.drop(['Id', 'Heading'], axis=1), test.drop(['Id', 'Heading'], axis=1)

# Define classifier
classifier = OneVsRestClassifier(MultinomialNB(alpha=0.1, fit_prior=True))

# Vectorization
vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2', max_df=0.9, min_df=5)
vectorizer.fit(train_text)
x_train, x_test = vectorizer.transform(train_text), vectorizer.transform(test_text)

# Training
start_time = time.time()
print("Training Multinomial Naive Bayes...")
classifier.fit(x_train, y_train)
y_pred = (classifier.predict_proba(x_test) >= 0.35).astype(int)

# Evaluation
print(f"Accuracy for Multinomial Naive Bayes: {accuracy_score(y_test, y_pred):.2f}")
print(f"Time taken: {time.time() - start_time:.2f} seconds")
print(f"Classification Report:\n{classification_report(y_test, y_pred, target_names=categories, zero_division=0)}\n")

# Hyperparameter tuning with RandomizedSearchCV
param_distributions = {
    "estimator__alpha": [0.1, 0.5, 1.0],
    "estimator__fit_prior": [True, False]
}

print("Tuning hyperparameters for Multinomial Naive Bayes with RandomizedSearchCV...")
start_time = time.time()
search = RandomizedSearchCV(classifier, param_distributions, n_iter=100, cv=5, scoring="f1_micro", n_jobs=-1)
search.fit(x_train, y_train)
print(f"Best parameters: {search.best_params_}")
print(f"Best cross-validated score: {search.best_score_:.2f}")
print(f"Time taken for tuning: {time.time() - start_time:.2f} seconds")
