import re
import sys
import warnings
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

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

def removeStopWords(sentence):
    return " ".join(
        [word for word in nltk.word_tokenize(sentence) 
         if word not in stop_words]
    )

data_raw['Heading'] = (
    data_raw['Heading']
    .str.lower()
    .str.replace(r'[^\w\s]', '', regex=True)
    .str.replace(r'\d+', '', regex=True)
    .str.replace(r'<.*?>', '', regex=True)
    .apply(removeStopWords)
)

# Train/Test split
train, test = train_test_split(data_raw, random_state=42, test_size=0.30, shuffle=True)
train_text, test_text = train['Heading'], test['Heading']
y_train, y_test = train.drop(['Id', 'Heading'], axis=1), test.drop(['Id', 'Heading'], axis=1)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
vectorizer.fit(train_text)
x_train, x_test = vectorizer.transform(train_text), vectorizer.transform(test_text)

# Define classifier with multi-label support
clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
clf.fit(x_train, y_train)

# Make predictions
y_pred = clf.predict(x_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Hyperparameter tuning
param_grid = {
    "estimator__C": [0.01, 0.1, 1, 10, 100],
    "estimator__penalty": ["l1", "l2"]
}

grid = GridSearchCV(OneVsRestClassifier(LogisticRegression(solver="liblinear")), param_grid, cv=5, scoring="f1_micro")
grid.fit(x_train, y_train)

print("Best Params:", grid.best_params_)
print("Best Score:", grid.best_score_)

# Retrain with best parameters
best_clf = grid.best_estimator_
best_clf.fit(x_train, y_train)