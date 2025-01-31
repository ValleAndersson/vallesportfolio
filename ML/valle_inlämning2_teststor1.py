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
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
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

# Define classifiers for comparison
classifiers = {
    "Logistic Regression": OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight=None, C=10)),
    "SVC": OneVsRestClassifier(SVC(kernel='linear', probability=True, class_weight='balanced', C=10, gamma=0.001)),
    "Random Forest": OneVsRestClassifier(RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=5, min_samples_leaf=1)),
    "Multinomial Naive Bayes": OneVsRestClassifier(MultinomialNB(alpha=0.1, fit_prior=True))
}

for name, clf in classifiers.items():
    start_time = time.time()
    vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,4) if name in ['Logistic Regression', 'SVC'] else (1,2) if name == 'Random Forest' else (1,3), norm='l2', max_df=0.9, min_df=(4 if name == 'Logistic Regression' else 3 if name == 'SVC' else 3 if name == 'Random Forest' else 5))
    vectorizer.fit(train_text)
    x_train, x_test = vectorizer.transform(train_text), vectorizer.transform(test_text)
    print(f"Training {name}...")
    clf.fit(x_train, y_train)
    y_pred = (clf.predict_proba(x_test) >= (0.38 if name == 'Logistic Regression' else 0.42 if name == 'SVC' else 0.3 if name == 'Random Forest' else 0.42)).astype(int)
    print(f"Accuracy for {name}: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Time taken for {name}: {time.time() - start_time:.2f} seconds")
    print(f"Classification Report ({name}):\n{classification_report(y_test, y_pred, target_names=categories, zero_division=0)}\n")

# Hyperparameter tuning with RandomizedSearchCV
param_distributions = {
    "Logistic Regression": {
        "estimator__C": [0.1, 1, 10],
        "estimator__penalty": ["l1", "l2"]
    },
    "SVC": {
        "estimator__C": [0.1, 1, 10, 100],
        "estimator__gamma": [0.001, 0.01, 0.1],
        "estimator__kernel": ["linear", "rbf"]
    },
    "Random Forest": {
        "estimator__n_estimators": [50, 100, 200],
        "estimator__max_depth": [None, 10, 20],
        "estimator__min_samples_split": [5, 10]
    },
    "Multinomial Naive Bayes": {
        "estimator__alpha": [0.1, 0.5, 1.0],
        "estimator__fit_prior": [True, False]
    }
}

for name, clf in classifiers.items():
    print(f"Tuning hyperparameters for {name} with RandomizedSearchCV...")
    start_time = time.time()
    search = RandomizedSearchCV(clf, param_distributions[name], n_iter=1000 if name in ['Logistic Regression', 'SVC'] else 100, cv=5, scoring="f1_micro", n_jobs=-1)
    search.fit(x_train, y_train)
    print(f"Best parameters for {name}: {search.best_params_}")
    print(f"Best cross-validated score for {name}: {search.best_score_:.2f}")
    print(f"Time taken for tuning {name}: {time.time() - start_time:.2f} seconds")
