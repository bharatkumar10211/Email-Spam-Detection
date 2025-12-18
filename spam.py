import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv("archive/spam.csv", encoding="latin-1")
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(
    data['message'],
    data['label'],
    test_size=0.2,
    random_state=42
)

v = CountVectorizer()
X_train_vect = v.fit_transform(X_train)
X_test_vect = v.transform(X_test)

m = MultinomialNB()
m.fit(X_train_vect, y_train)

y_pred = m.predict(X_test_vect)
print("Accuracy:", accuracy_score(y_test, y_pred))

def predict_message(msg):
    vec = v.transform([msg])
    probs = m.predict_proba(vec)[0]

    ham_prob = round(probs[0] * 100, 2)
    spam_prob = round(probs[1] * 100, 2)

    label = "SPAM" if spam_prob > ham_prob else "NOT SPAM"
    return label, spam_prob, ham_prob
