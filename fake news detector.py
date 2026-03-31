import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample dataset
data = pd.DataFrame({
    "text": [
        "Government launches new scheme",
        "Aliens landed in India yesterday",
        "Stock market hits record high",
        "Fake miracle cure discovered"
    ],
    "label": [1, 0, 1, 0]
})

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["text"])
y = data["label"]

model = LogisticRegression()
model.fit(X, y)

def predict_news(news):
    vec = vectorizer.transform([news])
    pred = model.predict(vec)[0]
    return "Real News" if pred == 1 else "Fake News"

print(predict_news("New government policy announced"))