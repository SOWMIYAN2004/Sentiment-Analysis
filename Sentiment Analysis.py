import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Example dataset (text, sentiment label)
data = [
    ("I love this product!", "positive"),
    ("This is the worst experience ever.", "negative"),
    ("It's okay, nothing special.", "neutral"),
    ("Absolutely fantastic service!", "positive"),
    ("I wouldn't recommend it to anyone.", "negative")
]

# Split data into texts and labels
texts, labels = zip(*data)

# Create a pipeline for vectorization and classification
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict sentiment on the test data
predicted_labels = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predicted_labels)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Example prediction
new_text = ["I had an amazing day!"]
predicted_sentiment = model.predict(new_text)
print(f"Predicted Sentiment: {predicted_sentiment[0]}")
