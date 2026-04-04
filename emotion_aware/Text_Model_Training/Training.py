import pandas as pd
import numpy as np
import pickle
import re
import os
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

df = pd.read_csv("training.csv")
df.columns = df.columns.str.strip()

print("Dataset shape:", df.shape)
print("\nClass Distribution:")
print(df["label"].value_counts())

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess(text):
    words = text.split()
    words = [
        lemmatizer.lemmatize(w)
        for w in words
        if w not in stop_words and len(w) > 2
    ]
    return " ".join(words)

df['cleaned'] = df['text'].apply(clean_text)
df['processed'] = df['cleaned'].apply(preprocess)

vectorizer = TfidfVectorizer(
    max_features=12000,
    ngram_range=(1,2),
    min_df=2,
    sublinear_tf=True
)

X = vectorizer.fit_transform(df['processed'])
y = df['label'].astype(int)

print("Feature shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LogisticRegression(
    max_iter=4000,
    class_weight='balanced',
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

label_names = ['Sad', 'Happy', 'Love', 'Angry', 'Fear', 'Surprise']

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_names,
            yticklabels=label_names)
plt.title('Confusion Matrix - Text Emotion Detection Model', fontsize=14)
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig('text_confusion_matrix.png', dpi=150)
plt.show()

os.makedirs("text_model", exist_ok=True)

with open("text_model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("text_model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model saved successfully!")

label_map = {
    0: "Sad",
    1: "Happy",
    2: "Love",
    3: "Angry",
    4: "Fear",
    5: "Surprise"
}

def predict_emotion(text):
    cleaned = clean_text(text)
    processed = preprocess(cleaned)

    vector = vectorizer.transform([processed])
    probs = model.predict_proba(vector)[0]

    pred_class = np.argmax(probs)
    confidence = float(np.max(probs))

    print("Input Text     :", text)
    print("Predicted      :", label_map[pred_class])
    print("Confidence     :", round(confidence * 100, 2), "%")
    print("\nAll Probabilities:")
    for i, prob in enumerate(probs):
        print(f"  {label_map[i]}: {round(prob * 100, 2)}%")

    return label_map[pred_class], confidence


predict_emotion("I am so happy today!")
predict_emotion("I feel really sad and lonely.")
predict_emotion("This makes me so angry!")
