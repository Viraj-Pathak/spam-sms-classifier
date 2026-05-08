import os
import urllib.request
import zipfile
import re

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "SMSSpamCollection")
DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/00228/smsspamcollection.zip"
)

_stemmer = PorterStemmer()
_stop_words = set(stopwords.words("english"))


def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DATA_FILE):
        print("Dataset already present.")
        return
    print("Downloading SMS Spam Collection dataset...")
    zip_path = os.path.join(DATA_DIR, "smsspamcollection.zip")
    urllib.request.urlretrieve(DATASET_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract("SMSSpamCollection", DATA_DIR)
    os.remove(zip_path)
    print("Download complete.")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_FILE,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="latin-1",
    )
    return df


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [_stemmer.stem(t) for t in tokens if t not in _stop_words]
    return " ".join(tokens)


def plot_confusion_matrix(cm: np.ndarray, path: str = "confusion_matrix.png"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["ham", "spam"],
        yticklabels=["ham", "spam"],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def plot_precision_recall(y_test, y_scores, path: str = "precision_recall_curve.png"):
    prec, rec, _ = precision_recall_curve(y_test, y_scores)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, color="steelblue", lw=2, label="Naive Bayes")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def main():
    # ── Data ─────────────────────────────────────────────────────────────────
    download_dataset()
    df = load_data()

    print(f"\nDataset shape : {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}")

    # ── Preprocessing ────────────────────────────────────────────────────────
    print("\nPreprocessing messages…")
    df["cleaned"] = df["message"].apply(preprocess)
    df["label_enc"] = (df["label"] == "spam").astype(int)

    # ── Split ────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned"],
        df["label_enc"],
        test_size=0.2,
        random_state=42,
        stratify=df["label_enc"],
    )

    # ── TF-IDF ───────────────────────────────────────────────────────────────
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # ── Train ────────────────────────────────────────────────────────────────
    print("Training Multinomial Naive Bayes…")
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test_tfidf)
    y_scores = model.predict_proba(X_test_tfidf)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    print(f"\n{'='*42}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"{'='*42}")
    print("\nFull Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    # ── Plots ────────────────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm)
    plot_precision_recall(y_test, y_scores)

    # ── Demo predictions ─────────────────────────────────────────────────────
    samples = [
        "Congratulations! You've won a free iPhone. Click here to claim your prize.",
        "Hey, are we still on for lunch tomorrow?",
        "URGENT: Your bank account has been compromised. Reply with your password NOW.",
        "Can you pick up some groceries on your way home?",
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
        "I'll be home around 8. Want me to grab dinner?",
    ]

    print("\n--- Sample Predictions ---")
    for sms in samples:
        cleaned = preprocess(sms)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0][pred]
        label = "SPAM" if pred == 1 else "HAM "
        print(f"  [{label}  {prob:5.1%}]  {sms[:65]}")


if __name__ == "__main__":
    main()
