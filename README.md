# Spam SMS Classifier

A machine learning web app that detects spam SMS messages using TF-IDF features and traditional ML classifiers. Built with Python, scikit-learn, NLTK, and Streamlit.

## Live Demo

**[https://spam-sms-classifier-afbcxcejp7c4ybuzmdotwk.streamlit.app](https://spam-sms-classifier-afbcxcejp7c4ybuzmdotwk.streamlit.app)**

## Features

- **Single message classifier** — paste any SMS and get an instant spam/ham prediction with confidence score
- **Word highlighting** — see exactly which words triggered the spam detection
- **Adjustable threshold** — tune the spam cutoff to balance precision vs recall
- **Bulk classification** — upload a CSV of messages, classify all at once, download results
- **Session history** — track every message classified in the current session
- **Model comparison** — Naive Bayes vs Logistic Regression vs LinearSVC side by side
- **Word clouds** — visual breakdown of top spam and ham vocabulary
- **Model persistence** — trained models saved to disk, instant load on restart

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Naive Bayes | 96.5% | 100% | 73.8% | 84.9% |
| Logistic Regression | ~98% | ~97% | ~90% | ~93% |
| LinearSVC | ~98% | ~97% | ~91% | ~94% |

> Evaluated on a stratified 80/20 train-test split of the UCI SMS Spam Collection (5,572 messages).

## Tech Stack

- **Language**: Python 3.10+
- **ML**: scikit-learn (TF-IDF, Naive Bayes, Logistic Regression, LinearSVC)
- **NLP**: NLTK (tokenization, stopword removal, Porter stemming)
- **App**: Streamlit
- **Data**: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Viraj-Pathak/spam-sms-classifier.git
cd spam-sms-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the web app
python -m streamlit run app.py
```

The dataset downloads automatically on first run. Trained models are cached to disk so subsequent starts are instant.

To run the plain training script without the UI:
```bash
python spam_classifier.py
```

## Project Structure

```
spam-sms-classifier/
├── app.py               # Streamlit web app
├── spam_classifier.py   # Standalone training + evaluation script
├── requirements.txt
├── .gitignore
└── README.md
```

## What You Learn

- Converting raw SMS text to TF-IDF feature vectors
- Training and comparing Naive Bayes, Logistic Regression, and SVM classifiers
- Evaluating with F1-score, precision-recall curves, and confusion matrices
- Building an interactive ML app with Streamlit
- Model persistence with joblib
