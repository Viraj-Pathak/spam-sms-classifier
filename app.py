import os
import urllib.request
import zipfile
import re

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import streamlit as st

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
        return
    zip_path = os.path.join(DATA_DIR, "smsspamcollection.zip")
    urllib.request.urlretrieve(DATASET_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract("SMSSpamCollection", DATA_DIR)
    os.remove(zip_path)


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [_stemmer.stem(t) for t in tokens if t not in _stop_words]
    return " ".join(tokens)


@st.cache_resource(show_spinner="Training models on SMS Spam Collection…")
def load_models():
    download_dataset()
    df = pd.read_csv(
        DATA_FILE, sep="\t", header=None,
        names=["label", "message"], encoding="latin-1",
    )
    df["cleaned"] = df["message"].apply(preprocess)
    df["label_enc"] = (df["label"] == "spam").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned"], df["label_enc"],
        test_size=0.2, random_state=42, stratify=df["label_enc"],
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    model_defs = {
        "Naive Bayes":         MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "LinearSVC":           LinearSVC(max_iter=1000, random_state=42),
    }

    trained, metrics = {}, {}
    for name, m in model_defs.items():
        m.fit(X_train_tfidf, y_train)
        y_pred = m.predict(X_test_tfidf)
        if hasattr(m, "predict_proba"):
            y_scores = m.predict_proba(X_test_tfidf)[:, 1]
        else:
            y_scores = m.decision_function(X_test_tfidf)
        trained[name] = m
        metrics[name] = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall":    recall_score(y_test, y_pred),
            "f1":        f1_score(y_test, y_pred),
            "cm":        confusion_matrix(y_test, y_pred),
            "report":    classification_report(y_test, y_pred, target_names=["ham", "spam"]),
            "pr_curve":  precision_recall_curve(y_test, y_scores),
        }

    return trained, vectorizer, metrics, df


def spam_probability(model, vec) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(vec)[0][1])
    score = float(model.decision_function(vec)[0])
    return float(1 / (1 + np.exp(-score)))


def top_spam_tokens(model, vectorizer, vec, n=12):
    if not hasattr(model, "feature_log_prob_"):
        return []
    names   = vectorizer.get_feature_names_out()
    diff    = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    tfidf   = vec.toarray()[0]
    contrib = diff * tfidf
    idx     = contrib.argsort()[-n:][::-1]
    return [(names[i], float(contrib[i])) for i in idx if contrib[i] > 0]


def highlight_html(message: str, spam_tokens: list) -> str:
    token_set = {w for w, _ in spam_tokens}
    parts = []
    for word in message.split():
        stem = _stemmer.stem(re.sub(r"[^a-z]", "", word.lower()))
        if stem in token_set:
            parts.append(
                f'<mark style="background:#ff4b4b;color:white;'
                f'padding:2px 5px;border-radius:4px;font-weight:600">{word}</mark>'
            )
        else:
            parts.append(word)
    return " ".join(parts)


# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Spam SMS Classifier", page_icon="📩", layout="wide")
st.title("📩 Spam SMS Classifier")
st.caption("TF-IDF · Naive Bayes · Logistic Regression · LinearSVC  |  UCI SMS Spam Collection (5,572 messages)")

trained_models, vectorizer, all_metrics, df = load_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox("Active model", list(trained_models.keys()))
    threshold = st.slider(
        "Spam threshold", 0.10, 0.90, 0.50, 0.05,
        help="Lower = catch more spam (higher recall). Higher = fewer false alarms (higher precision).",
    )
    st.caption(f"Spam if probability ≥ **{threshold:.0%}**")
    st.divider()
    st.markdown("**Model F1 scores**")
    for name, m in all_metrics.items():
        st.metric(name, f"{m['f1']:.2%}", delta=None)

active_model = trained_models[model_choice]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Classify", "📦 Bulk Classify", "📊 Model Performance", "🗄️ Dataset"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 · Classify
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Classify a single message")

    col_in, col_stats = st.columns([3, 1])
    with col_in:
        user_input = st.text_area(
            "Message", placeholder="Type or paste an SMS here…",
            height=130, label_visibility="collapsed", key="msg_input",
        )
    with col_stats:
        st.metric("Characters", len(user_input))
        st.metric("Words", len(user_input.split()) if user_input.strip() else 0)

    examples = {
        "🚨 Spam 1": "Congratulations! You've won a FREE iPhone. Click here to claim your prize now!",
        "🚨 Spam 2": "URGENT: Your bank account is suspended. Verify details or lose access permanently.",
        "✅ Ham 1":  "Hey, are we still on for lunch tomorrow?",
        "✅ Ham 2":  "I'll be home around 8. Want me to grab dinner?",
    }
    btn_cols = st.columns(len(examples))
    for col, (label, text) in zip(btn_cols, examples.items()):
        if col.button(label, use_container_width=True):
            st.session_state["_prefill"] = text
            st.rerun()
    if "_prefill" in st.session_state:
        user_input = st.session_state.pop("_prefill")

    if st.button("Classify", type="primary", use_container_width=True):
        if not user_input.strip():
            st.warning("Please enter a message first.")
        else:
            cleaned = preprocess(user_input)
            vec     = vectorizer.transform([cleaned])
            prob    = spam_probability(active_model, vec)
            is_spam = prob >= threshold

            if is_spam:
                st.error(f"🚨 **SPAM** — {prob:.1%} spam probability")
            else:
                st.success(f"✅ **HAM** (not spam) — {1-prob:.1%} ham confidence")

            st.progress(prob, text=f"Spam probability: {prob:.1%}")

            # Word highlighting for Naive Bayes
            top_tokens = top_spam_tokens(active_model, vectorizer, vec)
            if top_tokens and is_spam:
                st.markdown("**Spam trigger words** (highlighted in message):")
                st.markdown(highlight_html(user_input, top_tokens), unsafe_allow_html=True)
                with st.expander("Feature contribution scores"):
                    st.dataframe(
                        pd.DataFrame(top_tokens, columns=["token (stemmed)", "contribution"]),
                        use_container_width=True, hide_index=True,
                    )

            # Session history
            if "history" not in st.session_state:
                st.session_state["history"] = []
            st.session_state["history"].insert(0, {
                "Message":    user_input[:75] + ("…" if len(user_input) > 75 else ""),
                "Result":     "SPAM" if is_spam else "HAM",
                "Confidence": f"{prob:.1%}",
                "Model":      model_choice,
            })

    # History table
    if st.session_state.get("history"):
        st.divider()
        hcol, ccol = st.columns([5, 1])
        hcol.markdown("**Session history**")
        if ccol.button("Clear history"):
            st.session_state["history"] = []
            st.rerun()
        st.dataframe(
            pd.DataFrame(st.session_state["history"]),
            use_container_width=True, hide_index=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 · Bulk Classify
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Classify messages in bulk")
    st.markdown("Upload a **CSV** with a column named `message`. Results download as CSV.")

    # Sample CSV download
    with st.expander("Need a sample CSV to test with?"):
        sample = pd.DataFrame({"message": [
            "Congratulations! You've won a $1000 Walmart gift card. Claim now!",
            "Hey, see you at the gym later?",
            "URGENT: Claim your free prize now — limited time offer!",
            "I'll call you when I land.",
            "Win a holiday for 2! Text WIN to 80082 to enter.",
            "Can you grab some milk on the way home?",
        ]})
        st.dataframe(sample, use_container_width=True, hide_index=True)
        st.download_button(
            "Download sample CSV", sample.to_csv(index=False).encode(),
            "sample_messages.csv", "text/csv",
        )

    uploaded = st.file_uploader("Upload your CSV", type=["csv"])
    if uploaded:
        try:
            bulk_df = pd.read_csv(uploaded)
            if "message" not in bulk_df.columns:
                st.error("CSV must contain a column named `message`.")
            else:
                st.info(f"Loaded **{len(bulk_df)}** messages.")
                if st.button("Run bulk classification", type="primary"):
                    with st.spinner(f"Classifying {len(bulk_df)} messages…"):
                        cleaned  = bulk_df["message"].astype(str).apply(preprocess)
                        vecs     = vectorizer.transform(cleaned)
                        probs    = [spam_probability(active_model, vectorizer.transform([c])) for c in cleaned]
                        bulk_df["spam_probability"] = [f"{p:.1%}" for p in probs]
                        bulk_df["prediction"]       = ["SPAM" if p >= threshold else "HAM" for p in probs]

                    spam_n = (bulk_df["prediction"] == "SPAM").sum()
                    ham_n  = (bulk_df["prediction"] == "HAM").sum()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total messages", len(bulk_df))
                    c2.metric("Spam detected",  spam_n)
                    c3.metric("Ham messages",   ham_n)

                    st.dataframe(
                        bulk_df[["message", "prediction", "spam_probability"]],
                        use_container_width=True, hide_index=True,
                    )
                    st.download_button(
                        "⬇️ Download results as CSV",
                        bulk_df.to_csv(index=False).encode(),
                        "spam_results.csv", "text/csv",
                    )
        except Exception as e:
            st.error(f"Could not read file: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 · Model Performance
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Model comparison")

    summary = pd.DataFrame([{
        "Model":     name,
        "Accuracy":  f"{r['accuracy']:.2%}",
        "Precision": f"{r['precision']:.2%}",
        "Recall":    f"{r['recall']:.2%}",
        "F1 Score":  f"{r['f1']:.2%}",
    } for name, r in all_metrics.items()])
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.divider()
    detail = st.selectbox("Detailed view:", list(all_metrics.keys()), key="detail_sel")
    r = all_metrics[detail]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{r['accuracy']:.2%}")
    c2.metric("Precision", f"{r['precision']:.2%}")
    c3.metric("Recall",    f"{r['recall']:.2%}")
    c4.metric("F1 Score",  f"{r['f1']:.2%}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=["ham","spam"], yticklabels=["ham","spam"], ax=ax)
        ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
        plt.tight_layout(); st.pyplot(fig)

    with col_b:
        st.markdown("**Precision-Recall Curves (all models)**")
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        for name, res in all_metrics.items():
            p_arr, r_arr, _ = res["pr_curve"]
            ax2.plot(r_arr, p_arr, lw=2, label=name)
        ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
        ax2.legend(fontsize=8); plt.tight_layout(); st.pyplot(fig2)

    st.markdown("**Classification Report**")
    st.code(r["report"], language="text")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 · Dataset
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Dataset overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total messages", len(df))
    c2.metric("Spam",  int(df["label_enc"].sum()))
    c3.metric("Ham",   int((df["label_enc"] == 0).sum()))

    col_dist, col_wc = st.columns(2)

    with col_dist:
        st.markdown("**Class distribution**")
        fig3, ax3 = plt.subplots(figsize=(4, 3))
        counts = df["label"].value_counts()
        ax3.bar(counts.index, counts.values, color=["steelblue", "tomato"])
        ax3.set_ylabel("Count"); plt.tight_layout(); st.pyplot(fig3)

    with col_wc:
        st.markdown("**Word cloud**")
        wc_type = st.radio("Show words for:", ["spam", "ham"], horizontal=True)
        corpus  = " ".join(df[df["label"] == wc_type]["cleaned"].dropna().values)
        wc = WordCloud(
            width=500, height=300, background_color="white",
            colormap="Reds" if wc_type == "spam" else "Blues",
        ).generate(corpus)
        fig4, ax4 = plt.subplots(figsize=(5, 3))
        ax4.imshow(wc, interpolation="bilinear"); ax4.axis("off")
        plt.tight_layout(); st.pyplot(fig4)

    st.markdown("**Sample messages**")
    filter_label = st.radio("Filter by:", ["all", "spam", "ham"], horizontal=True)
    sample_df = df if filter_label == "all" else df[df["label"] == filter_label]
    st.dataframe(
        sample_df[["label", "message"]].sample(min(10, len(sample_df)), random_state=7).reset_index(drop=True),
        use_container_width=True,
    )
