"""
Thiranex Phishing Email Detection Model
Educational demo: trains a text + URL-feature classifier from data/emails.csv.
"""
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "emails.csv"
MODEL_TEXT_FEATURES = None

SUSPICIOUS_WORDS = [
    "urgent", "verify", "password", "suspended", "claim", "prize", "bank",
    "payment", "locked", "immediately", "credentials", "refund", "click",
    "confirm", "expire", "unusual activity", "one-time code"
]

def url_features(text):
    """Return simple numeric features from email text/URLs."""
    text = str(text).lower()
    urls = re.findall(r'https?://\S+|www\.\S+', text)
    url_text = " ".join(urls)
    return [
        len(urls),
        int("http://" in text),
        int(bool(re.search(r'https?://\d{1,3}(?:\.\d{1,3}){3}', text))),
        int("@" in url_text),
        int(any(word in text for word in SUSPICIOUS_WORDS)),
        int(any(word in text for word in ["password", "pin", "bank details", "card number", "credentials"])),
        len(text),
        text.count("!"),
        int("bit.ly" in text or "tinyurl" in text),
    ]

def build_features(texts, vectorizer=None, fit=False):
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2),
            max_features=5000, sublinear_tf=True
        )
    text_matrix = vectorizer.fit_transform(texts) if fit else vectorizer.transform(texts)
    url_matrix = csr_matrix([url_features(t) for t in texts])
    return hstack([text_matrix, url_matrix]), vectorizer

def train_and_evaluate():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH).dropna(subset=["subject", "body", "label"])
    df["text"] = df["subject"].astype(str) + " " + df["body"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )
    X_train_vec, vectorizer = build_features(X_train.tolist(), fit=True)
    X_test_vec, _ = build_features(X_test.tolist(), vectorizer=vectorizer)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train_vec, y_train)
    predictions = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, predictions)
    print("\n=== Thiranex Phishing Email Detection ===")
    print(f"Dataset rows: {len(df)}")
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (rows = actual, columns = predicted):")
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, labels=["Phishing", "Safe"], cmap="Blues", values_format="d"
    )
    plt.title("Phishing Email Detection — Confusion Matrix")
    plt.tight_layout()
    output = ROOT / "confusion_matrix.png"
    plt.savefig(output, dpi=150)
    print(f"Confusion matrix saved to: {output}")
    return model, vectorizer

def predict_email(model, vectorizer, subject, body):
    text = f"{subject} {body}"
    X, _ = build_features([text], vectorizer=vectorizer)
    prediction = model.predict(X)[0]
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        confidence = max(probabilities) * 100
    else:
        confidence = None
    return prediction, confidence

def main():
    model, vectorizer = train_and_evaluate()
    print("\nTry an email. This is an educational demo, not a guarantee of safety.")
    while True:
        subject = input("\nEmail subject (or type exit): ").strip()
        if subject.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        body = input("Email body: ").strip()
        result, confidence = predict_email(model, vectorizer, subject, body)
        print(f"\nPrediction: {result}")
        if confidence is not None:
            print(f"Model confidence: {confidence:.1f}% (not a guarantee)")
        print("Safety tip: do not open unexpected links or share passwords/OTP details.")

if __name__ == "__main__":
    main()
