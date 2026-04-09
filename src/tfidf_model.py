import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold


def load_data(path, url_col="url", label_col="label"):
    df = pd.read_csv(path)

    if url_col not in df.columns:
        for c in df.columns:
            if "url" in c.lower():
                url_col = c

    if label_col not in df.columns:
        for c in df.columns:
            if "type" in c.lower() or "class" in c.lower():
                label_col = c

    df = df[[url_col, label_col]].dropna()

    # normalize labels
    df[label_col] = df[label_col].astype(str).str.lower().str.strip()

    # remove duplicates
    df = df.drop_duplicates()

    return df, url_col, label_col


def train(
    data_path,
    url_col="url",
    label_col="label",
    model_dir="models",
    model="svm",
    test_size=0.2
):
    df, url_col, label_col = load_data(data_path, url_col, label_col)

    X = df[url_col].astype(str).values
    y = df[label_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # ------------------------------
    # BEST TF-IDF SETTINGS FOR URLS
    # ------------------------------
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 7),
        min_df=5,
        max_df=0.95,
        sublinear_tf=True          # IMPORTANT for URL text
    )

    # -------------------------------
    # Best performing classifier
    # -------------------------------
    base_clf = LinearSVC()
    # clf = CalibratedClassifierCV(base_clf, cv=5)  # gives predict_proba() this gave error hence the below
    clf = CalibratedClassifierCV(base_clf, cv=StratifiedKFold(n_splits=3))

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", clf)
    ])

    print("Training improved TF-IDF + SVM pipeline...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    preds = pipeline.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, preds))
    print("\nClassification report:\n", classification_report(y_test, preds))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    os.makedirs(model_dir, exist_ok=True)
    out_path = os.path.join(model_dir, "tfidf_pipeline.pkl")

    with open(out_path, "wb") as f:
        pickle.dump(pipeline, f)

    print("\n✅ Saved unified TF-IDF pipeline →", out_path)

    return accuracy_score(y_test, preds)
