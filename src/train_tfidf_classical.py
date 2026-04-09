# src/train_tfidf_classical.py
"""
Train TF-IDF -> RandomForest and TF-IDF -> XGBoost pipelines.

Saves:
 - models/tfidf_rf.pkl
 - models/tfidf_xgb.pkl

This file is intentionally standalone and does NOT touch your existing
tfidf_model.py or engineered-feature pipelines.
"""
import os
import joblib
import pandas as pd
import time

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from src.config import MODEL_DIR, DATA_PATH

TFIDF_RF_FILENAME = "tfidf_rf.pkl"
TFIDF_XGB_FILENAME = "tfidf_xgb.pkl"
TFIDF_RF_PATH = os.path.join(MODEL_DIR, TFIDF_RF_FILENAME)
TFIDF_XGB_PATH = os.path.join(MODEL_DIR, TFIDF_XGB_FILENAME)

def _get_tfidf_vectorizer():
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),    # optimized to reduce memory
        lowercase=True,
        min_df=5,            
        max_features=300_000,     # 🚀 HUGE MEMORY SAVING
        strip_accents="unicode",
    )

def _detect_label_col(df):
    candidates = [
        "label","Label","LABEL",
        "type","Type","TYPE",
        "class","Class","CLASS",
        "target","Target",
        "status","Status",
        "result","Result",
        "CLASS_LABEL","Category"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback heuristic (low-cardinality)
    for col in df.columns:
        if col.lower() == "url":
            continue
        if df[col].nunique() <= 50:
            return col
    raise KeyError("No label column found in dataframe.")

def _prepare_labels(y):
    if pd.api.types.is_numeric_dtype(y):
        return y.replace(-1, 0).astype(int)
    y_clean = y.astype(str).str.strip().str.lower()
    mapping = {v: i for i, v in enumerate(sorted(y_clean.unique()))}
    return y_clean.map(mapping).astype(int)

def train_from_dataframe(df, url_col_candidates=("url","URL","Url"), test_size=0.2, random_state=42):
    # detect label
    label_col = _detect_label_col(df)
    y = _prepare_labels(df[label_col])

    # detect URL column
    url_col = next((c for c in url_col_candidates if c in df.columns), None)
    if url_col is None:
        raise KeyError("No URL column found. TF-IDF classical pipelines require a raw URL column.")

    X = df[url_col].astype(str).fillna("")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("Train / Test sizes:", X_train.shape, X_test.shape)

    # TF-IDF -> RandomForest
    rf_pipe = Pipeline([
        ("tfidf", _get_tfidf_vectorizer()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state
        ))
    ])

    print("\nTraining TF-IDF -> RandomForest ...")
    # -------------------------------
    # STEP 1 — FIT TF-IDF VECTORIZER
    # -------------------------------
    print("\n🟦 Step 1/3: Fitting TF-IDF vectorizer (this is the largest step)...")
    start = time.time()

    tfidf = rf_pipe.named_steps["tfidf"]
    tfidf.fit(X_train)     # fit only on training URLs

    end = time.time()
    vocab_size = len(tfidf.vocabulary_)
    print(f"🟩 TF-IDF fit complete! Vocab size = {vocab_size:,}  (Time: {end-start:.2f} sec)")

    # Transform train & test sets
    print("🟦 Transforming train & test URLs into TF-IDF vectors...")
    X_train_tfidf = tfidf.transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    print("🟩 Transform complete.")

    # -------------------------------
    # STEP 2 — TRAIN RANDOM FOREST
    # -------------------------------
    print("\n🌲 Step 2/3: Training RandomForest (300 trees)...")
    start = time.time()

    rf = rf_pipe.named_steps["clf"]
    rf.fit(X_train_tfidf, y_train)

    end = time.time()
    print(f"🌲 RandomForest training complete! (Time: {end-start:.2f} sec)")

    # -----------------------------------------
    # REBUILD FITTED RF PIPELINE (REQUIRED)
    # -----------------------------------------
    rf_pipe_fitted = Pipeline([
        ("tfidf", tfidf),      # fitted TF-IDF
        ("clf", rf)            # fitted RandomForest
    ])

    joblib.dump(rf_pipe_fitted, TFIDF_RF_PATH)
    print(f"💾 Saved TF-IDF → RandomForest pipeline to: {TFIDF_RF_PATH}")


    #print(f"💾 Saved TF-IDF → RandomForest pipeline to: {TFIDF_RF_PATH}")

    # TF-IDF -> XGBoost
    xgb_pipe = Pipeline([
        ("tfidf", _get_tfidf_vectorizer()),
        ("clf", XGBClassifier(
            objective="binary:logistic" if y.nunique() == 2 else "multi:softprob",
            eval_metric="logloss",
            n_estimators=150,        # reduce from 300 → 150
            max_depth=5,             # reduce from 6–9 → 5
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.8,    # reduce from 0.9 → 0.8
            tree_method="hist",      # MOST IMPORTANT → low memory RAM-safe
            n_jobs=-1,
            random_state=random_state,
        ))
    ])

    print("\nTraining TF-IDF -> XGBoost ...")
    # -------------------------------
    # STEP 3 — TRAIN XGBOOST
    # -------------------------------
    print("\n⚙️ Step 3/3: Training XGBoost (150 trees)...")
    start = time.time()

    xgb_model = xgb_pipe.named_steps["clf"]
    xgb_model.fit(X_train_tfidf, y_train)

    end = time.time()
    print(f"⚙️ XGBoost training complete! (Time: {end-start:.2f} sec)")

    # Attach trained model back into pipeline
    # -----------------------------------------
    # REBUILD FITTED XGB PIPELINE (REQUIRED)
    # -----------------------------------------
    xgb_pipe_fitted = Pipeline([
        ("tfidf", tfidf),        # reuse the same fitted TF-IDF
        ("clf", xgb_model)       # fitted XGBoost
    ])

    joblib.dump(xgb_pipe_fitted, TFIDF_XGB_PATH)
    print(f"💾 Saved TF-IDF → XGBoost pipeline to: {TFIDF_XGB_PATH}")

    print(f"💾 Saved TF-IDF → XGBoost pipeline to: {TFIDF_XGB_PATH}")


    return {
        "tfidf_rf_path": TFIDF_RF_PATH,
        "tfidf_xgb_path": TFIDF_XGB_PATH
    }

if __name__ == "__main__":
    print("Loading data from:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    train_from_dataframe(df)
