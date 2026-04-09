"""
evaluate.py
------------

Unified evaluator for:
- Classical ML models (RF, LR, XGBoost)
- TF-IDF pipeline model (tfidf_pipeline.pkl)

Key features:
- Automatically detects label column
- Handles both text labels and numeric labels
- Supports two dataset modes:
    A) Pre-engineered numeric datasets
    B) Raw-URL datasets (extract_all_features)
- Evaluates TF-IDF pipeline directly on raw URLs (no feature extraction)
- Computes Accuracy, Confusion Matrix, Classification Report
- Computes ROC-AUC if probabilities are available
"""

import os
import joblib
import pandas as pd
import numpy as np
import json

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

from src.features import extract_all_features
from src.config import MODEL_DIR


# --------------------------------------------------------
# Utility helpers
# --------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------
# Detect label column
# --------------------------------------------------------
def detect_label_column(df, verbose=True):
    candidates = [
        "label","Label","LABEL",
        "type","Type","TYPE",
        "class","Class","CLASS",
        "target","Target",
        "status","Status",
        "result","Result",
        "CLASS_LABEL","Category"
    ]

    for cand in candidates:
        if cand in df.columns:
            if verbose:
                print(f"🔎 Label column detected → {cand}")
            return df[cand].copy(), cand

    # Heuristic fallback
    for col in df.columns:
        if col.lower() == "url":
            continue
        if df[col].nunique() <= 50:
            print(f"ℹ️ Heuristic label detection → {col}")
            return df[col].copy(), col

    raise KeyError("❌ No valid label column found.")


# --------------------------------------------------------
# Normalize label values (numeric or text)
# --------------------------------------------------------
def clean_label_values(y):
    # numeric labels (-1/1 etc)
    if pd.api.types.is_numeric_dtype(y):
        y = y.replace(-1, 0).astype(int)
        uniq = sorted(y.unique())
        label_map = {str(x): int(x) for x in uniq}
        return y, label_map

    # text labels → integer map
    y_clean = y.astype(str).str.strip()
    lower = y_clean.str.lower().str.strip()
    uniq_lower = sorted(lower.unique())

    label_map = {v: i for i, v in enumerate(uniq_lower)}
    y_mapped = lower.map(label_map).astype(int)

    return y_mapped, label_map


# --------------------------------------------------------
# Detect dataset mode A (numeric) vs B (raw URLs)
# --------------------------------------------------------
def detect_dataset_mode(df):
    if any(c in df.columns for c in ["url","URL","Url"]):
        print("🔎 Detected URL column → Using RAW URL mode (Option B)")
        return "B"

    numeric_cols = df.select_dtypes(include=["int","float"]).columns
    if len(numeric_cols) >= 10:
        print("🔎 Many numeric columns → Using Numeric Feature mode (Option A)")
        return "A"

    raise ValueError("❌ Unable to detect dataset structure.")


# --------------------------------------------------------
# Main evaluator
# --------------------------------------------------------
def evaluate_models(df):
    print("\n📊 Evaluating saved models in:", MODEL_DIR)

    mode = detect_dataset_mode(df)

    # Detect label column + map labels → integers
    y_raw, label_col = detect_label_column(df)
    y_mapped, label_map = clean_label_values(y_raw)
    print("🔢 Label mapping (case-insensitive):", label_map)

    # ------------------------------------------------------------
    # Prepare feature matrix for classical models
    # TF-IDF model does NOT use this (uses raw URLs directly)
    # ------------------------------------------------------------
    print("\n🔍 Preparing evaluation feature matrix...")
    if mode == "A":
        print("📊 Using numeric dataset features")
        X = df.drop(columns=[label_col, "id"], errors="ignore")
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    else:
        print("🌐 Extracting numeric features from URLs (Option B)")
        url_col = next(c for c in ["url","URL","Url"] if c in df.columns)
        feature_rows = []
        for i, row in df.iterrows():
            try:
                feature_rows.append(extract_all_features(row[url_col], ""))
            except Exception as e:
                print(f"⚠️ Row {i} skipped: {e}")
                feature_rows.append({})
        X = pd.DataFrame(feature_rows).apply(pd.to_numeric, errors="coerce").fillna(0)

    print(f"   ✅ X.shape = {X.shape}")

    results = {}

    # ------------------------------------------------------------
    # Iterate all files in models/ directory
    # ------------------------------------------------------------
    for file in os.listdir(MODEL_DIR):

        # ========================================================
        # 1) Handle TF-IDF PIPELINE (NEW)
        # ========================================================
        if file == "tfidf_pipeline.pkl":
            print("\n🧠 Evaluating TF-IDF pipeline model")
            pipeline_path = os.path.join(MODEL_DIR, "tfidf_pipeline.pkl")
            try:
                pipeline = joblib.load(pipeline_path)
            except Exception as e:
                print("❌ Failed to load TF-IDF pipeline:", e)
                continue

            url_col = next((c for c in ["url","URL","Url"] if c in df.columns), None)
            if url_col is None:
                print("❌ URL column missing — cannot evaluate TF-IDF model")
                continue

            try:
                preds_raw = pipeline.predict(df[url_col].astype(str))
            except:
                print("❌ TF-IDF prediction failed")
                continue

            # Convert string → integer
            try:
                preds = pd.Series(preds_raw).str.lower().map(label_map).astype(int)
            except:
                preds = pd.Series(preds_raw).astype(int)

            acc = accuracy_score(y_mapped, preds)
            cm  = confusion_matrix(y_mapped, preds)
            report = classification_report(y_mapped, preds, output_dict=True)

            # ROC-AUC
            auc = None
            try:
                if hasattr(pipeline, "predict_proba"):
                    proba = pipeline.predict_proba(df[url_col].astype(str))
                    if len(label_map) == 2:
                        auc = roc_auc_score(y_mapped, proba[:,1])
                    else:
                        y_bin = label_binarize(
                            y_mapped, classes=sorted(label_map.values())
                        )
                        auc = roc_auc_score(
                            y_bin, proba, multi_class="ovr", average="macro"
                        )
            except:
                auc = None

            print(f"   🎯 Accuracy: {acc:.4f}, AUC: {auc}")
            print("   📉 Confusion Matrix:\n", cm)

            results["tfidf_pipeline"] = {
                "accuracy": float(acc),
                "roc_auc": float(auc) if auc else None,
                "confusion_matrix": cm.tolist(),
                "report": report
            }

            continue

        # ========================================================
        # Evaluate TF-IDF -> RF and TF-IDF -> XGB if present
        # ========================================================
        if file in ("tfidf_rf.pkl", "tfidf_xgb.pkl"):
            model_path = os.path.join(MODEL_DIR, file)
            print(f"\n🧠 Evaluating TF-IDF classical pipeline: {file}")
            try:
                pipe = joblib.load(model_path)
            except Exception as e:
                print("❌ Failed to load:", e)
                continue

            url_col = next((c for c in ["url","URL","Url"] if c in df.columns), None)
            if url_col is None:
                print("❌ URL column missing — cannot evaluate", file)
                continue

            try:
                preds_raw = pipe.predict(df[url_col].astype(str))
            except:
                print("❌ Prediction failed")
                continue

            # Convert predictions to integers
            try:
                preds = pd.Series(preds_raw).str.lower().map(label_map).astype(int)
            except:
                preds = pd.Series(preds_raw).astype(int)

            acc = accuracy_score(y_mapped, preds)
            cm = confusion_matrix(y_mapped, preds)
            report = classification_report(y_mapped, preds, output_dict=True)

            # ROC-AUC
            auc = None
            try:
                if hasattr(pipe, "predict_proba"):
                    proba = pipe.predict_proba(df[url_col].astype(str))
                    if len(label_map) == 2:
                        auc = roc_auc_score(y_mapped, proba[:, 1])
                    else:
                        y_bin = label_binarize(
                            y_mapped, classes=sorted(label_map.values())
                        )
                        auc = roc_auc_score(
                            y_bin, proba, multi_class="ovr", average="macro"
                        )
            except:
                auc = None

            print(f"   🎯 Accuracy: {acc:.4f}, AUC: {auc}")
            print("   📉 Confusion Matrix:\n", cm)

            results[file.replace(".pkl", "")] = {
                "accuracy": float(acc),
                "roc_auc": float(auc) if auc else None,
                "confusion_matrix": cm.tolist(),
                "report": report
            }

            continue
        '''
        # =================
        #  Char-CNN Evaluation
        # =================
        if file == "charcnn_model.pt":
            print("\n🧠 Evaluating Char-CNN model")

            from src.charcnn_predict import load_charcnn, encode_url_batch
            import torch
            from tqdm import tqdm

            model, vocab, cfg = load_charcnn(MODEL_DIR)
            maxlen = cfg["maxlen"]

            url_col = next((c for c in ["url", "URL", "Url"] if c in df.columns), None)
            urls = df[url_col].astype(str).tolist()

            batch_size = 256
            preds = []

            with torch.no_grad():
                for i in tqdm(range(0, len(urls), batch_size)):
                    batch = urls[i:i + batch_size]
                    Xb = encode_url_batch(batch, vocab, maxlen)
                    logits = model(Xb)
                    probs = torch.sigmoid(logits).cpu().numpy().ravel()
                    preds.extend((probs >= 0.5).astype(int))

            preds = np.array(preds).reshape(-1)

            acc = accuracy_score(y_mapped, preds)
            cm  = confusion_matrix(y_mapped, preds)
            report = classification_report(y_mapped, preds, output_dict=True)

            print(f"   🎯 Accuracy: {acc:.4f}")
            print("   📉 Confusion Matrix:\n", cm)

            results["charcnn"] = {
                "accuracy": float(acc),
                "roc_auc": None,
                "confusion_matrix": cm.tolist(),
                "report": report
            }
        '''

        # =================
        #  Char-CNN Evaluation
        # =================
        if file == "charcnn_model.pt":
            print("\n🧠 Evaluating Char-CNN model")

            from src.charcnn_predict import load_charcnn, encode_url_batch
            import torch
            from tqdm import tqdm
            from sklearn.metrics import roc_auc_score

            model, vocab, cfg = load_charcnn(MODEL_DIR)
            maxlen = cfg["maxlen"]

            url_col = next((c for c in ["url", "URL", "Url"] if c in df.columns), None)
            urls = df[url_col].astype(str).tolist()

            batch_size = 256
            preds = []
            probs_list = []

            with torch.no_grad():
                for i in tqdm(range(0, len(urls), batch_size)):
                    batch = urls[i:i+batch_size]
                    Xb = encode_url_batch(batch, vocab, maxlen)
                    logits = model(Xb)                     # raw outputs
                    probs = torch.sigmoid(logits).cpu().numpy().ravel()

                    probs_list.extend(probs.tolist())      # <-- store probabilities
                    preds.extend((probs >= 0.5).astype(int))

            preds = np.array(preds).reshape(-1)

            acc = accuracy_score(y_mapped, preds)
            auc = roc_auc_score(y_mapped, probs_list)      # <-- compute AUC

            cm  = confusion_matrix(y_mapped, preds)
            report = classification_report(y_mapped, preds, output_dict=True)

            print(f"   🎯 Accuracy: {acc:.4f}")
            print(f"   📈 AUC: {auc:.4f}")
            print("   📉 Confusion Matrix:\n", cm)

            results["charcnn"] = {
                "accuracy": float(acc),
                "roc_auc": float(auc),
                "confusion_matrix": cm.tolist(),
                "report": report
            }


            continue

        # ========================================================
        # 2) Classical Models
        # ========================================================
        if not file.endswith("_model.pkl"):
            continue

        model_name = file.replace("_model.pkl", "")
        model_path = os.path.join(MODEL_DIR, file)
        scaler_path = os.path.join(MODEL_DIR, f"{model_name}_scaler.pkl")

        print(f"\n🧠 Evaluating classical model: {model_name}")

        # ------------------------------------------------------------------
        # FIX ADDED: Load shared scaler + saved feature names
        # ------------------------------------------------------------------
        feature_names = joblib.load(f"{MODEL_DIR}/classical_feature_names.pkl")  ### <-- FIX ADDED
        scaler = joblib.load(f"{MODEL_DIR}/classical_scaler.pkl")              ### <-- FIX ADDED
        model = joblib.load(model_path)                                        ### <-- FIX ADDED

        # Force X to match training feature columns
        X_eval = X.copy()                                                      ### <-- FIX ADDED

        # Add missing columns
        for col in feature_names:                                              ### <-- FIX ADDED
            if col not in X_eval.columns:
                X_eval[col] = 0

        # Drop extra columns
        X_eval = X_eval[feature_names]                                         ### <-- FIX ADDED

        # Scale correctly
        X_scaled = scaler.transform(X_eval)                                    ### <-- FIX ADDED
        # ------------------------------------------------------------------

        preds = model.predict(X_scaled)

        acc = accuracy_score(y_mapped, preds)
        cm  = confusion_matrix(y_mapped, preds)
        report = classification_report(y_mapped, preds, output_dict=True)

        # ROC-AUC
        auc = None
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_scaled)
            elif hasattr(model, "decision_function"):
                dec = model.decision_function(X_scaled)
                if dec.ndim == 1:
                    pos = sigmoid(dec)
                    proba = np.stack([1-pos, pos], axis=1)
                else:
                    proba = softmax(dec)
            else:
                proba = None

            if proba is not None:
                if len(label_map) == 2:
                    auc = roc_auc_score(y_mapped, proba[:,1])
                else:
                    y_bin = label_binarize(
                        y_mapped, classes=sorted(np.unique(y_mapped))
                    )
                    auc = roc_auc_score(
                        y_bin, proba, multi_class="ovr", average="macro"
                    )
        except:
            auc = None

        print(f"   🎯 Accuracy: {acc:.4f}, AUC: {auc}")
        print("   📉 Confusion Matrix:\n", cm)

        results[model_name] = {
            "accuracy": float(acc),
            "roc_auc": float(auc) if auc else None,
            "confusion_matrix": cm.tolist(),
            "report": report
        }

    return results


# --------------------------------------------------------
# Standalone execution
# --------------------------------------------------------
if __name__ == "__main__":
    from src.config import DATA_PATH
    print("\n🚀 Standalone evaluation starting...")

    if not os.path.exists(DATA_PATH):
        print("❌ Dataset not found:", DATA_PATH)
        exit()

    df = pd.read_csv(DATA_PATH)

    # Run full evaluation (all models auto-detected)
    results = evaluate_models(df)

    print("\n📊 FINAL SUMMARY")
    for name, r in results.items():
        print(f"{name:<20} | Accuracy={r['accuracy']} | AUC={r['roc_auc']}")

    # -----------------------------------------
    # SAVE METRICS FOR VISUALIZATION
    # -----------------------------------------
    print("\n💾 Saving metrics JSON files...")
    os.makedirs("metrics", exist_ok=True)

    for model_name, r in results.items():
        report = r["report"]     # already in output_dict format
        with open(f"metrics/{model_name}_metrics.json", "w") as f:
            json.dump(report, f, indent=4)
        print(f"   ✔ Saved → metrics/{model_name}_metrics.json")

    print("\n✅ Evaluation complete. Now run:")
    print("   python visualize.py")
