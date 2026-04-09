"""
Main entrypoint for the Phishing URL Detection project.

This script:
1. Loads the dataset
2. Trains multiple classical ML models (RF, LR, XGBoost)
3. Trains the RNN model on raw URL sequences ---- This is for the next phase
4. Saves all trained models in the /models directory
"""

import pandas as pd
#from src import train_models, train_rnn_model
from src.train_classical import train_models
from src.tfidf_model import train as train_tfidf
from src.config import DATA_PATH
from src.evaluate import evaluate_models


def main():
    print("=" * 80)
    print("🚀 PHISHING URL DETECTION PROJECT - TRAINING PIPELINE")
    print("=" * 80)
    
    # --- Phase 1: Load Dataset ---
    print("\n📥 Loading dataset...")
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"✅ Dataset loaded successfully! Total samples: {len(df)}")
    except FileNotFoundError:
        print(f"❌ Dataset not found at {DATA_PATH}. Please check the path in config.py.")
        return

    print("\n📋 Columns in dataset:", df.columns.tolist())
    print(df.head())
    # --- Phase 2: Train Classical ML Models ---
    print("\n🧠 Training Classical ML Models (RandomForest, LogisticRegression, XGBoost)...")
    model_results = train_models(df)
    # --- Phase 2B: Train TF-IDF Model ---
    print("\n🧠 Training TF-IDF Model (LogReg / SVM on URL text)...")
    try:
        print("DEBUG: Entered TF-IDF training block")
        tfidf_accuracy = train_tfidf(
            data_path=DATA_PATH,
            url_col="url",
            label_col="type",
            model_dir="models",
            model="logreg"
        )
        print("✅ TF-IDF model trained and saved!")
    except Exception as e:
        print(f"❌ TF-IDF training failed: {e}")
        tfidf_accuracy = None

        # --- Phase 2C: Train TF-IDF -> RF / XGBoost Classical Pipelines ---
    print("\n🧠 Training Additional TF-IDF Classical Pipelines (RF & XGBoost)...")
    try:
        from src.train_tfidf_classical import train_from_dataframe
        tfidf_classical_results = train_from_dataframe(df)
        print("✅ TF-IDF Classical pipelines trained and saved!")
        print("   📌 Saved models:")
        for k, v in tfidf_classical_results.items():
            print(f"   - {k}: {v}")
    except Exception as e:
        print(f"❌ TF-IDF Classical pipelines failed: {e}")


        # --- Phase 2D: Train Char-CNN deep model (PyTorch) ---
    print("\n🤖 Training Char-CNN deep model (PyTorch, CPU-friendly)...")
    try:
        from src.train_char_cnn import train_charcnn
        # this will read DATA_PATH from src.config if you call without args
        train_charcnn(
            data_path=DATA_PATH,
            url_col="url",
            label_col="type",
            model_dir="models",
            epochs=3,
            batch_size=256
        )
        print("✅ Char-CNN model trained and saved!")
    except Exception as e:
        print(f"⚠️ Char-CNN training failed/skipped: {e}")

    

    print("\n📊 Model Performance Summary:")
    for model_name, metrics in model_results.items():
        print(f"   {model_name:<25} Accuracy: {metrics['accuracy']:.4f}")

    if tfidf_accuracy is not None:
        print(f"   TF-IDF Model              Accuracy: {tfidf_accuracy:.4f}")
    else:
        print("   TF-IDF Model              Accuracy: FAILED")


    '''
    # --- Phase 3: Train RNN Model ---
    print("\n🤖 Training RNN (LSTM/GRU) on URL text sequences...")
    try:
        rnn_accuracy = train_rnn_model(df)
        print(f"✅ RNN Model Accuracy: {rnn_accuracy:.4f}")
    except Exception as e:
        print(f"⚠️ RNN training skipped or failed: {e}")
    '''

    print("\n🏁 Training complete! All models are saved under the /models directory.")
    print("=" * 80)

    # ... after training
    print("\n📊 Evaluating models after training...")
    evaluate_models(df)


if __name__ == "__main__":
    main()
