# train_all.py
import subprocess

print("Training classical models (RF, LR, XGB)...")
subprocess.run(["python", "train_classical.py"])

print("Training TF-IDF model...")
subprocess.run([
    "python", "tfidf_model.py",
    "--data_path", "malicious_phish.csv",
    "--url_col", "url", "--label_col", "label"
])

# Training TF-IDF -> RF and TF-IDF -> XGBOOST
print("Training TF-IDF -> classical (RF & XGB) additional pipelines...")
subprocess.run(["python", "train_tfidf_classical.py"])


print("ALL MODELS TRAINED AND SAVED.")
