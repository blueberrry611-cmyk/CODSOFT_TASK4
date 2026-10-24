import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

# 1. Load data (this dataset needs latin-1 encoding)
df = pd.read_csv("spam.csv", encoding="latin-1", sep=None, engine="python")
print("Columns found:", df.columns.tolist())

# Find label and text columns automatically
cols = {c.lower().strip(): c for c in df.columns}
label_col = next(cols[k] for k in cols if k in ("v1", "label", "class", "category", "type"))
text_col = next(cols[k] for k in cols if k in ("v2", "sms", "text", "message", "content"))

df = df[[label_col, text_col]].dropna()
df.columns = ["label", "text"]
df["label"] = df["label"].astype(str).str.strip().str.lower()
df = df.drop_duplicates().reset_index(drop=True)
df["label_num"] = df["label"].map({"ham": 0, "spam": 1, "0": 0, "1": 1})

print(df["label"].value_counts())

# 2. Split (stratify keeps the spam ratio the same in train and test)
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label_num"], test_size=0.2, random_state=42,
    stratify=df["label_num"]
)

# 3. TF-IDF features
tfidf = TfidfVectorizer(stop_words="english", lowercase=True,
                        ngram_range=(1, 2), max_df=0.95, min_df=2)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# 4. Train and compare models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Linear SVM": LinearSVC(class_weight="balanced"),
}

results = {}
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    results[name] = {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
    }
    print(f"\n=== {name} ===")
    print(classification_report(y_test, preds, target_names=["ham", "spam"]))

results_df = pd.DataFrame(results).T
print("\nModel comparison:\n", results_df.round(4))

# 5. Confusion matrix for the best model (by F1)
best_name = results_df["F1"].idxmax()
best_model = models[best_name]
cm = confusion_matrix(y_test, best_model.predict(X_test_vec))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png", bbox_inches="tight")
plt.show()

# 6. Save the best model and vectorizer
joblib.dump(best_model, "spam_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

# 7. Demo: classify new messages
def predict_sms(message):
    vec = tfidf.transform([message])
    return "SPAM" if best_model.predict(vec)[0] == 1 else "HAM (legitimate)"

samples = [
    "Congratulations! You've won a free iPhone. Click here to claim now!",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT: Your account is suspended. Send your PIN to verify.",
]
for s in samples:
    print(f"\n'{s}'\n -> {predict_sms(s)}")
