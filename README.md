# Spam SMS Detection (CodSoft ML Internship - Task 4)

Classifies SMS messages as spam or legitimate (ham).

## Dataset
SMS Spam Collection dataset (`spam.csv`) with 5,000+ labelled SMS messages (ham = legitimate, spam = unwanted).

## Approach
- Preprocessing: duplicate removal, lowercase, stop-word removal
- Features: TF-IDF (unigrams + bigrams)
- Models: Naive Bayes, Logistic Regression, Linear SVM
- Evaluation: accuracy, precision, recall, F1, confusion matrix

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes | 0.9691 | 1.0000 | 0.7557 | 0.8609 |
| Logistic Regression | 0.9739 | 0.9062 | 0.8855 | 0.8958 |
| Linear SVM | 0.9816 | 0.9516 | 0.9008 | 0.9255 |

Best model: **Linear SVM** (98.16% accuracy, F1 = 0.9255)


## Limitations
The dataset mostly contains prize/offer style spam, so phishing-style messages (e.g. fake bank alerts) may be missed. Training on newer data would improve this.

## How to run
```bash
pip install -r requirements.txt
python spam_detection.py
```

## Files
- `spam_detection.py` - main code
- `requirements.txt` - libraries needed
- `spam.csv` - dataset
- `confusion_matrix.png` - generated after running"# CODSOFT_TASK4" 
