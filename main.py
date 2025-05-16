import os
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import glob
import re
import torch


##-----ADVICE MODEL TRAINING-----

def train_advice_model():
    qa_path = os.path.join('data', 'medical_qa.csv')
    if not os.path.exists(qa_path):
        print(f" Medical Q&A dataset not found at {qa_path}. Skipping advice training.")
        return

    df = pd.read_csv(qa_path)
    # Ensure required columns
    df.columns = [c.strip().lower() for c in df.columns]
    if 'question' not in df.columns or 'answer' not in df.columns:
        print(" Columns 'question' and 'answer' required in medical_qa.csv. Skipping advice model training.")
        return

    questions = df['question'].astype(str).tolist()
    answers = df['answer'].astype(str).tolist()

    # Load Sentence-BERT Model (Pre-trained)
    print("🔧Loading Sentence-BERT model...")
    sbert = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')  # Set to CPU for portability

    # Encode questions with progress bar
    print(" Encoding questions with Sentence-BERT...")
    question_embeddings = sbert.encode(questions, show_progress_bar=True, convert_to_tensor=True)

    # Save assets in a portable, CPU-compatible format
    os.makedirs('models', exist_ok=True)

    # Save the Sentence-BERT model as state_dict
    model_save_path = 'models/sbert_model_state_dict.pth'
    torch.save(sbert.state_dict(), model_save_path)
    print(f" Model state dictionary saved: {model_save_path}")

    # Save the question embeddings and text data
    np.save('models/question_embeddings.npy', question_embeddings.cpu().numpy())
    joblib.dump(questions, 'models/questions.pkl')
    joblib.dump(answers, 'models/answers.pkl')
    print(" Saved advice assets: question_embeddings.npy, questions.pkl, answers.pkl")

##-----HEART RISK MODEL TRAINING-----

def train_heart_risk_model():
    risk_path = os.path.join('data', 'health_risk_data.csv')
    if not os.path.exists(risk_path):
        print(f"Health risk dataset not found at {risk_path}. Skipping risk model training.")
        return
    df = pd.read_csv(risk_path)
    # Preprocess columns
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    # Encode target
    if 'heart_disease' not in df.columns:
        print("Column 'heart_disease' required in health_risk_data.csv. Skipping risk model training.")
        return
    le = LabelEncoder()
    df['heart_disease'] = le.fit_transform(df['heart_disease'])
    y = df['heart_disease']
    X = df.drop('heart_disease', axis=1)
    # One-hot encode categoricals
    X_encoded = pd.get_dummies(X)
    # SMOTE for imbalance
    print("Applying SMOTE to balance classes...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_encoded, y)
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    # XGBoost with GridSearch
    print("Training Health Risk model with XGBoost and GridSearch...")
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.1, 0.01]
    }
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    grid = GridSearchCV(xgb_clf, param_grid, cv=3, scoring='accuracy', verbose=1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    # Save model and encoders
    os.makedirs('models', exist_ok=True)
    joblib.dump(best, 'models/health_risk_model.pkl')
    joblib.dump(X_encoded.columns.tolist(), 'models/health_risk_features.pkl')
    joblib.dump(le, 'models/risk_label_encoder.pkl')
    print(f"Saved Heart Risk model: health_risk_model.pkl (best params: {grid.best_params_})")

##-----SYMPTOM CHECKER TRAINING-----

def train_symptom_checker():
    nlice_dir = os.path.join('data', 'nlice/sample_data')
    if not os.path.exists(nlice_dir):
        print(f"NLICE data not found at {nlice_dir}. Skipping symptm checker model training.")
        return

    records = []
    for csv_file in glob.glob(os.path.join(nlice_dir, '*.csv')):
        df = pd.read_csv(csv_file)
        if 'SYMPTOMS' in df.columns and 'PATHOLOGY' in df.columns:
            for _, row in df.iterrows():
                symptoms = str(row['SYMPTOMS']).split(';')
                condition = row['PATHOLOGY'].strip()
                for sym in symptoms:
                    records.append((sym.lower().strip(), condition))

    if not records:
        print("No symptom records found. Skipping symptm checker model training.")
        return

    # Prepare data
    symptoms, conditions = zip(*records)
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform([[cond] for cond in conditions])

    # TF-IDF + Logistic Regression
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(symptoms)

    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    clf.fit(X, Y)

    # Save models
    joblib.dump(vectorizer, 'models/triage_vectorizer.pkl')
    joblib.dump(clf, 'models/triage_classifier.pkl')
    joblib.dump(mlb, 'models/triage_label_binarizer.pkl')

    print("symptm checker model trained and saved.")

if __name__ == '__main__':
    print("Starting training for Virtual Health Coach...")
    train_advice_model()
    train_heart_risk_model()
    train_symptom_checker()
