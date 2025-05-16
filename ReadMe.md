# VitaGenie - Your Smart Health Companion

An AI-powered, trio-mode health companion that helps users:

1. **Health Advice (Ask Medical Questions)**
   – Free-text Q&A using SentenceBERT model trained on a Medical Q&A dataset.  
2. **Predict Heart Disease Risk**
   – Demographic & vitals input → XGBoost + SMOTE pipeline predicts heart disease probability.  
3. **Symptom Checker**
   – Multi-label classifier (TF-IDF + LogisticRegression) maps symptom descriptions → possible conditions, suggests next steps, and flags emergencies.

## Folder Structure

```
bash
├── app.py                         # Streamlit frontend
├── main.py                        # Training models for all 3 tasks
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker build configuration
├── entrypoints/
│   └── run_app.sh                 # Script to launch the Streamlit app
│   └── download_corpus.sh         # Script to download the dataset
├── data/
│   ├── health_risk_data.csv                # Downloaded dataset
|   ├── medical_qa.csv                      # Downloaded dataset
│   └── nlice/                              # Downloaded dataset
│       ├── conditions.json        
│       └── sample_data          
└── models/
    ├── answers.pkl
    ├── health_risk_features.pkl
    ├── health_risk_model.pkl
    ├── question_embeddings.npy
    ├── questions.pkl
    ├── risk_label_encoder.pkl
    ├── sbert_model.pkl
    ├── triage_classifier.pkl
    ├── triage_label_binarizer.pkl
    └── triage_vectorizer.pkl
└── README.md                      # Project overview and setup instructions
```

## Steps to run this project

1. To spawn a container
   > docker build -t rag-app .

2. To downloading the corpus
   > GPU: `docker run --gpus all -v ./.kaggle:/root/.kaggle --entrypoint bash rag-app entrypoints/download_corpus.sh`
   >
   > CPU: `docker run -v ./.kaggle:/root/.kaggle --entrypoint bash rag-app entrypoints/download_corpus.sh`

3. To get the vector embeddings of the corpus
   > GPU: `docker run --gpus all --entrypoint bash rag-app entrypoints/build_index.sh`
   >
   > CPU: `docker run --entrypoint bash rag-app entrypoints/build_index.sh`

4. To run the app
   > GPU: `docker run -d --gpus all -p 8501:8501 rag-app`
   >
   > CPU: `docker run -d -p 8501:8501 rag-app`

5. To access the app, open your browser and go to: `http://<external-ip>:8501`




