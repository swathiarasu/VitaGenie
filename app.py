
import os
import pandas as pd
import streamlit as st
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn as nn

st.title("Virtual Health Coach")
mode = st.sidebar.selectbox("Choose Mode", ["Health Advice", "Heart Risk Prediction", "Symptom Checker"] )

# --------HEALTH ADVICE--------

class SBERTModel(nn.Module):
    def __init__(self):
        super(SBERTModel, self).__init__()
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

    def forward(self, input_texts):
        embeddings = self.sbert.encode(input_texts, convert_to_tensor=True)
        return embeddings
    
if mode == "Health Advice":
    # Load NLP assets
    if not os.path.exists('models/sbert_model.pkl'):
        st.error("NLP assets not found. Run `python main.py` to generate models.")
    else:
        model = SBERTModel()
        model.sbert.load_state_dict(torch.load('models/sbert_model_state_dict.pth', map_location='cpu'))
        model.sbert.eval()
        question_embeddings = np.load('models/question_embeddings.npy')
        questions = joblib.load('models/questions.pkl')
        answers = joblib.load('models/answers.pkl')
        query = st.text_input("Enter your health question:")
        if st.button("Get Advice"):
            if not query:
                st.warning("Please enter a question.")
            else:
                user_embedding = model.sbert.encode([query], convert_to_tensor=True,device='cpu')
                similarity_scores = util.pytorch_cos_sim(user_embedding, torch.tensor(question_embeddings))
                best_idx = torch.argmax(similarity_scores).item()
                st.markdown(f"**Advice:** {answers[best_idx]}")

# --------HEART RISK PREDICTION--------
elif mode == "Heart Risk Prediction":
    # Health Risk Prediction
    if not os.path.exists('models/health_risk_model.pkl'):
        st.error("Health risk model not found. Run `python main.py` to generate models.")
    else:
        st.subheader("Enter patient details for risk prediction:")
        # Load raw data columns for UI
        df_raw = pd.read_csv('data/health_risk_data.csv')
        df_raw.columns = [c.strip().lower().replace(' ', '_') for c in df_raw.columns]
        raw_cols = [c for c in df_raw.columns if c != 'heart_disease']
        categorical_mappings = {
            'sex':                {0: 'Female', 1: 'Male'},
            'chest_pain_type':    {1: 'Typical angina', 2: 'Atypical angina',
                                3: 'Non-anginal pain', 4: 'Asymptomatic'},
            'fbs_over_120':       {0: 'No', 1: 'Yes'},
            'ekg_results':        {0: 'Normal', 1: 'ST-T wave abnormality',
                                2: 'Left ventricular hypertrophy'},
            'exercise_angina':    {0: 'No', 1: 'Yes'},
            'slope_of_st':        {1: 'Upsloping', 2: 'Flat', 3: 'Downsloping'},
            'num_vessels_fluro':  {0: '0', 1: '1', 2: '2', 3: '3'},
            'thallium':           {3: 'Normal', 6: 'Fixed defect', 7: 'Reversible defect'}
        }

        # === then replace your input loop with this ===
        user_input = {}
        for col in raw_cols:
            title = col.replace('_', ' ').title()
            if col in categorical_mappings:
                # show dropdown of readable labels
                opts = list(categorical_mappings[col].values())
                choice = st.selectbox(f"{title}", opts, key=col)
                # reverse‐map back to the original code
                inv_map = {v:k for k,v in categorical_mappings[col].items()}
                user_input[col] = inv_map[choice]
            else:
                # numeric (age, bp, cholesterol, max_hr, st_depression, etc.)
                median = float(df_raw[col].median())
                user_input[col] = st.number_input(f"{title}", value=median, key=col)
        
        if st.button("Predict Risk"):
            # Prepare input
            X = pd.DataFrame([user_input])
            # Encode
            X_enc = pd.get_dummies(X)
            features = joblib.load('models/health_risk_features.pkl')
            for feat in features:
                if feat not in X_enc.columns:
                    X_enc[feat] = 0
            X_enc = X_enc[features]
            # Predict
            model = joblib.load('models/health_risk_model.pkl')
            le = joblib.load('models/risk_label_encoder.pkl')
            prob = model.predict_proba(X_enc)[0][1]
            pred = le.inverse_transform(model.predict(X_enc))[0]
            st.write(f"Predicted Risk: **{pred}** (Probability: {prob:.2f})")
            st.line_chart([prob])
            
# --------SYMPTOM CHECKER--------
else:
    if not os.path.exists('models/triage_vectorizer.pkl'):
        st.error("Run `python main.py` first to train models.")
    else:
        st.subheader("Symptom Checker / Triage")
        vectorizer = joblib.load('models/triage_vectorizer.pkl')
        clf = joblib.load('models/triage_classifier.pkl')
        mlb = joblib.load('models/triage_label_binarizer.pkl')
        next_steps = {code: "Rest, monitor symptoms, and hydrate." for code in mlb.classes_}
        red_flags = {"chest pain", "shortness of breath", "slurred speech", "severe bleeding"}
        text = st.text_area("Describe your symptoms:")
        if st.button("Triage"):
            if not text:
                st.warning("Please describe your symptoms.")
            else:
                Xq = vectorizer.transform([text])
                y_prob = clf.predict_proba(Xq)[0]
                top_idxs = np.where(y_prob >= 0.3)[0]
                if len(top_idxs) == 0:
                    st.info("No conditions matched. Try adding more details.")
                else:
                    conds = mlb.classes_[top_idxs]
                    st.write("**Possible conditions:**", ", ".join(conds))
                    st.markdown("### Recommended Next Steps")
                    for cond in conds:
                        st.markdown(f"**{cond}**: {next_steps.get(cond)}")
                    if any(flag in text.lower() for flag in red_flags):
                        st.error("⚠️ Some symptoms indicate potential emergencies. Seek immediate medical attention!")
