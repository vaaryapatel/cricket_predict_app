import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

import statsmodels.api as sm

st.set_page_config(page_title="Song Virality Predictor", layout="wide")

st.title("🎵 Song Viral Potential Predictor (Spotify + TikTok)")

# ==============================
# FILE UPLOAD
# ==============================
st.sidebar.header("Upload Data")

spotify_file = st.sidebar.file_uploader("Upload Spotify Dataset", type=["csv"])
tiktok_file = st.sidebar.file_uploader("Upload TikTok Dataset", type=["csv"])

if spotify_file and tiktok_file:

    spotify = pd.read_csv(spotify_file)
    tiktok = pd.read_csv(tiktok_file)

    st.success("Datasets uploaded successfully!")

    # ==============================
    # COLUMN VALIDATION
    # ==============================
    required_cols = [
        'track_name',
        'track_popularity',
        'artist_popularity',
        'artist_followers',
        'album_total_tracks',
        'track_number',
        'track_duration_ms'
    ]

    missing = [col for col in required_cols if col not in spotify.columns]

    if missing:
        st.error(f"Missing columns in Spotify dataset: {missing}")
        st.stop()

    # ==============================
    # PREPROCESSING
    # ==============================
    spotify['track_name'] = spotify['track_name'].astype(str).str.lower()
    tiktok['track_name'] = tiktok['track_name'].astype(str).str.lower()

    spotify['viral'] = spotify['track_name'].isin(tiktok['track_name']).astype(int)

    spotify['track_duration_min'] = spotify['track_duration_ms'] / 60000

    features = [
        'track_popularity',
        'artist_popularity',
        'artist_followers',
        'album_total_tracks',
        'track_number',
        'track_duration_min'
    ]

    X = spotify[features].dropna()
    y = spotify.loc[X.index, 'viral']

    if X.empty:
        st.error("No data left after cleaning. Check your dataset.")
        st.stop()

    # ==============================
    # TRAIN MODEL
    # ==============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ==============================
    # METRICS
    # ==============================
    st.subheader("📊 Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        st.metric("ROC-AUC", f"{roc_auc_score(y_test, y_prob):.4f}")

    with col2:
        st.write("Confusion Matrix")
        st.write(confusion_matrix(y_test, y_pred))

    st.write("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # ==============================
    # HYPOTHESIS TESTING (SAFE)
    # ==============================
    st.subheader("🧪 Hypothesis Testing (Feature Significance)")

    try:
        X_sm = sm.add_constant(X)
        logit_model = sm.Logit(y, X_sm)
        result = logit_model.fit(disp=0)

        summary_df = pd.DataFrame({
            "Feature": result.params.index,
            "Coefficient": result.params.values,
            "p-value": result.pvalues.values
        })

        st.dataframe(summary_df)

    except Exception as e:
        st.error("Statsmodels failed to run. Possible reasons: perfect separation or data issues.")
        st.text(str(e))

    # ==============================
    # INTERACTIVE PREDICTION
    # ==============================
    st.subheader("🎯 Predict Virality for a New Song")

    user_input = {}

    for feature in features:
        user_input[feature] = st.number_input(
            f"{feature}",
            value=float(X[feature].mean()),
            step=0.1
        )

    input_df = pd.DataFrame([user_input])

    if st.button("Predict"):
        try:
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            if prediction == 1:
                st.success(f"🔥 This song is likely to go VIRAL! (Probability: {probability:.2f})")
            else:
                st.warning(f"❌ This song is NOT likely to go viral (Probability: {probability:.2f})")

        except Exception as e:
            st.error("Prediction failed. Check input values.")
            st.text(str(e))

else:
    st.info("Please upload both datasets to proceed.")