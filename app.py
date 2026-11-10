import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AgriTrust AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE - PREDICTION HISTORY
# ============================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
}

.hero {
    padding: 30px;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        #0f5132 0%,
        #198754 100%
    );
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 17px;
}

.card {
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #d9dee3;
    background-color: #ffffff;
    min-height: 150px;
}

.card-title {
    font-size: 15px;
    color: #6c757d;
    margin-bottom: 8px;
}

.card-value {
    font-size: 30px;
    font-weight: 700;
}

.card-subtitle {
    font-size: 14px;
    color: #6c757d;
}

.reliable {
    padding: 20px;
    border-radius: 15px;
    background-color: #d1e7dd;
    border: 1px solid #a3cfbb;
}

.unreliable {
    padding: 20px;
    border-radius: 15px;
    background-color: #f8d7da;
    border: 1px solid #f1aeb5;
}

.info-box {
    padding: 18px;
    border-radius: 12px;
    background-color: #e9f2ff;
    border: 1px solid #b6d4fe;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<h1>🌱 AgriTrust AI</h1>

<p>
AI-Powered Crop Identification, Disease Detection
and Prediction Reliability Analysis
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🌿 AgriTrust AI")

    st.markdown("""
    ### System Pipeline

    1. 📷 Upload Leaf Image
    2. 🧠 MobileNetV2
    3. 🌾 Crop Detection
    4. 🩺 Condition Detection
    5. 🤖 Model Comparison
    6. 🛡️ Reliability Detection
    """)

    st.divider()

    st.info(
        "AgriTrust AI predicts the crop and condition "
        "and estimates whether the condition prediction "
        "may be unreliable."
    )


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    cnn = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3)
    )

    crop_dt = joblib.load(
        "crop_decision_tree_model.pkl"
    )

    crop_rf = joblib.load(
        "crop_random_forest_model.pkl"
    )

    crop_labels = joblib.load(
        "crop_labels.pkl"
    )

    disease_dt = joblib.load(
        "decision_tree_model.pkl"
    )

    disease_rf = joblib.load(
        "random_forest_model.pkl"
    )

    reliability_rf = joblib.load(
        "improved_reliability_random_forest.pkl"
    )

    return (
        cnn,
        crop_dt,
        crop_rf,
        crop_labels,
        disease_dt,
        disease_rf,
        reliability_rf
    )


try:

    (
        cnn,
        crop_dt_model,
        crop_rf_model,
        crop_labels,
        disease_dt_model,
        disease_rf_model,
        reliability_model
    ) = load_models()

except Exception as e:

    st.error("Model loading failed.")
    st.code(str(e))
    st.stop()


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📷 Upload Leaf Image'
    '</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a crop leaf image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    uploaded_image = Image.open(
        uploaded_file
    ).convert("RGB")

    col1, col2 = st.columns(
        [1, 1.8]
    )

    with col1:

        st.image(
            uploaded_image,
            caption="Uploaded Leaf Image",
            use_container_width=True
        )

    with col2:

        st.markdown(
            '<div class="section-title">'
            '🔬 AI Analysis'
            '</div>',
            unsafe_allow_html=True
        )

        predict_button = st.button(
            "🚀 Analyze Leaf",
            use_container_width=True,
            type="primary"
        )

    # ========================================================
    # BUTTON CLICK
    # ========================================================

    if predict_button:

        with st.spinner(
            "Analyzing leaf image..."
        ):

            # ------------------------------------------------
            # SAVE TEMP IMAGE
            # ------------------------------------------------

            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg"
            )

            uploaded_image.save(
                temp_file.name
            )

            image_path = temp_file.name

            # ------------------------------------------------
            # PREPROCESS IMAGE
            # ------------------------------------------------

            img = image.load_img(
                image_path,
                target_size=(224, 224)
            )

            img_array = image.img_to_array(
                img
            )

            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            img_array = preprocess_input(
                img_array
            )

            # ------------------------------------------------
            # FEATURE EXTRACTION
            # ------------------------------------------------

            features = cnn.predict(
                img_array,
                verbose=0
            )

            # =================================================
            # CROP PREDICTION
            # =================================================

            crop_dt_prediction = (
                crop_dt_model.predict(
                    features
                )[0]
            )

            crop_rf_prediction = (
                crop_rf_model.predict(
                    features
                )[0]
            )

            crop_dt_prob = (
                crop_dt_model.predict_proba(
                    features
                )[0]
            )

            crop_rf_prob = (
                crop_rf_model.predict_proba(
                    features
                )[0]
            )

            crop_dt_confidence = (
                np.max(crop_dt_prob) * 100
            )

            crop_rf_confidence = (
                np.max(crop_rf_prob) * 100
            )

            crop_dt_name = [
                name
                for name, value
                in crop_labels.items()
                if value == crop_dt_prediction
            ][0]

            crop_rf_name = [
                name
                for name, value
                in crop_labels.items()
                if value == crop_rf_prediction
            ][0]

            crop_agreement = (
                crop_dt_prediction ==
                crop_rf_prediction
            )

            final_crop = crop_rf_name

            final_crop_confidence = (
                crop_rf_confidence
            )

            # =================================================
            # CONDITION PREDICTION
            # =================================================

            dt_prediction = (
                disease_dt_model.predict(
                    features
                )[0]
            )

            rf_prediction = (
                disease_rf_model.predict(
                    features
                )[0]
            )

            dt_prob = (
                disease_dt_model.predict_proba(
                    features
                )[0]
            )

            rf_prob = (
                disease_rf_model.predict_proba(
                    features
                )[0]
            )

            dt_confidence = (
                np.max(dt_prob) * 100
            )

            rf_confidence = (
                np.max(rf_prob) * 100
            )

            dt_disease_probability = (
                dt_prob[1]
            )

            rf_disease_probability = (
                rf_prob[1]
            )

            dt_condition = (
                "Disease"
                if dt_prediction == 1
                else "Healthy"
            )

            rf_condition = (
                "Disease"
                if rf_prediction == 1
                else "Healthy"
            )

            model_agreement = (
                dt_prediction ==
                rf_prediction
            )

            # =================================================
            # RELIABILITY FEATURES
            # =================================================

            def calculate_entropy(probabilities):

                probabilities = np.clip(
                    probabilities,
                    1e-10,
                    1
                )

                return -np.sum(
                    probabilities *
                    np.log2(probabilities)
                )

            dt_confidence_gap = abs(
                2 * dt_disease_probability - 1
            )

            rf_confidence_gap = abs(
                2 * rf_disease_probability - 1
            )

            dt_entropy = calculate_entropy(
                dt_prob
            )

            rf_entropy = calculate_entropy(
                rf_prob
            )

            confidence_difference = (
                abs(
                    dt_confidence -
                    rf_confidence
                ) / 100
            )

            probability_difference = abs(
                dt_disease_probability -
                rf_disease_probability
            )

            agreement_value = (
                1
                if model_agreement
                else 0
            )

            # =================================================
            # RELIABILITY PREDICTION
            # =================================================

            reliability_features = np.array([[
                dt_prediction,
                dt_confidence / 100,
                dt_disease_probability,
                dt_confidence_gap,
                dt_entropy,

                rf_prediction,
                rf_confidence / 100,
                rf_disease_probability,
                rf_confidence_gap,
                rf_entropy,

                confidence_difference,
                probability_difference,
                agreement_value
            ]])

            reliability_prediction = (
                reliability_model.predict(
                    reliability_features
                )[0]
            )

            reliability_prob = (
                reliability_model.predict_proba(
                    reliability_features
                )[0]
            )

            reliability_confidence = (
                np.max(
                    reliability_prob
                ) * 100
            )

            if reliability_prediction == 1:

                reliability_status = (
                    "RELIABLE"
                )

            else:

                reliability_status = (
                    "POTENTIALLY UNRELIABLE"
                )

            # =================================================
            # SAVE PREDICTION HISTORY
            # =================================================

            prediction_record = {
                "Crop": final_crop.title(),
                "Crop Confidence":
                    f"{final_crop_confidence:.2f}%",
                "Condition": rf_condition,
                "Condition Confidence":
                    f"{rf_confidence:.2f}%",
                "DT Prediction":
                    dt_condition,
                "RF Prediction":
                    rf_condition,
                "Model Agreement":
                    (
                        "YES"
                        if model_agreement
                        else "NO"
                    ),
                "AI Reliability":
                    reliability_status,
                "Reliability Confidence":
                    f"{reliability_confidence:.2f}%"
            }

            st.session_state.prediction_history.append(
                prediction_record
            )

        # =====================================================
        # PREDICTION RESULTS
        # =====================================================

        st.divider()

        st.markdown(
            '<div class="section-title">'
            '📊 Prediction Results'
            '</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.markdown(
                f"""
                <div class="card">

                <div class="card-title">
                🌾 DETECTED CROP
                </div>

                <div class="card-value">
                {final_crop.title()}
                </div>

                <div class="card-subtitle">
                Confidence:
                {final_crop_confidence:.2f}%
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:

            st.markdown(
                f"""
                <div class="card">

                <div class="card-title">
                🩺 CONDITION
                </div>

                <div class="card-value">
                {rf_condition}
                </div>

                <div class="card-subtitle">
                Confidence:
                {rf_confidence:.2f}%
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:

            agreement_text = (
                "YES"
                if model_agreement
                else "NO"
            )

            st.markdown(
                f"""
                <div class="card">

                <div class="card-title">
                🤖 MODEL AGREEMENT
                </div>

                <div class="card-value">
                {agreement_text}
                </div>

                <div class="card-subtitle">
                DT vs Random Forest
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with c4:

            st.markdown(
                f"""
                <div class="card">

                <div class="card-title">
                🛡️ AI RELIABILITY
                </div>

                <div class="card-value">
                {reliability_status}
                </div>

                <div class="card-subtitle">
                Confidence:
                {reliability_confidence:.2f}%
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        # =====================================================
        # CONFIDENCE ANALYSIS
        # =====================================================

        st.markdown(
            '<div class="section-title">'
            '📈 Confidence Analysis'
            '</div>',
            unsafe_allow_html=True
        )

        b1, b2 = st.columns(2)

        with b1:

            st.write(
                f"🌾 **Crop Confidence — "
                f"{final_crop_confidence:.2f}%**"
            )

            st.progress(
                min(
                    int(final_crop_confidence),
                    100
                )
            )

            st.write(
                f"🩺 **Condition Confidence — "
                f"{rf_confidence:.2f}%**"
            )

            st.progress(
                min(
                    int(rf_confidence),
                    100
                )
            )

        with b2:

            st.write(
                f"🌳 **Decision Tree — "
                f"{dt_confidence:.2f}%**"
            )

            st.progress(
                min(
                    int(dt_confidence),
                    100
                )
            )

            st.write(
                f"🌲 **Random Forest — "
                f"{rf_confidence:.2f}%**"
            )

            st.progress(
                min(
                    int(rf_confidence),
                    100
                )
            )

        # =====================================================
        # AI SELF RELIABILITY
        # =====================================================

        st.markdown(
            '<div class="section-title">'
            '🛡️ AI Self-Reliability'
            '</div>',
            unsafe_allow_html=True
        )

        if reliability_prediction == 1:

            st.markdown(
                f"""
                <div class="reliable">

                <h2>✅ Prediction is RELIABLE</h2>

                <p>
                The reliability model estimates that the
                condition prediction is likely to be correct.
                </p>

                <b>
                Reliability Confidence:
                {reliability_confidence:.2f}%
                </b>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="unreliable">

                <h2>
                ⚠️ Potentially Unreliable Prediction
                </h2>

                <p>
                The reliability model detected signals that
                the condition prediction may be incorrect.
                </p>

                <b>
                Reliability Confidence:
                {reliability_confidence:.2f}%
                </b>

                </div>
                """,
                unsafe_allow_html=True
            )

        # =====================================================
        # MODEL DETAILS
        # =====================================================

        st.markdown(
            '<div class="section-title">'
            '🔍 Model Details'
            '</div>',
            unsafe_allow_html=True
        )

        d1, d2 = st.columns(2)

        with d1:

            st.subheader(
                "🌾 Crop Classification"
            )

            st.write(
                f"**Decision Tree:** "
                f"{crop_dt_name.title()} "
                f"({crop_dt_confidence:.2f}%)"
            )

            st.write(
                f"**Random Forest:** "
                f"{crop_rf_name.title()} "
                f"({crop_rf_confidence:.2f}%)"
            )

            st.write(
                "**Agreement:** "
                + (
                    "YES"
                    if crop_agreement
                    else "NO"
                )
            )

        with d2:

            st.subheader(
                "🩺 Condition Classification"
            )

            st.write(
                f"**Decision Tree:** "
                f"{dt_condition} "
                f"({dt_confidence:.2f}%)"
            )

            st.write(
                f"**Random Forest:** "
                f"{rf_condition} "
                f"({rf_confidence:.2f}%)"
            )

            st.write(
                "**Agreement:** "
                + (
                    "YES"
                    if model_agreement
                    else "NO"
                )
            )

        # =====================================================
        # OVERALL MODEL PERFORMANCE
        # =====================================================

        st.markdown(
            '<div class="section-title">'
            '📊 Overall Model Performance'
            '</div>',
            unsafe_allow_html=True
        )

        st.write(
            "Performance measured on the held-out test dataset."
        )

        p1, p2 = st.columns(2)

        with p1:

            st.subheader(
                "🌾 Crop Classification"
            )

            st.metric(
                "Decision Tree",
                "90.20%"
            )

            st.progress(90)

            st.metric(
                "Random Forest",
                "98.90%"
            )

            st.progress(98)

        with p2:

            st.subheader(
                "🩺 Condition Classification"
            )

            st.metric(
                "Decision Tree",
                "87.44%"
            )

            st.progress(87)

            st.metric(
                "Random Forest",
                "92.84%"
            )

            st.progress(92)

        # =====================================================
        # RELIABILITY PERFORMANCE
        # =====================================================

        st.markdown(
            '<div class="section-title">'
            '🛡️ Reliability Model Performance'
            '</div>',
            unsafe_allow_html=True
        )

        r1, r2, r3 = st.columns(3)

        with r1:

            st.metric(
                "Reliability Accuracy",
                "81.83%"
            )

        with r2:

            st.metric(
                "Incorrect Condition Predictions Detected",
                "76.9%"
            )

        with r3:

            st.metric(
                "Reliability Model",
                "Random Forest"
            )

        st.info(
            "The reliability model detected approximately "
            "76.9% of incorrect condition predictions in "
            "the test set."
        )

        # =====================================================
        # PREDICTION HISTORY
        # =====================================================

        st.markdown(
            '<div class="section-title">'
            '🕘 Prediction History'
            '</div>',
            unsafe_allow_html=True
        )

        if len(st.session_state.prediction_history) > 0:

            history_df = pd.DataFrame(
                st.session_state.prediction_history
            )

            # Compact table for dashboard
            display_df = history_df[
                [
                    "Crop",
                    "Crop Confidence",
                    "Condition",
                    "Condition Confidence",
                    "Model Agreement",
                    "AI Reliability",
                    "Reliability Confidence"
                ]
            ]

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

            # -------------------------------------------------
            # DOWNLOAD
            # -------------------------------------------------

            csv_data = history_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="📥 Download Prediction History",
                data=csv_data,
                file_name="agritrust_prediction_history.csv",
                mime="text/csv",
                use_container_width=True
            )

            # -------------------------------------------------
            # CLEAR HISTORY
            # -------------------------------------------------

            if st.button(
                    "🗑️ Clear Prediction History"
            ):
                st.session_state.prediction_history = []

                st.rerun()

        else:

            st.info(
                "No predictions have been recorded yet."
            )

        # =====================================================
        # VISUAL ANALYTICS
        # =====================================================

        st.markdown(
            '<div class="section-title">📊 Visual Analytics</div>',
            unsafe_allow_html=True
        )

        history = st.session_state.get("prediction_history", [])

        if history:

            st.subheader("📊 Prediction Analytics")

            # Convert history to DataFrame
            analytics_df = pd.DataFrame(history)

            # Show available data
            st.write("Total Predictions:", len(analytics_df))

            # -----------------------------------------------
            # SUMMARY
            # -----------------------------------------------

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "🔢 Total Predictions",
                    len(analytics_df)
                )

            with col2:
                if "Crop" in analytics_df.columns:
                    most_common_crop = (
                        analytics_df["Crop"]
                        .value_counts()
                        .idxmax()
                    )
                    st.metric(
                        "🌾 Most Predicted Crop",
                        most_common_crop
                    )

            with col3:
                if "Condition Confidence" in analytics_df.columns:
                    confidence = (
                        analytics_df["Condition Confidence"]
                        .astype(str)
                        .str.replace("%", "", regex=False)
                        .astype(float)
                    )

                    st.metric(
                        "📈 Avg Condition Confidence",
                        f"{confidence.mean():.2f}%"
                    )

            with col4:
                if "AI Reliability" in analytics_df.columns:
                    reliable = (
                        analytics_df["AI Reliability"]
                        .astype(str)
                        .str.upper()
                        .eq("RELIABLE")
                        .sum()
                    )

                    st.metric(
                        "🛡️ Reliable Predictions",
                        reliable
                    )

            st.divider()

            # -----------------------------------------------
            # CROP DISTRIBUTION
            # -----------------------------------------------

            if "Crop" in analytics_df.columns:
                st.subheader("🌾 Crop Prediction Distribution")

                crop_counts = analytics_df["Crop"].value_counts()

                st.bar_chart(crop_counts)

            # -----------------------------------------------
            # CONDITION DISTRIBUTION
            # -----------------------------------------------

            if "Condition" in analytics_df.columns:
                st.subheader("🩺 Condition Distribution")

                condition_counts = (
                    analytics_df["Condition"].value_counts()
                )

                st.bar_chart(condition_counts)

            # -----------------------------------------------
            # RELIABILITY DISTRIBUTION
            # -----------------------------------------------

            if "AI Reliability" in analytics_df.columns:
                st.subheader("🛡️ AI Reliability Distribution")

                reliability_counts = (
                    analytics_df["AI Reliability"].value_counts()
                )

                st.bar_chart(reliability_counts)

            # -----------------------------------------------
            # CONFIDENCE ANALYSIS
            # -----------------------------------------------

            confidence_columns = []

            if "Crop Confidence" in analytics_df.columns:
                analytics_df["Crop Confidence"] = (
                    analytics_df["Crop Confidence"]
                    .astype(str)
                    .str.replace("%", "", regex=False)
                    .astype(float)
                )

                confidence_columns.append(
                    "Crop Confidence"
                )

            if "Condition Confidence" in analytics_df.columns:
                analytics_df["Condition Confidence"] = (
                    analytics_df["Condition Confidence"]
                    .astype(str)
                    .str.replace("%", "", regex=False)
                    .astype(float)
                )

                confidence_columns.append(
                    "Condition Confidence"
                )

            if "Reliability Confidence" in analytics_df.columns:
                analytics_df["Reliability Confidence"] = (
                    analytics_df["Reliability Confidence"]
                    .astype(str)
                    .str.replace("%", "", regex=False)
                    .astype(float)
                )

                confidence_columns.append(
                    "Reliability Confidence"
                )

            if confidence_columns:
                st.subheader("📈 Confidence Analysis")

                st.line_chart(
                    analytics_df[confidence_columns]
                )

        else:

            st.info(
                "No prediction history available. "
                "Run a prediction first."
            )

        # =====================================================
        # PROJECT INFORMATION
        # =====================================================

        st.markdown(
            '<div class="section-title">'
            'ℹ️ About AgriTrust AI'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-box">

            <b>AgriTrust AI</b> is an AI-based crop image
            analysis system.

            <br><br>

            It identifies the crop, predicts whether the leaf
            is Healthy or Diseased, compares Decision Tree and
            Random Forest predictions, and uses a separate
            reliability model to estimate whether the condition
            prediction may be unreliable.

            </div>
            """,
            unsafe_allow_html=True
        )

        # -----------------------------------------------------
        # REMOVE TEMPORARY FILE
        # -----------------------------------------------------

        try:
            os.remove(image_path)
        except:
            pass


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.markdown(
        """
        <div class="info-box">

        <h3>👆 Upload a leaf image to begin</h3>

        <p>
        Upload a JPG, JPEG, PNG, or WEBP image of a crop leaf.
        AgriTrust AI will identify the crop, predict its
        condition, compare multiple ML models, and estimate
        the reliability of the condition prediction.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )