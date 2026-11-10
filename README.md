# 🌱 AgriTrust AI

## AI-Based Crop Disease Prediction and Self-Reliability Detection

AgriTrust AI is a machine learning system that analyzes crop leaf images to predict the **crop type** and **crop condition (Healthy or Disease)**. Unlike a conventional prediction system, AgriTrust AI also estimates whether its own prediction is **reliable or potentially unreliable**.

The system combines **MobileNetV2 feature extraction, Decision Tree, Random Forest, and reliability analysis** into an interactive Streamlit dashboard.

---

## 🚀 Key Features

- 🌾 Automatic crop identification
- 🩺 Healthy/Disease prediction
- 🌳 Decision Tree prediction
- 🌲 Random Forest prediction
- 🤝 Model agreement analysis
- 🛡️ AI self-reliability detection
- 📊 Prediction confidence scores
- 📈 Visual analytics dashboard
- 📝 Prediction history
- 📥 Prediction history CSV export
- 🖥️ Interactive Streamlit interface

---

## 🎯 Project Objective

The main objective of AgriTrust AI is to build an AI system that does not only make predictions but also estimates **when its prediction may be unreliable**.

### Research Question

> Can an AI system identify situations where its own prediction may be incorrect?

---

## 🧠 System Workflow

```text
                 Input Crop Image
                        │
                        ▼
                MobileNetV2
             Feature Extraction
                        │
                        ▼
              ┌─────────────────┐
              │ Crop Prediction │
              │  DT + Random    │
              │     Forest      │
              └────────┬────────┘
                       │
                       ▼
             Crop Type + Confidence
                       │
                       ▼
             Disease Prediction
                DT + Random Forest
                       │
                       ▼
          Healthy / Disease Prediction
                       │
                       ▼
             Model Agreement
                       │
                       ▼
          Reliability Feature Analysis
                DT + Random Forest
                       │
                       ▼
           Reliable / Potentially
                Unreliable
                       │
                       ▼
             Streamlit Dashboard
```

---

## 📊 Dataset

The project uses the **Indian Crop Disease Dataset** from Kaggle.

### Dataset Categories

The dataset contains six crop categories:

- Coffee
- Cotton
- Jute
- Rice
- Sugarcane
- Wheat

The dataset contains **6,048 images** in total.

| Crop | Images |
|---|---:|
| Coffee | 652 |
| Cotton | 237 |
| Jute | 920 |
| Rice | 1,311 |
| Sugarcane | 2,521 |
| Wheat | 407 |
| **Total** | **6,048** |

The dataset provides **Healthy** and **Disease** conditions within the crop categories.

> Note: The dataset does not provide individual disease names. Therefore, AgriTrust AI predicts **Healthy vs Disease**, rather than a specific disease name.

---

## 🔬 Data Processing

The dataset was divided into:

- **70% Training**
- **15% Validation**
- **15% Testing**

### Dataset Split

| Dataset | Images |
|---|---:|
| Training | 4,233 |
| Validation | 907 |
| Testing | 908 |
| **Total** | **6,048** |

---

## 🧩 Feature Extraction

AgriTrust AI uses **MobileNetV2** as a feature extractor.

### Configuration

```text
Model: MobileNetV2
Weights: ImageNet
Input Size: 224 × 224 × 3
Pooling: Global Average Pooling
Feature Size: 1,280
```

The extracted 1,280-dimensional features are used as input for the machine learning models.

---

# 🤖 Machine Learning Models

## 1. Crop Classification

The system automatically identifies the crop from the uploaded image.

### Decision Tree

**Accuracy: 90.20%**

### Random Forest

**Accuracy: 98.90%**

Random Forest achieved the best crop classification performance.

---

## 2. Crop Condition Prediction

The system predicts whether the crop image is:

```text
Healthy
or
Disease
```

### Decision Tree

**Accuracy: 87.44%**

### Random Forest

**Accuracy: 92.84%**

Random Forest performed better than Decision Tree for crop condition prediction.

---

## 3. AI Reliability Detection

The reliability module estimates whether the first-stage condition prediction is likely to be correct.

The reliability model uses information such as:

- Decision Tree prediction
- Random Forest prediction
- Prediction confidence
- Disease probability
- Confidence gap
- Prediction entropy
- Confidence difference
- Probability difference
- Model agreement

### Reliability Random Forest

**Accuracy: 81.83%**

More importantly, the improved reliability model detected approximately:

**76.9% of incorrect condition predictions.**

This makes the reliability model useful as an additional warning mechanism rather than treating every model prediction as equally trustworthy.

---

# 📈 Model Performance

| Task | Model | Accuracy |
|---|---|---:|
| Crop Classification | Decision Tree | 90.20% |
| Crop Classification | Random Forest | **98.90%** |
| Condition Prediction | Decision Tree | 87.44% |
| Condition Prediction | Random Forest | **92.84%** |
| Reliability Detection | Random Forest | 81.83% |

### Best Models

```text
Crop Prediction
      ↓
Random Forest
      ↓
98.90%

Condition Prediction
      ↓
Random Forest
      ↓
92.84%

Reliability Detection
      ↓
Improved Random Forest
      ↓
81.83%
```

---

# 🛡️ AI Self-Reliability Concept

The main research contribution of AgriTrust AI is the **self-reliability layer**.

Instead of only displaying:

```text
Prediction: Disease
Confidence: 92%
```

the system also evaluates:

```text
AI Reliability: RELIABLE
Reliability Confidence: 99%
```

If the reliability model identifies a potentially uncertain prediction, the system can indicate:

```text
AI Reliability: POTENTIALLY UNRELIABLE
```

This provides an additional layer of transparency for the prediction.

> Reliability is an estimate based on model behavior. It does not guarantee that a prediction is correct.

---

# 🖥️ Streamlit Dashboard

AgriTrust AI provides an interactive Streamlit dashboard containing:

### Prediction Dashboard

- Uploaded crop image
- Predicted crop
- Crop confidence
- Predicted condition
- Condition confidence
- Decision Tree result
- Random Forest result
- Model agreement
- AI reliability
- Reliability confidence

### Visual Analytics

- Total predictions
- Most predicted crop
- Average condition confidence
- Reliable prediction count
- Crop prediction distribution
- Condition distribution
- AI reliability distribution
- Confidence analysis

### Prediction History

The dashboard also maintains prediction history and provides CSV export functionality.

---

# 📁 Project Structure

```text
AgriTrust_AI/
│
├── app.py
├── agritrust_predict.py
│
├── preprocessing.py
├── data_split.py
├── feature_extraction.py
│
├── crop_classification.py
├── train_prediction_models.py
├── train_reliability_models.py
├── train_improved_reliability_models.py
├── build_reliability_dataset.py
│
├── dataset_analysis.py
├── dataset_inspection.py
├── main.py
│
├── crop_decision_tree_model.pkl
├── crop_random_forest_model.pkl
├── crop_labels.pkl
│
├── decision_tree_model.pkl
├── random_forest_model.pkl
│
├── reliability_decision_tree.pkl
├── reliability_random_forest.pkl
│
├── improved_reliability_decision_tree.pkl
├── improved_reliability_random_forest.pkl
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/AgriTrust-AI.git
```

Move into the project directory:

```bash
cd AgriTrust-AI
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The Streamlit dashboard will open in your browser.

Upload a crop image to generate the prediction.

---

# 🔍 Example Prediction

```text
Crop:
Jute

Crop Confidence:
92%

Condition:
Disease

Condition Confidence:
93%

Decision Tree:
Disease

Random Forest:
Disease

Model Agreement:
YES

AI Reliability:
RELIABLE
```

The exact result depends on the input image.

---

# 🧪 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow | MobileNetV2 feature extraction |
| MobileNetV2 | Image feature extraction |
| Scikit-learn | Machine learning models |
| Random Forest | Classification and reliability detection |
| Decision Tree | Classification and reliability detection |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Pillow | Image processing |
| Streamlit | Interactive dashboard |
| Joblib | Model serialization |

---

# 🌟 Advantages

- Combines computer vision and traditional machine learning.
- Automatically identifies crop type.
- Provides Healthy/Disease prediction.
- Uses multiple models for comparison.
- Provides confidence information.
- Includes an AI self-reliability layer.
- Provides visual analytics.
- Provides prediction history.
- Can be extended to larger agricultural datasets.

---

# ⚠️ Limitations

- The dataset contains only six crop categories.
- Disease prediction is limited to Healthy/Disease classification.
- The system does not identify specific disease names.
- Reliability detection is an estimate based on model behavior.
- Real-world field conditions may differ from the dataset.
- Performance can vary depending on image quality and lighting.

---

# 🔮 Future Enhancements

Future versions can include:

- Specific disease classification
- More Indian crop categories
- Real-time camera-based prediction
- Weather and environmental data integration
- Soil information integration
- Explainable AI using SHAP or Grad-CAM
- Mobile application deployment
- Cloud deployment
- Larger and more diverse agricultural datasets
- Farmer-focused multilingual interface

---

# 👩‍💻 Project Concept

**AgriTrust AI** focuses on a broader machine learning question:

> **"Can AI recognize when it should not fully trust its own prediction?"**

This makes the project different from a standard crop classification system by introducing an additional **AI reliability detection layer**.

---

## 📌 Project Summary

AgriTrust AI combines:

```text
Computer Vision
       +
Machine Learning
       +
Model Comparison
       +
Confidence Analysis
       +
AI Reliability Detection
       =
Trust-Aware Crop Prediction System
```

---

## 📜 License

This project is developed for educational and academic purposes.

The dataset used in this project belongs to its original source and should be used according to the dataset's licensing terms.