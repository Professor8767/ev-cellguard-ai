# ev-cellguard-ai
# 🔋 EV CellGuard AI
### AI-Based Detection of EV Battery Cell Imbalance

EV CellGuard AI is a machine-learning-based prototype designed to analyze lithium-ion battery cell voltage measurements and identify unusual cell-to-cell voltage patterns.

The project combines cell-voltage feature engineering, unsupervised anomaly detection, and an interactive Streamlit dashboard to support battery data exploration and preliminary diagnostics.

## 🚀 Live Application

**Streamlit Dashboard:** Add your deployed Streamlit Community Cloud URL here after deployment.

## 🎯 Project Objectives

- Analyze individual battery cell voltage measurements.
- Calculate cell-to-cell voltage spread and deviation features.
- Identify unusual voltage patterns using an Isolation Forest model.
- Visualize voltage spread and anomaly scores interactively.
- Export prediction results for further analysis.
- Establish a foundation for future integration with Battery Management Systems (BMS).

## ✨ Features

- **CSV Upload:** Upload battery cell-voltage datasets.
- **Cell Voltage Analysis:** Inspect individual cell measurements.
- **Feature Engineering:** Calculate voltage spread, standard deviation, and deviations from the mean cell voltage.
- **AI-Based Anomaly Detection:** Use a trained Isolation Forest model to flag unusual observations.
- **Interactive Visualizations:** Explore voltage spread and anomaly scores.
- **Prediction Table:** Review records ranked by anomaly score.
- **CSV Export:** Download analysis results for further investigation.

## 🧠 Machine Learning Approach

The project uses the Isolation Forest algorithm from Scikit-learn.

Isolation Forest is an unsupervised anomaly-detection algorithm that identifies observations that differ from learned data patterns.

### Engineered Features

The current model uses the following features:

1. Cell-voltage spread: maximum cell voltage minus minimum cell voltage.
2. Cell-voltage standard deviation.
3. Maximum absolute deviation from the average cell voltage.
4. Mean absolute deviation from the average cell voltage.
5. Individual cell-voltage deviations from the average cell voltage.

### Training Process

The model was trained in Google Colab using the `GMP_LFP_40Ah_3000Cycle.csv` dataset.

The workflow includes:

1. Dataset inspection and cleaning.
2. Cell-voltage column identification.
3. Feature engineering.
4. Chronological training and holdout split.
5. Isolation Forest training.
6. Anomaly-score analysis.
7. Model serialization using Joblib.
8. Deployment through Streamlit Community Cloud.

The trained model is saved as `ev_cellguard_model.joblib`.

**Important:** The current model is an unsupervised anomaly detector. An anomaly flag indicates an unusual pattern relative to the learned data distribution; it does not establish that a battery fault is present.

## 📊 Dataset

**Dataset filename:** `GMP_LFP_40Ah_3000Cycle.csv`

The dataset is used to develop and test the prototype's battery-cell analysis pipeline.

The cell-voltage measurements in this dataset include structured or simulated values. Therefore, results obtained from this dataset must not be interpreted as proof of performance on real-world EV battery faults.

Please refer to the dataset provider's documentation and license when redistributing the data.

## 🛠️ Technology Stack

- **Python** — Core programming language
- **Pandas** — Data processing
- **NumPy** — Numerical computation
- **Scikit-learn** — Machine learning and anomaly detection
- **Joblib** — Model serialization and loading
- **Plotly** — Interactive charts
- **Streamlit** — Web application and dashboard
- **Google Colab** — Model training environment
- **GitHub** — Source code and version control
- **Streamlit Community Cloud** — Application hosting

## 📁 Project Structure

```text
EV-CellGuard-AI/
├── app.py
├── requirements.txt
├── ev_cellguard_model.joblib
├── README.md
└── .gitignore
```

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_REPOSITORY_NAME
```

Replace the example placeholders with your actual GitHub repository URL and folder name.

### 2. Create a virtual environment

Windows:

```bash
py -m venv .venv
.venv\Scripts\activate
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

## ☁️ Deploy on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, `ev_cellguard_model.joblib`, and `README.md` to your GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new application.
4. Select the GitHub repository and deployment branch.
5. Set the main file path to `app.py`.
6. Deploy the application.
7. Open the generated public URL and upload a compatible