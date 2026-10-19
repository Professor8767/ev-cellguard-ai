
import io
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="EV CellGuard AI",
    page_icon="🔋",
    layout="wide",
)

MODEL_PATH = "ev_cellguard_model.joblib"


@st.cache_resource
def load_model_bundle():
    # Only load a trusted model file that you created.
    return joblib.load(MODEL_PATH)


def build_features(data, voltage_cols):
    """Must match the feature engineering used in Colab."""
    voltages = data[voltage_cols].astype(float)
    mean_v = voltages.mean(axis=1)
    deviations = voltages.sub(mean_v, axis=0)

    features = pd.DataFrame(index=data.index)
    features["voltage_spread_v"] = (
        voltages.max(axis=1) - voltages.min(axis=1)
    )
    features["voltage_std_v"] = voltages.std(axis=1, ddof=0)
    features["max_abs_deviation_v"] = deviations.abs().max(axis=1)
    features["mean_abs_deviation_v"] = deviations.abs().mean(axis=1)

    for i, col in enumerate(voltage_cols, 1):
        features[f"cell_{i}_deviation_v"] = deviations[col]

    return features.replace([np.inf, -np.inf], np.nan)


st.title("🔋 EV CellGuard AI")
st.subheader("AI-Based Detection of EV Battery Cell Imbalance")

st.write(
    "Upload battery cell-voltage records to identify unusual "
    "voltage patterns using your trained Isolation Forest model."
)

st.warning(
    "Research prototype only. An anomaly flag is not a confirmed "
    "battery fault or a guarantee of safety. Do not use this app "
    "to control, charge, or protect a real battery."
)

try:
    bundle = load_model_bundle()
    model = bundle["model"]
    voltage_cols = bundle["cell_voltage_columns"]
    feature_cols = bundle["feature_columns"]
    medians = bundle.get("cell_medians", {})
except Exception as exc:
    st.error(
        "Could not load the model. Confirm that "
        f"{MODEL_PATH} is uploaded and valid."
    )
    st.exception(exc)
    st.stop()

st.success("Trained model loaded successfully.")

uploaded_file = st.file_uploader(
    "Upload battery CSV",
    type=["csv"],
    help=(
        "The CSV must contain the same cell-voltage column names "
        "used when training the model in Colab."
    ),
)

if uploaded_file is None:
    st.info("Upload a CSV to begin analysis.")
    st.write("Required cell-voltage columns:")
    st.code(", ".join(voltage_cols))
    st.stop()

try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

raw_df.columns = raw_df.columns.astype(str).str.strip()

missing_cols = [c for c in voltage_cols if c not in raw_df.columns]
if missing_cols:
    st.error("Your CSV is missing required cell-voltage columns:")
    st.write(missing_cols)
    st.write("Available columns:")
    st.write(raw_df.columns.tolist())
    st.stop()

# Keep the original row number for traceability.
raw_df = raw_df.reset_index(drop=True)
raw_df.insert(0, "source_row", np.arange(1, len(raw_df) + 1))

for col in voltage_cols:
    raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

# Exclude records where every cell voltage is missing.
valid_mask = raw_df[voltage_cols].notna().any(axis=1)
analysis_df = raw_df.loc[valid_mask].copy()

if analysis_df.empty:
    st.error("No valid cell-voltage records were found.")
    st.stop()

# Match the median imputation used during training.
for col in voltage_cols:
    fallback = medians.get(col, np.nan)
    if pd.isna(fallback):
        fallback = analysis_df[col].median()

    if pd.isna(fallback):
        st.error(f"Cannot impute missing values in {col}.")
        st.stop()

    analysis_df[col] = analysis_df[col].fillna(float(fallback))

X = build_features(analysis_df, voltage_cols)

if X.isna().any().any():
    st.error("Feature calculation produced missing values.")
    st.stop()

# Ensure the feature order matches training exactly.
missing_features = [c for c in feature_cols if c not in X.columns]
if missing_features:
    st.error(f"Model features are missing: {missing_features}")
    st.stop()

X = X[feature_cols]

try:
    # Isolation Forest: -1 = anomaly, 1 = not flagged.
    predictions = model.predict(X)
    scores = -model.score_samples(X)
except Exception as exc:
    st.error(f"Model prediction failed: {exc}")
    st.stop()

results = analysis_df.copy()
results["voltage_spread_mv"] = (
    analysis_df[voltage_cols].max(axis=1)
    - analysis_df[voltage_cols].min(axis=1)
) * 1000

results["anomaly_score"] = scores
results["anomaly_flag"] = predictions == -1
results["status"] = np.where(
    results["anomaly_flag"],
    "Anomaly flagged",
    "Not flagged",
)

# Dashboard metrics
total_records = len(results)
flagged_records = int(results["anomaly_flag"].sum())
flagged_pct = 100 * flagged_records / total_records

c1, c2, c3, c4 = st.columns(4)
c1.metric("Records analysed", f"{total_records:,}")
c2.metric("Anomalies flagged", f"{flagged_records:,}")
c3.metric("Flagged records", f"{flagged_pct:.1f}%")
c4.metric(
    "Maximum voltage spread",
    f"{results['voltage_spread_mv'].max():.1f} mV",
)

st.divider()

st.subheader("Cell Voltage Spread")
fig_spread = px.line(
    results,
    x="source_row",
    y="voltage_spread_mv",
    title="Maximum cell voltage − minimum cell voltage",
    labels={
        "source_row": "Source CSV row",
        "voltage_spread_mv": "Voltage spread (mV)",
    },
)
st.plotly_chart(fig_spread, use_container_width=True)

st.subheader("Anomaly Score")
fig_score = px.scatter(
    results,
    x="source_row",
    y="anomaly_score",
    color="status",
    hover_data=["voltage_spread_mv"],
    title="Higher scores indicate more unusual records",
    labels={
        "source_row": "Source CSV row",
        "anomaly_score": "Anomaly score",
    },
)
st.plotly_chart(fig_score, use_container_width=True)

st.subheader("Cell Voltage Measurements")
voltage_display = results[
    ["source_row"] + voltage_cols
].copy()
st.dataframe(voltage_display, use_container_width=True)

st.subheader("Detailed Predictions")
st.dataframe(
    results[
        [
            "source_row",
            "voltage_spread_mv",
            "anomaly_score",
            "status",
        ]
    ].sort_values("anomaly_score", ascending=False),
    use_container_width=True,
)

st.download_button(
    "Download prediction results",
    data=results.to_csv(index=False).encode("utf-8"),
    file_name="ev_cellguard_predictions.csv",
    mime="text/csv",
)

st.caption(
    "The anomaly threshold was learned from the training configuration. "
    "The flagged percentage is not a measured fault rate. Results depend "
    "on the training dataset and should be independently validated."
)
