"""Interactive Streamlit dashboard for appliance energy prediction."""

from __future__ import annotations

import json
from datetime import date, time

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf

from config import (
    FIGURE_DIR,
    METRIC_DIR,
    MODEL_DIR,
    PREDICTION_DIR,
    REPORT_DIR,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
)
from src.data_loader import DatasetError, load_dataset
from src.predict import predict_from_frame
from src.train import train_pipeline

st.set_page_config(page_title="Appliance Energy MLP", page_icon="⚡", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.8rem; padding-bottom: 2.5rem;}
    [data-testid="stMetric"] {background:#f3f8f6; border:1px solid #dce9e4;
      padding:14px 16px; border-radius:12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

PAGES = [
    "Project Overview",
    "Dataset Explorer",
    "Model Training",
    "Model Evaluation",
    "Energy Prediction",
    "Reports and Downloads",
]


@st.cache_data
def get_dataset() -> pd.DataFrame:
    return load_dataset()


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Not generated yet: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Not generated yet: {path.name}")
    return pd.read_csv(path)


def model_status() -> bool:
    return (MODEL_DIR / "deep_mlp.keras").exists() and (MODEL_DIR / "preprocessor.joblib").exists()


def show_setup_message() -> None:
    st.warning("The official dataset is not ready. Download `energydata_complete.csv` from the UCI repository and place it at `data/energydata_complete.csv`.")
    st.code("python main.py\nstreamlit run app.py", language="powershell")
    st.markdown("[Open the official UCI dataset page](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)")


def overview() -> None:
    st.title("Household Appliance Energy Prediction")
    st.write("Estimate household appliance energy demand from indoor climate, weather, lighting, and time features using a deep Multi-Layer Perceptron.")
    st.info("Sustainability application: historical energy-use estimates can support household energy awareness and demand planning. Predictions are estimates, not guarantees.")
    try:
        frame = get_dataset()
        st.success(f"Dataset ready: {len(frame):,} records × {len(frame.columns)} columns.")
        if (METRIC_DIR / "results.json").exists():
            result = read_json(METRIC_DIR / "results.json")
            cols = st.columns(4)
            cols[0].metric("Dataset rows", f"{len(frame):,}")
            cols[1].metric("Training / validation / test", "70% / 15% / 15%")
            cols[2].metric("Test MAE", f"{result['deep_mlp']['MAE']:.2f} Wh")
            cols[3].metric("Test R²", f"{result['deep_mlp']['R2']:.4f}")
            st.subheader("Test-set model comparison")
            comparison = pd.DataFrame(
                [{"Model": "Training-mean baseline", **result["baseline"]}, {"Model": "Deep MLP", **result["deep_mlp"]}]
            )
            st.dataframe(comparison, width="stretch", hide_index=True)
            st.caption("Metrics are from the saved held-out test predictions; they are not hard-coded.")
        else:
            st.info("Dataset is ready; model evaluation is not available yet. Start a training run in Model Training.")
    except (DatasetError, FileNotFoundError) as exc:
        show_setup_message()
        st.caption(str(exc))

    st.subheader("Model design")
    st.write("Three dense hidden layers (128 → 64 → 32 ReLU units), one linear output, MSE objective, and explicit full-batch gradient descent. Exactly one update is applied per epoch using every training observation.")
    if model_status():
        st.success("A trained model and its preprocessing bundle are available.")
    else:
        st.warning("No trained model is saved yet. Visit Model Training or run `python main.py`.")


def dataset_explorer() -> None:
    st.title("Dataset Explorer")
    try:
        frame = get_dataset()
    except (DatasetError, FileNotFoundError) as exc:
        show_setup_message()
        st.caption(str(exc))
        return
    timestamp = frame.attrs.get("timestamp_column", "date")
    metrics = st.columns(4)
    metrics[0].metric("Observations", f"{len(frame):,}")
    metrics[1].metric("Columns", len(frame.columns))
    metrics[2].metric("Missing values", int(frame.isna().sum().sum()))
    metrics[3].metric("Duplicate rows", int(frame.duplicated().sum()))
    st.subheader("Dataset preview")
    st.dataframe(frame.head(20), width="stretch")
    st.subheader("Schema and missing values")
    schema = pd.DataFrame({"column": frame.columns, "dtype": [str(value) for value in frame.dtypes], "missing": frame.isna().sum().values})
    st.dataframe(schema, width="stretch", hide_index=True)
    st.subheader("Target distribution and time series")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.histogram(frame, x="Appliances", nbins=40, title="Appliance energy distribution (Wh)", color_discrete_sequence=["#16856b"]), width="stretch")
    with right:
        st.plotly_chart(px.line(frame, x=timestamp, y="Appliances", title="Appliance energy over time"), width="stretch")
    st.subheader("Correlation heatmap")
    numeric = frame.select_dtypes(include=np.number)
    corr = numeric.corr()
    st.plotly_chart(go.Figure(go.Heatmap(z=corr, x=corr.columns, y=corr.index, zmin=-1, zmax=1, colorscale="RdBu")), width="stretch")
    numeric_columns = list(numeric.columns)
    temperature_columns = [name for name in numeric_columns if name.startswith("T")]
    humidity_columns = [name for name in numeric_columns if name.startswith("RH")]
    for heading, columns, unit in (
        ("Temperature trends", temperature_columns, "°C"),
        ("Humidity trends", humidity_columns, "%"),
    ):
        if columns:
            st.subheader(heading)
            st.plotly_chart(px.line(frame, x=timestamp, y=columns, labels={"value": unit}), width="stretch")
    st.subheader("Feature statistics")
    st.dataframe(numeric.describe().T, width="stretch")


def training_page() -> None:
    st.title("Model Training")
    st.write("Configure a run below. Training starts only when you submit the form; each epoch performs one full-training-set gradient calculation and one parameter update.")
    with st.form("training_configuration"):
        learning_rate = st.number_input("Learning rate", min_value=0.00001, max_value=0.1, value=float(DEFAULT_LEARNING_RATE), format="%.5f")
        epochs = st.number_input("Maximum epochs", min_value=1, max_value=1000, value=int(DEFAULT_EPOCHS), step=10)
        patience = st.number_input("Early-stopping patience", min_value=1, max_value=100, value=15)
        st.caption("Architecture: 128 → 64 → 32 ReLU; linear output. Training and target scaling are fitted on training data only.")
        start = st.form_submit_button("Start training", type="primary")
    if start:
        try:
            load_dataset()
            progress = st.progress(0.0)
            status = st.empty()
            chart = st.empty()
            log = st.empty()
            losses: list[dict] = []

            def update(epoch: int, total: int, loss: float, val_loss: float) -> None:
                progress.progress(min(epoch / total, 1.0))
                status.write(f"Epoch {epoch}/{total} · training MSE {loss:.6f} · validation MSE {val_loss:.6f}")
                losses.append({"epoch": epoch, "training": loss, "validation": val_loss})
                chart.line_chart(pd.DataFrame(losses).set_index("epoch"))
                if epoch == 1 or epoch % 10 == 0:
                    log.code(f"Epoch {epoch}: training={loss:.6f}, validation={val_loss:.6f}")

            with st.spinner("Training the MLP using explicit full-batch gradient descent..."):
                result = train_pipeline(float(learning_rate), int(epochs), int(patience), update)
            st.success(f"Training and evaluation complete. Test MLP MAE: {result['deep_mlp']['MAE']:.3f} Wh; R²: {result['deep_mlp']['R2']:.4f}.")
            st.cache_data.clear()
        except (DatasetError, FileNotFoundError, ValueError, RuntimeError) as exc:
            st.error(f"Training failed: {exc}")
    if (METRIC_DIR / "training_history.csv").exists():
        st.subheader("Saved training history")
        history = read_csv(METRIC_DIR / "training_history.csv")
        st.line_chart(history.set_index("epoch")[["loss", "val_loss"]])
        st.dataframe(history, width="stretch", hide_index=True)


def evaluation_page() -> None:
    st.title("Model Evaluation")
    try:
        results = read_json(METRIC_DIR / "results.json")
        predictions = read_csv(PREDICTION_DIR / "test_predictions.csv")
        history = read_csv(METRIC_DIR / "training_history.csv")
    except FileNotFoundError as exc:
        st.info(f"{exc}. Train the model from Model Training or run `python main.py`.")
        return
    cols = st.columns(3)
    cols[0].metric("MLP MAE", f"{results['deep_mlp']['MAE']:.3f} Wh")
    cols[1].metric("MLP RMSE", f"{results['deep_mlp']['RMSE']:.3f} Wh")
    cols[2].metric("MLP R²", f"{results['deep_mlp']['R2']:.4f}")
    comparison = pd.DataFrame([{"Model": "Training-mean baseline", **results["baseline"]}, {"Model": "Deep MLP", **results["deep_mlp"]}])
    st.subheader("Baseline versus MLP")
    st.dataframe(comparison, width="stretch", hide_index=True)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.scatter(predictions, x="actual_Wh", y="predicted_Wh", title="Actual vs predicted (Wh)", opacity=0.55), width="stretch")
    with right:
        residual = px.scatter(predictions, x="predicted_Wh", y="residual_Wh", title="Residuals vs predicted")
        residual.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(residual, width="stretch")
    st.plotly_chart(px.histogram(predictions, x="residual_Wh", nbins=40, title="Residual distribution"), width="stretch")
    st.plotly_chart(px.line(predictions, x="timestamp", y=["actual_Wh", "predicted_Wh"], title="Test-period actual and predicted energy"), width="stretch")
    st.plotly_chart(px.line(history, x="epoch", y=["loss", "val_loss"], title="Training and validation loss"), width="stretch")


def prediction_page() -> None:
    st.title("Energy Prediction")
    st.caption("Enter sensor readings and a timestamp. The form uses the saved training feature schema, imputer, scaler, and feature order.")
    if not model_status():
        st.info("Prediction is unavailable until a model has been trained. Run `python main.py` or use Model Training.")
        return
    try:
        frame = get_dataset()
        bundle = joblib.load(MODEL_DIR / "preprocessor.joblib")
        timestamp_column = bundle["timestamp_column"]
        raw_features = bundle["raw_feature_names"]
        with st.form("energy_prediction"):
            chosen_date = st.date_input("Prediction date", value=date(2016, 5, 1))
            chosen_time = st.time_input("Prediction time", value=time(12, 0))
            values: dict[str, float] = {}
            columns = st.columns(3)
            for index, feature in enumerate(raw_features):
                series = pd.to_numeric(frame[feature], errors="coerce").dropna()
                if series.empty:
                    default, lower, upper = 0.0, -1_000_000.0, 1_000_000.0
                else:
                    default = float(series.median())
                    lower = float(series.quantile(0.01))
                    upper = float(series.quantile(0.99))
                    if lower >= upper:
                        lower, upper = default - 1.0, default + 1.0
                    default = min(max(default, lower), upper)
                with columns[index % len(columns)]:
                    values[feature] = st.number_input(
                        feature,
                        min_value=lower,
                        max_value=upper,
                        value=default,
                        help=f"Typical observed range: {lower:.3g} to {upper:.3g}",
                    )
            submit = st.form_submit_button("Estimate appliance energy", type="primary")
        if submit:
            timestamp_value = pd.Timestamp.combine(chosen_date, chosen_time)
            row = pd.DataFrame([{**values, timestamp_column: timestamp_value}])
            model = tf.keras.models.load_model(MODEL_DIR / "deep_mlp.keras")
            prediction = predict_from_frame(row, model=model, preprocessor=bundle)
            st.success(f"Estimated appliance energy consumption: {prediction:.2f} Wh")
            st.caption("This is a model estimate based on historical data. It is not a guaranteed measurement, and unusual values may reduce reliability.")
    except (DatasetError, FileNotFoundError, KeyError, ValueError) as exc:
        st.error(f"Could not prepare prediction: {exc}")


def reports_page() -> None:
    st.title("Reports and Downloads")
    requested_files = [
        (METRIC_DIR / "metrics.csv", "Evaluation metrics CSV"),
        (PREDICTION_DIR / "test_predictions.csv", "Test predictions CSV"),
        (METRIC_DIR / "training_history.csv", "Training history CSV"),
        (REPORT_DIR / "household_appliance_energy_prediction_report.md", "Project report (Markdown)"),
    ]
    available = False
    for path, label in requested_files:
        if path.exists():
            available = True
            st.download_button(label, path.read_bytes(), file_name=path.name, mime="text/markdown" if path.suffix == ".md" else "text/csv")
    figures = sorted(FIGURE_DIR.glob("*.png")) if FIGURE_DIR.exists() else []
    if figures:
        available = True
        st.subheader("Generated figures")
        for path in figures:
            st.download_button(f"Download {path.stem}", path.read_bytes(), file_name=path.name, mime="image/png", key=f"download_{path.name}")
    if not available:
        st.info("No evaluation artifacts are available yet. Run training first.")


st.sidebar.title("⚡ Energy MLP")
selected_page = st.sidebar.radio("Navigation", PAGES)
st.sidebar.caption("UCI Appliances Energy Prediction · Regression")
{
    "Project Overview": overview,
    "Dataset Explorer": dataset_explorer,
    "Model Training": training_page,
    "Model Evaluation": evaluation_page,
    "Energy Prediction": prediction_page,
    "Reports and Downloads": reports_page,
}[selected_page]()
