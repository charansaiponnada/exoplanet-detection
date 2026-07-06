"""
Interactive dashboard (Layer 8) for the exoplanet transit detection
pipeline: upload/select a light curve, run the full detection pipeline
(sigma-clip -> detrend -> BLS -> vetting -> feature engineering -> ML
classifier -> confidence scoring), and inspect + export the result.

Run with:
    uv run streamlit run app.py
"""

import io
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_io import make_synthetic_light_curve  # noqa: E402
from main import run_pipeline, DEFAULT_MODEL_PATH  # noqa: E402
from classifier import load_model  # noqa: E402

st.set_page_config(page_title="Exoplanet Transit Detection", layout="wide")
st.title("AI-Enabled Exoplanet Transit Detection")
st.caption("Bharatiya Antariksh Hackathon 2026 — Challenge 07")


@st.cache_resource
def get_ml_model():
    try:
        return load_model(DEFAULT_MODEL_PATH)
    except FileNotFoundError:
        return None


ml_model = get_ml_model()
if ml_model is None:
    st.sidebar.warning(
        "No trained classifier found at models/candidate_classifier.joblib — "
        "run `uv run src/synth_dataset.py` then `uv run src/classifier.py` to train one. "
        "The pipeline still runs fine with classical vetting only."
    )

st.sidebar.header("Input")
mode = st.sidebar.radio(
    "Data source",
    ["Synthetic: planet", "Synthetic: eclipsing binary", "Local TESS file",
     "Upload CSV", "Real TIC ID (MAST)"],
)

df, truth, label = None, None, None
auto_run = False

if mode == "Synthetic: planet":
    label = "synthetic_planet"
    df, truth = make_synthetic_light_curve(period_days=3.5, transit_depth_ppm=2500,
                                            transit_duration_hours=2.5, seed=1)
elif mode == "Synthetic: eclipsing binary":
    label = "synthetic_eclipsing_binary"
    df, truth = make_synthetic_light_curve(period_days=4.2, transit_depth_ppm=8000,
                                            transit_duration_hours=3.0, add_secondary_eclipse=True,
                                            secondary_depth_ppm=2500, seed=2)
elif mode == "Local TESS file":
    data_dir = Path(__file__).parent / "data" / "tess"
    files = sorted(p.name for p in data_dir.glob("TIC_*.csv")) if data_dir.exists() else []
    if files:
        chosen_name = st.sidebar.selectbox("File", files)
        df = pd.read_csv(data_dir / chosen_name)
        label = Path(chosen_name).stem
    else:
        st.sidebar.info("No files in data/tess/ — download some with helper_scripts/download/, "
                         "or use another data source.")
elif mode == "Upload CSV":
    uploaded = st.sidebar.file_uploader("CSV with time, flux, flux_err columns", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        label = Path(uploaded.name).stem
elif mode == "Real TIC ID (MAST)":
    tic_id = st.sidebar.text_input("TIC ID", value="TIC 307210830")
    if st.sidebar.button("Download from MAST"):
        from data_io import load_real_target
        try:
            with st.spinner(f"Downloading {tic_id} from MAST (requires internet)..."):
                df, meta = load_real_target(tic_id)
            label = tic_id
            auto_run = True
        except Exception as e:
            st.error(f"Download failed: {e}")

run_clicked = st.sidebar.button("Run pipeline", type="primary")

if df is not None and (run_clicked or auto_run):
    with st.spinner("Running detection pipeline..."):
        out_prefix = f"output/dashboard_{label.replace(' ', '_')}"
        results, fig = run_pipeline(
            df["time"].values, df["flux"].values, df["flux_err"].values,
            label=label, truth=truth, out_prefix=out_prefix,
            ml_model=ml_model, return_figure=True,
        )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rule-based classification", results["classification"]["label"])
    col2.metric("Confidence", f"{results['confidence']['score']:.2f}", results["confidence"]["verdict"])
    col3.metric("Detection SNR", f"{results['recovered_parameters']['detection_snr']:.1f}")
    col4.metric("Plausible?", "Yes" if results["plausibility"]["plausible"] else "No")

    st.pyplot(fig, width="stretch")

    tab_params, tab_vetting, tab_ml, tab_features, tab_report = st.tabs(
        ["Parameters", "Vetting & plausibility", "ML classifier (Layer 5)", "Features (Layer 4)", "Export"]
    )

    with tab_params:
        st.subheader("Recovered orbital/transit parameters")
        st.json(results["recovered_parameters"])
        if "ground_truth" in results:
            st.subheader("Ground truth (synthetic injection)")
            st.json(results["ground_truth"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Period error", f"{results['period_error_pct']:.2f}%")
            c2.metric("Depth error", f"{results['depth_error_pct']:.2f}%")
            c3.metric("Duration error", f"{results['duration_error_pct']:.2f}%")

    with tab_vetting:
        st.subheader("Classical vetting tests")
        st.write("**Odd-even depth test**", results["vetting"]["odd_even"])
        st.write("**Secondary eclipse search**", results["vetting"]["secondary_eclipse"])
        st.write("**Shape test (V vs U)**", results["vetting"]["shape"])
        if results["classification"]["flags"]:
            st.warning("Flags raised: " + "; ".join(results["classification"]["flags"]))
        else:
            st.success("No vetting flags raised.")
        st.subheader("Physical plausibility")
        if results["plausibility"]["flags"]:
            st.warning("Flags raised: " + "; ".join(results["plausibility"]["flags"]))
        else:
            st.success("No plausibility flags raised.")
        st.caption("Confidence breakdown (each term independently inspectable):")
        st.json(results["confidence"]["breakdown"])

    with tab_ml:
        if results["ml_classification"]:
            ml = results["ml_classification"]
            st.subheader(f"Predicted: {ml['predicted_label']}")
            st.bar_chart(pd.Series(ml["probabilities"], name="probability"))
            st.caption("Top contributing features (global model importance, not per-sample attribution):")
            st.table(pd.DataFrame(ml["top_contributing_features"]))
            st.info("Trained on synthetic-only data (`src/synth_dataset.py` + `src/classifier.py`) as an "
                    "honest interim for Layer 5, pending ISRO's curated real dataset for the planned "
                    "Mamba sequence model. Treat disagreements with the rule-based label above as a "
                    "signal for human review, not as an error.")
        else:
            st.info("No trained ML classifier found — showing classical vetting only.")

    with tab_features:
        st.dataframe(pd.DataFrame([results["features"]]).T.rename(columns={0: "value"}))

    with tab_report:
        st.download_button("Download results (JSON)", data=json.dumps(results, indent=2, default=str),
                            file_name=f"{label.replace(' ', '_')}_results.json", mime="application/json")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130)
        st.download_button("Download figure (PNG)", data=buf.getvalue(),
                            file_name=f"{label.replace(' ', '_')}_figure.png", mime="image/png")
else:
    st.info("Choose a data source in the sidebar and click **Run pipeline**.")
