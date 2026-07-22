"""Streamlit dashboard: browse applicants, see their calibrated default
probability and (for tree champions) the SHAP drivers behind it.

    make dashboard
"""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from home_credit import config, data
from home_credit.explain.shap_explainer import (
    TREE_MODEL_NAMES,
    build_explainer,
    contributions_for_row,
    load_champion,
    positive_class_values,
)
from home_credit.features import prepare_tree_dtypes

# Diverging pair (red/blue) and status colors from the project's data-viz
# palette - red/blue read as opposite poles (SHAP pushing risk up vs. down),
# the status steps are reserved for state (risk tier) and never reused as a
# categorical series color.
COLOR_INCREASES_RISK = "#e34948"
COLOR_DECREASES_RISK = "#2a78d6"
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_CRITICAL = "#d03b3b"

st.set_page_config(page_title="Home Credit Default Risk", layout="wide")


@st.cache_resource
def load_model_and_meta():
    champion, meta = load_champion()
    calibrated_model = joblib.load(config.MODEL_DIR / "calibrated_model.joblib")
    explainer = None
    if meta["champion_model"] in TREE_MODEL_NAMES:
        explainer = build_explainer(meta["champion_model"], champion)
    return calibrated_model, explainer, meta


@st.cache_data
def load_applicant_data():
    return data.load_mart()


def risk_tier(probability: float) -> tuple[str, str]:
    """Cut points are illustrative, not derived from a real cost/benefit
    analysis - a production build would set these from the business's
    actual approve/review/decline economics, not a flat 8%/20% split.
    """
    if probability < 0.08:
        return "Low", STATUS_GOOD
    if probability < 0.20:
        return "Medium", STATUS_WARNING
    return "High", STATUS_CRITICAL


def render_contributions_chart(contributions: list[dict], top_n: int = 15):
    top = contributions[:top_n][::-1]
    features = [c["feature"] for c in top]
    values = [c["shap_value"] for c in top]
    colors = [COLOR_INCREASES_RISK if v > 0 else COLOR_DECREASES_RISK for v in values]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(top))))
    ax.barh(features, values, color=colors)
    ax.axvline(0, color="#52514e", linewidth=1)
    ax.set_xlabel("SHAP value (impact on predicted score)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("🔴 pushes risk up  ·  🔵 pushes risk down")


def main():
    st.title("Home Credit Default Risk")
    st.caption("Browse applicants, see the calibrated default probability and the SHAP drivers behind it.")

    try:
        calibrated_model, explainer, meta = load_model_and_meta()
    except FileNotFoundError:
        st.error("No trained model found under `models/`. Run `make train` first.")
        return

    df = load_applicant_data()
    feature_cols, categorical_cols = meta["feature_cols"], meta["categorical_cols"]

    with st.sidebar:
        st.header("Applicant")
        pool = st.radio("Pool", ["Scoring set (application_test)", "Training set (application_train)"])
        pool_df = df[~df["is_train"]] if pool.startswith("Scoring") else df[df["is_train"]]
        sk_id = st.selectbox("SK_ID_CURR", sorted(pool_df[config.ID_COL].tolist()))
        st.caption(f"Champion model: **{meta['champion_model']}**")

    row = df[df[config.ID_COL] == sk_id]
    X = row[feature_cols]
    if meta["champion_model"] in TREE_MODEL_NAMES:
        X = prepare_tree_dtypes(row, categorical_cols, categories=meta["categorical_categories"])[feature_cols]

    probability = float(calibrated_model.predict_proba(X)[:, 1][0])
    tier, tier_color = risk_tier(probability)
    actual = row[config.TARGET_COL].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Default probability", f"{probability:.1%}")
    col2.markdown(
        f"**Risk tier**  \n"
        f"<span style='color:{tier_color}; font-size:1.3em'>&#9679;</span> {tier}",
        unsafe_allow_html=True,
    )
    col3.metric(
        "Actual outcome",
        "Default" if actual == 1 else ("No default" if actual == 0 else "Unknown (scoring set)"),
    )

    st.divider()

    if explainer is None:
        st.info(
            f"SHAP explanation isn't available for champion_model={meta['champion_model']!r} "
            "(scoped to LightGBM/XGBoost champions - see the README's Explainability section)."
        )
        return

    explanation = explainer(X)
    shap_row = positive_class_values(explanation.values)[0]
    contributions = contributions_for_row(feature_cols, row[feature_cols].iloc[0].tolist(), shap_row)

    st.subheader(f"Top drivers for applicant {sk_id}")
    render_contributions_chart(contributions)

    with st.expander("All feature contributions"):
        # "value" mixes numeric and string features across rows (a raw
        # applicant value, whatever type that feature is) - stringified for
        # display only, so Arrow doesn't have to guess a common dtype for a
        # genuinely mixed-type column.
        contributions_df = pd.DataFrame(contributions)
        contributions_df["value"] = contributions_df["value"].astype(str)
        st.dataframe(contributions_df, width="stretch")

    st.divider()
    st.subheader("Global feature importance")
    global_importance_path = config.REPORTS_DIR / "shap_summary.png"
    if global_importance_path.exists():
        st.image(str(global_importance_path))
    else:
        st.info("Run `make explain` to generate the global SHAP summary plot.")


if __name__ == "__main__":
    main()
