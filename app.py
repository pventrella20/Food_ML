"""
Food ML - Modern web UI (Streamlit).

Run:
    streamlit run app.py

Replaces the old Tkinter GUI with a modern user experience:
validated input form, color-coded Nutri-Score badge, health verdict,
and interactive table of similar foods. Models are trained once and cached.
"""
from __future__ import annotations

import streamlit as st

from program.predictor import FoodPredictor

# --- Page configuration ------------------------------------------------------
st.set_page_config(
    page_title="Food ML - Nutritional Analysis",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Official Nutri-Score colors (A green -> E red)
NUTRISCORE_COLORS = {
    "a": "#038141", "b": "#85BB2F", "c": "#FECB02", "d": "#EE8100", "e": "#E63E11",
}


@st.cache_resource(show_spinner="Training models...")
def load_predictor() -> FoodPredictor:
    """Load dataset and train models once (cached)."""
    return FoodPredictor()


def nutriscore_badge(grade: str) -> str:
    """Return HTML for a color-coded Nutri-Score badge."""
    g = grade.strip().lower()
    color = NUTRISCORE_COLORS.get(g, "#777777")
    return (
        f"<div style='display:inline-block;background:{color};color:white;"
        f"font-size:48px;font-weight:800;width:90px;height:90px;line-height:90px;"
        f"text-align:center;border-radius:16px;box-shadow:0 4px 10px rgba(0,0,0,.2);'>"
        f"{g.upper()}</div>"
    )


# --- Sidebar: input ----------------------------------------------------------
st.sidebar.title("🥗 Food ML")
st.sidebar.caption("Nutritional analysis powered by Machine Learning")
st.sidebar.markdown("### Nutritional values (per 100 g)")

with st.sidebar.form("input_form"):
    energy = st.number_input(
        "Calories (kcal)", min_value=0.0, max_value=1000.0, value=250.0, step=10.0,
        help="Energy per 100 g (0 - 1000 kcal)",
    )
    carbs = st.number_input(
        "Carbohydrates (g)", min_value=0.0, max_value=100.0, value=30.0, step=1.0,
    )
    proteins = st.number_input(
        "Proteins (g)", min_value=0.0, max_value=100.0, value=10.0, step=1.0,
    )
    fats = st.number_input(
        "Fats (g)", min_value=0.0, max_value=100.0, value=8.0, step=1.0,
    )
    submitted = st.form_submit_button("🔍 Analyze", use_container_width=True, type="primary")

# --- Main area ---------------------------------------------------------------
st.title("Smart Nutritional Analysis")
st.markdown(
    "Enter nutritional values in the sidebar to predict "
    "**Nutri-Score**, evaluate the food **healthiness**, and discover "
    "**similar foods**."
)

predictor = load_predictor()

if not submitted:
    st.info("👈 Fill in the sidebar form and click **Analyze**.")
    st.stop()

if carbs + proteins + fats > 100:
    st.warning(
        "⚠️ The sum of carbohydrates, proteins, and fats exceeds 100 g per "
        "100 g of product: results may be unreliable."
    )

values = FoodPredictor.build_values(energy, carbs, proteins, fats)

with st.spinner("Analyzing..."):
    result = predictor.predict(values)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Nutriscore")
    st.markdown(nutriscore_badge(result.nutriscore), unsafe_allow_html=True)
    st.caption("Scale from A (best) to E (worst)")

with col2:
    st.subheader("Health assessment")
    if result.is_healthy:
        st.success(f"✅ {result.health}")
    else:
        st.error(f"⚠️ {result.health}")

st.divider()
st.subheader(f"Similar foods ({len(result.similar_foods)})")

similar = result.similar_foods
if similar.empty:
    st.info("No similar food found for the provided values.")
else:
    display_cols = {
        "product_name": "Product",
        "energy_kcal": "kcal",
        "proteins_100g": "Proteins (g)",
        "fat_100g": "Fats (g)",
        "carbohydrates_100g": "Carbohydrates (g)",
    }
    available = [c for c in display_cols if c in similar.columns]
    st.dataframe(
        similar[available].rename(columns=display_cols).reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

