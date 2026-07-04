"""
Food ML - Interfaccia web moderna (Streamlit).

Avvio:
    streamlit run app.py

Sostituisce la vecchia GUI Tkinter con un'esperienza utente moderna:
form di input validato, badge Nutriscore colorato, verdetto di salubrita'
e tabella interattiva degli alimenti simili. I modelli vengono addestrati
una sola volta e messi in cache.
"""
from __future__ import annotations

import streamlit as st

from program.predictor import FoodPredictor

# --- Configurazione pagina ---------------------------------------------------
st.set_page_config(
    page_title="Food ML - Analisi Nutrizionale",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Colori ufficiali Nutriscore (A verde -> E rosso)
NUTRISCORE_COLORS = {
    "a": "#038141", "b": "#85BB2F", "c": "#FECB02", "d": "#EE8100", "e": "#E63E11",
}


@st.cache_resource(show_spinner="Addestramento dei modelli in corso...")
def load_predictor() -> FoodPredictor:
    """Carica dataset e addestra i modelli una sola volta (cache)."""
    return FoodPredictor()


def nutriscore_badge(grade: str) -> str:
    """Restituisce l'HTML di un badge Nutriscore colorato."""
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
st.sidebar.caption("Analisi nutrizionale basata su Machine Learning")
st.sidebar.markdown("### Valori nutrizionali (per 100 g)")

with st.sidebar.form("input_form"):
    energy = st.number_input(
        "Calorie (kcal)", min_value=0.0, max_value=1000.0, value=250.0, step=10.0,
        help="Energia per 100 g (0 - 1000 kcal)",
    )
    carbs = st.number_input(
        "Carboidrati (g)", min_value=0.0, max_value=100.0, value=30.0, step=1.0,
    )
    proteins = st.number_input(
        "Proteine (g)", min_value=0.0, max_value=100.0, value=10.0, step=1.0,
    )
    fats = st.number_input(
        "Grassi (g)", min_value=0.0, max_value=100.0, value=8.0, step=1.0,
    )
    submitted = st.form_submit_button("🔍 Analizza", use_container_width=True, type="primary")

# --- Corpo principale --------------------------------------------------------
st.title("Analisi Nutrizionale Intelligente")
st.markdown(
    "Inserisci i valori nutrizionali nella barra laterale per predire il "
    "**Nutriscore**, valutare la **salubrità** dell'alimento e scoprire "
    "**alimenti simili**."
)

predictor = load_predictor()

if not submitted:
    st.info("👈 Compila il form nella barra laterale e premi **Analizza**.")
    st.stop()

if carbs + proteins + fats > 100:
    st.warning(
        "⚠️ La somma di carboidrati, proteine e grassi supera i 100 g su 100 g "
        "di prodotto: i risultati potrebbero non essere attendibili."
    )

values = FoodPredictor.build_values(energy, carbs, proteins, fats)

with st.spinner("Analisi in corso..."):
    result = predictor.predict(values)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Nutriscore")
    st.markdown(nutriscore_badge(result.nutriscore), unsafe_allow_html=True)
    st.caption("Scala da A (migliore) a E (peggiore)")

with col2:
    st.subheader("Valutazione salubrità")
    if result.is_healthy:
        st.success(f"✅ {result.health}")
    else:
        st.error(f"⚠️ {result.health}")

st.divider()
st.subheader(f"Alimenti simili ({len(result.similar_foods)})")

similar = result.similar_foods
if similar.empty:
    st.info("Nessun alimento simile trovato per i valori inseriti.")
else:
    display_cols = {
        "product_name": "Prodotto",
        "energy_kcal": "kcal",
        "proteins_100g": "Proteine (g)",
        "fat_100g": "Grassi (g)",
        "carbohydrates_100g": "Carboidrati (g)",
    }
    available = [c for c in display_cols if c in similar.columns]
    st.dataframe(
        similar[available].rename(columns=display_cols).reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

