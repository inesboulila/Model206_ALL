"""
Streamlit app for Model206_ALL_model.pkl
Run: streamlit run app_all.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re

st.set_page_config(page_title="miRNA Predictor (ALL)", page_icon="🧬", layout="wide")

# ── Load model bundle ─────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('Model206_ALL_model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    bundle           = load_model()
    model            = bundle['model']
    metrics          = bundle['metrics']
    options          = bundle['options']
    mirna_lookup     = bundle['mirna_lookup']
    accession_lookup = bundle['accession_lookup']
except FileNotFoundError:
    st.error("Model file not found. Run the training script first.")
    st.stop()


def normalize_mirna(name: str) -> str:
    """Strip species prefix (hsa-, mmu-) and arm suffix (-5p, -3p) for fuzzy matching."""
    name = name.strip().lower()
    name = re.sub(r'^[a-z]{3}-', '', name)
    name = re.sub(r'[-.](5p|3p)$', '', name)
    return name

def resolve_input(user_input: str):
    """
    Accepts either:
      - Full miRNA name:  hsa-miR-155-5p
      - Short miRNA name: miR-155-5p  (no species prefix)
      - Accession number: MIMAT0000646
    Returns (seed_family, family_conservation) or (None, None) if not found.
    """
    s = user_input.strip()

    # 1. Try exact miRNA name match
    if s in mirna_lookup:
        info = mirna_lookup[s]
        return info['seed_family'], info.get('family_conservation')

    # 2. Try accession number match
    if s in accession_lookup:
        info = accession_lookup[s]
        return info['seed_family'], info.get('family_conservation')

    # 3. Try fuzzy: normalize input and compare against normalized miRNA names
    norm_input = normalize_mirna(s)
    for mirna_name, info in mirna_lookup.items():
        if normalize_mirna(mirna_name) == norm_input:
            return info['seed_family'], info.get('family_conservation')

    return None, None


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.title("🧬 miRNA Upregulation Predictor")
st.markdown("Predicts whether a miRNA is **upregulated** or **downregulated** during *Leishmania* infection.")
st.caption("Model: LightGBM · all conservation levels · family_conservation feature included")
st.divider()

col_input, col_result = st.columns([1, 1], gap="large")

# ── LEFT: Inputs ──────────────────────────────────────────────
with col_input:
    st.subheader("Experimental conditions")

    mirna_input = st.text_input(
        "miRNA name or accession number",
        placeholder="e.g. hsa-miR-155-5p  or  miR-155-5p  or  MIMAT0000646"
    )

    parasite  = st.selectbox("Parasite species", options=options['parasite'])
    organism  = st.selectbox("Host organism",    options=options['organism'])
    cell_type = st.selectbox("Cell type",        options=options['cell_type'])

    time = st.number_input(
        "Time point (hours post-infection)",
        min_value=0, max_value=10000, value=24, step=1
    )

    predict_btn = st.button("Predict", type="primary", use_container_width=True)

# ── RIGHT: Result ─────────────────────────────────────────────
with col_result:
    st.subheader("Prediction")

    if predict_btn:
        if not mirna_input.strip():
            st.info("Please enter a miRNA name or accession number.")
        else:
            # Resolve input → seed_family + family_conservation
            seed_family, family_conservation = resolve_input(mirna_input.strip())

            if seed_family and seed_family not in ('not_broadly_conserved', 'not_found'):
                is_conserved = 1
                cons_label = {2: "broadly conserved", 1: "mammal conserved",
                              0: "poorly conserved", -1: "species-specific"}.get(
                              int(family_conservation) if family_conservation is not None else -99, "unknown")
                st.info(f"Found · seed family: **{seed_family}** · conservation: **{cons_label}**")
            else:
                # Not found — model relies on parasite, organism, cell type, time
                seed_family          = np.nan
                is_conserved         = 0
                family_conservation  = np.nan
                st.info(f"**{mirna_input.strip()}** not found. Prediction relies on parasite, organism, cell type, and time.")

            # Build input row — pass parasite/cell type exactly as in training (no lowercasing)
            parasite_celltype = f"{parasite}_{cell_type}"
            input_df = pd.DataFrame([{
                'parasite':            parasite,
                'organism':            organism,
                'cell type':           cell_type,
                'seed_family':         seed_family,
                'parasite_celltype':   parasite_celltype,
                'time':                time,
                'is_conserved':        is_conserved,
                'family_conservation': family_conservation,
            }])

            try:
                proba     = model.predict_proba(input_df)[0]
                pred      = model.predict(input_df)[0]
                prob_up   = proba[1]
                prob_down = proba[0]

                st.divider()
                if pred == 1:
                    st.success("## ⬆ Upregulated")
                else:
                    st.error("## ⬇ Downregulated")

                st.markdown(f"**Confidence:** {max(prob_up, prob_down)*100:.1f}%")

                c1, c2 = st.columns(2)
                c1.metric("Upregulated",   f"{prob_up   * 100:.1f}%")
                c2.metric("Downregulated", f"{prob_down * 100:.1f}%")

                st.progress(float(prob_up), text=f"↑ {prob_up*100:.1f}%  |  ↓ {prob_down*100:.1f}%")

                with st.expander("Input summary"):
                    st.dataframe(input_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")
    else:
        st.markdown("<div style='color:gray;margin-top:2rem'>Fill in the conditions and click <b>Predict</b>.</div>",
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════
st.divider()
st.subheader("Model performance")
st.caption(f"LightGBM · Optuna-tuned · {metrics['n_train']} training samples · 5-fold CV")

m1, m2, m3 = st.columns(3)
m1.metric("ROC-AUC",  f"{metrics['auc_mean']:.3f}", f"± {metrics['auc_std']:.3f}")
m2.metric("Accuracy", f"{metrics['acc_mean']:.3f}", f"± {metrics['acc_std']:.3f}")
m3.metric("F1 Score", f"{metrics['f1_mean']:.3f}",  f"± {metrics['f1_std']:.3f}")

st.markdown("**AUC per fold:**")
fold_cols = st.columns(len(metrics['auc_folds']))
for col, (i, v) in zip(fold_cols, enumerate(metrics['auc_folds'])):
    col.metric(f"Fold {i+1}", f"{v:.3f}")

st.markdown("**Feature importance:**")
fi = pd.DataFrame(metrics['feature_importance'])
fi['importance'] = fi['importance'].round(4)
fi['std']        = fi['std'].round(4)
fi.columns       = ['Feature', 'Importance (AUC drop)', 'Std']
st.dataframe(fi.style.background_gradient(subset=['Importance (AUC drop)'], cmap='Greens'),
             use_container_width=True, hide_index=True)

with st.expander("Best hyperparameters"):
    st.json(metrics['best_params'])
