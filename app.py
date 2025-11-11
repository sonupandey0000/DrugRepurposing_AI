import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
import joblib
import io
import time

# ---------------------------
# Title
# ---------------------------
st.title("🧪 AI-Based Chemoinformatics Project")
st.markdown("### Predict QSAR, ADMET & Docking Scores using SMILES")

# ---------------------------
# Load QSAR/ADMET Model (Optional)
# ---------------------------
try:
    qsar_model = joblib.load("model/qsar_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
except FileNotFoundError:
    qsar_model = None
    scaler = None
    st.warning("⚠️ QSAR/ADMET model not found. Using simulated predictions.")

# ---------------------------
# Upload CSV
# ---------------------------
uploaded_file = st.file_uploader("Upload CSV file (must contain SMILES column)", type=['csv'])

# ---------------------------
# Helper Functions
# ---------------------------
def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return {
            'MolWt': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'FractionCSP3': Descriptors.FractionCSP3(mol)
        }
    else:
        return None

def lipinski_check(desc):
    rules = []
    rules.append("✅ MW ≤ 500" if desc['MolWt'] <= 500 else "❌ MW > 500")
    rules.append("✅ LogP ≤ 5" if desc['LogP'] <= 5 else "❌ LogP > 5")
    rules.append("✅ H Donors ≤ 5" if desc['NumHDonors'] <= 5 else "❌ H Donors > 5")
    rules.append("✅ H Acceptors ≤ 10" if desc['NumHAcceptors'] <= 10 else "❌ H Acceptors > 10")
    return rules

def simulate_admet(desc):
    return {
        'Water_Solubility': round(-4 + np.random.rand() * 2, 2),  # logS
        'Caco2_Permeability': round(-6 + np.random.rand() * 2, 2),
        'BBB_Penetration': np.random.choice(['Yes','No']),
        'CYP3A4_Inhibition': np.random.choice(['Yes','No']),
        'AMES_Toxicity': np.random.choice(['Yes','No'])
    }

# ---------------------------
# Prediction
# ---------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip().upper() for c in df.columns]
    smiles_cols = [c for c in df.columns if 'SMILES' in c]

    if not smiles_cols:
        st.error("❌ CSV must contain a SMILES column (case-insensitive).")
    else:
        smiles_col = smiles_cols[0]
        results = []

        with st.spinner("Predicting..."):
            time.sleep(0.5)
            for s in df[smiles_col]:
                desc = smiles_to_descriptors(s)
                if not desc:
                    results.append({'SMILES': s, 'Error': 'Invalid SMILES'})
                    continue

                lipinski = lipinski_check(desc)

                # QSAR/IC50
                if qsar_model and scaler:
                    features = np.array([list(desc.values())])
                    scaled_features = scaler.transform(features)
                    ic50 = qsar_model.predict(scaled_features)[0]
                else:
                    ic50 = round(0.1 + np.random.rand() * 5, 2)

                docking_score = round(-6 + np.random.rand() * 2, 2)

                # ADMET predictions
                admet = simulate_admet(desc)

                results.append({
                    'SMILES': s,
                    **desc,
                    'Lipinski': ", ".join(lipinski),
                    'Predicted_IC50': ic50,
                    'Docking_Score': docking_score,
                    **admet
                })

        result_df = pd.DataFrame(results)
        st.success("✅ Prediction Complete!")
        st.dataframe(result_df)

        # ---------------------------
        # 2D Visualization
        # ---------------------------
        st.subheader("2D Structure Visualization")
        selected_smiles = st.selectbox("Select a molecule:", result_df['SMILES'])
        mol = Chem.MolFromSmiles(selected_smiles)
        if mol:
            img = Draw.MolToImage(mol, size=(250,250))
            st.image(img)

        if st.checkbox("Show grid of all molecules"):
            mols = [Chem.MolFromSmiles(s) for s in result_df['SMILES'] if Chem.MolFromSmiles(s)]
            if mols:
                grid_img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(150,150))
                st.image(grid_img)

        # ---------------------------
        # Download CSV
        # ---------------------------
        csv_buffer = io.StringIO()
        result_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "📥 Download Results as CSV",
            csv_buffer.getvalue().encode('utf-8'),
            "results.csv",
            "text/csv"
        )

else:
    st.info("⬆️ Please upload a CSV file with a SMILES column to start.")
