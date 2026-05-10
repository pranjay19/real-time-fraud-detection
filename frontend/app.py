"""
Fraud Detection Dashboard — Phase 4: Frontend Development
==========================================================
Day 12 : Streamlit scaffold + page config
Day 13 : Full UI — sidebar inputs for all 30 features
Day 14 : API integration via requests + conditional alert formatting
"""

import requests
import streamlit as st

# ─── Day 12: Page Configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detect Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* ── Global font ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        /* ── Header banner ── */
        .hero {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
            border-radius: 12px;
            padding: 2rem 2.5rem;
            margin-bottom: 1.5rem;
        }
        .hero h1 { color: #e2e8f0; font-size: 2rem; margin: 0; }
        .hero p  { color: #94a3b8; margin: 0.4rem 0 0; font-size: 0.95rem; }

        /* ── Result cards ── */
        .result-fraud {
            background: linear-gradient(135deg, #7f1d1d, #991b1b);
            border: 1px solid #ef4444;
            border-radius: 12px;
            padding: 1.5rem 2rem;
            color: #fef2f2;
        }
        .result-safe {
            background: linear-gradient(135deg, #052e16, #14532d);
            border: 1px solid #22c55e;
            border-radius: 12px;
            padding: 1.5rem 2rem;
            color: #f0fdf4;
        }
        .result-fraud h2, .result-safe h2 { margin: 0 0 0.5rem; font-size: 1.6rem; }
        .result-fraud p,  .result-safe p  { margin: 0; font-size: 1rem; opacity: 0.9; }

        /* ── Metric cards ── */
        div[data-testid="metric-container"] {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 0.75rem 1rem;
        }

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {
            background: #0f172a;
        }
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] label { color: #94a3b8 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Hero Banner ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>🛡️ Real-Time Credit Card Fraud Detection</h1>
        <p>Enter transaction details in the sidebar, then click <strong>Analyse Transaction</strong>
           to get an instant fraud verdict from the ML engine.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Day 13: Sidebar — All 30 Feature Inputs ────────────────────────────────
API_URL = "http://127.0.0.1:8000/predict"

with st.sidebar:
    st.markdown("## ⚙️ Transaction Features")
    st.markdown("Adjust the parameters to simulate a transaction.")
    st.markdown("---")

    # Transaction metadata
    st.subheader("Transaction Info")
    time_val   = st.number_input("Time (seconds from first record)",
                                 min_value=0.0, max_value=200_000.0,
                                 value=0.0, step=1.0)
    amount_val = st.number_input("Amount ($)",
                                 min_value=0.0, max_value=30_000.0,
                                 value=150.0, step=0.01, format="%.2f")

    st.markdown("---")
    st.subheader("PCA Features (V1 – V28)")
    st.caption("These are anonymised components from PCA transformation.")

    # Generate sliders for V1–V28 dynamically
    v_values: dict[str, float] = {}
    for i in range(1, 29):
        v_values[f"V{i}"] = st.slider(
            f"V{i}", min_value=-50.0, max_value=50.0, value=0.0, step=0.01
        )

    st.markdown("---")
    analyse_btn = st.button("🔍 Analyse Transaction", use_container_width=True)

# ─── Main Panel — Info Columns ───────────────────────────────────────────────
col_a, col_b, col_c = st.columns(3)
col_a.metric("Backend URL",   "127.0.0.1:8000")
col_b.metric("Model",         "Random Forest + SMOTE")
col_c.metric("Features",      "30 (scaled + PCA)")

st.markdown("---")

# ─── Day 14: API Integration & Conditional Alerts ───────────────────────────
if analyse_btn:
    # Build payload matching the Pydantic TransactionData schema
    payload = {
        "Time":   time_val,
        "Amount": amount_val,
        **v_values,          # unpacks V1 … V28
    }

    with st.spinner("Connecting to Fraud Detection Engine..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=10)

            if response.status_code == 200:
                result      = response.json()
                prediction  = result["fraud_prediction"]
                probability = result["confidence_probability"]
                verdict     = result["status"]

                # ── Result card ──
                if prediction == 1:
                    st.markdown(
                        f"""
                        <div class="result-fraud">
                            <h2>🚨 FRAUD DETECTED</h2>
                            <p>Confidence this transaction is fraudulent:
                               <strong>{probability:.2%}</strong></p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-safe">
                            <h2>✅ TRANSACTION LEGITIMATE</h2>
                            <p>Fraud probability:
                               <strong>{probability:.2%}</strong></p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # ── Detail metrics ──
                st.markdown("#### Prediction Details")
                d1, d2, d3 = st.columns(3)
                d1.metric("Verdict",              verdict)
                d2.metric("Fraud Probability",    f"{probability:.4f}")
                d3.metric("Binary Prediction",    str(prediction))

                # ── Raw JSON expander ──
                with st.expander("View raw API response"):
                    st.json(result)

            elif response.status_code == 503:
                st.warning(
                    "The model is not loaded on the API server. "
                    "Run `02_model_training.ipynb` to generate model.pkl and scaler.pkl, "
                    "then restart the FastAPI server."
                )
            else:
                st.error(
                    f"API returned status {response.status_code}: {response.text}"
                )

        except requests.exceptions.ConnectionError:
            st.error(
                "**Connection Error** — Is the FastAPI backend running?  \n"
                "Start it with: `cd api && python main.py`"
            )
        except requests.exceptions.Timeout:
            st.error("Request timed out. The API took too long to respond.")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")

else:
    # Placeholder state — shown before any button press
    st.info(
        "👈 Adjust the transaction features in the sidebar, then click "
        "**Analyse Transaction** to get the fraud verdict."
    )
