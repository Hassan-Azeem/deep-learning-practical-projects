import requests
import streamlit as st
from sklearn.datasets import load_digits


st.set_page_config(page_title="Digit Reader", page_icon="0", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1160px; padding: 2rem 2rem 3rem;}
    [data-testid="stAppViewContainer"] {background: #f3f6f8;}
    .eyebrow {color: #0f766e; font-size: .72rem; font-weight: 700; letter-spacing: .12em;}
    h1 {color: #172b3a; letter-spacing: -.025em;}
    .intro {color: #62717d; margin-bottom: 1.2rem;}
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #fff; border-color: #dce3e8; border-radius: 6px;
    }
    [data-testid="stMetric"] {
        background: #f7f9fa; border: 1px solid #e3e8ec;
        border-radius: 5px; padding: .8rem 1rem;
    }
    </style>
    <div class="eyebrow">PROJECT 04 &nbsp; / &nbsp; DIGIT CLASSIFIER</div>
    """,
    unsafe_allow_html=True,
)
st.title("Digit reader")
st.markdown(
    '<p class="intro">Select a labelled 8 × 8 sample and send it to the local prediction API.</p>',
    unsafe_allow_html=True,
)

digits = load_digits()
index = st.slider("Sample index", 0, len(digits.images) - 1, 0)
sample_col, result_col = st.columns([1, 1.25], gap="large")

with sample_col:
    with st.container(border=True):
        st.subheader("Input sample")
        st.image(digits.images[index] / 16.0, width=280, clamp=True)
        st.caption(
            f"Sample {index + 1} of {len(digits.images)} · "
            f"Reference label: {int(digits.target[index])}"
        )

with result_col:
    with st.container(border=True):
        st.subheader("Model prediction")
        st.caption("Model: PyTorch feed-forward network · API: FastAPI")
        if st.button("Predict digit", type="primary", width="stretch"):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    json={"pixels": digits.data[index].tolist()},
                    timeout=10,
                )
                result = response.json()
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.metric("Predicted digit", result["digit"])
                    st.metric("Model confidence", f"{result['confidence']:.1%}")
                    st.caption(f"Reference label: {int(digits.target[index])}")
            except requests.RequestException as exc:
                st.error(f"Could not reach the prediction API: {exc}")
        else:
            st.info("Run a prediction to see the model output and confidence.")

st.caption(
    "Demo uses public scikit-learn samples. It is a digit-classification exercise, "
    "not handwriting verification."
)
