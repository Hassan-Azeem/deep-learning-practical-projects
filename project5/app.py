import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

try:
    from project5 import trainer
except ModuleNotFoundError:
    import trainer


st.set_page_config(page_title="Model Lab | Experiment monitor", layout="wide")
st.markdown(
    """
    <style>
    :root {
        --ink: #172b3a;
        --muted: #62717d;
        --line: #dce3e8;
        --paper: #ffffff;
        --canvas: #f3f6f8;
        --teal: #0f766e;
        --navy: #345b78;
    }
    .block-container {max-width: 1440px; padding: 1.4rem 2.2rem 3rem;}
    [data-testid="stAppViewContainer"] {background: var(--canvas);}
    [data-testid="stSidebar"] {background: var(--paper); border-right: 1px solid var(--line);}
    [data-testid="stSidebar"] > div:first-child {padding-top: 1rem;}
    .page-kicker {color: var(--teal); font-size: .72rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;}
    h1 {color: var(--ink); letter-spacing: -.025em; margin: .25rem 0 .25rem;}
    .page-intro {color: var(--muted); margin: 0 0 1.4rem; max-width: 780px;}
    .dataset-strip {
        display: grid; grid-template-columns: 2fr 1fr 1fr 1.2fr; gap: 1rem;
        background: var(--paper); border: 1px solid var(--line);
        border-radius: 6px; padding: .9rem 1rem; margin: .8rem 0 1.5rem;
    }
    .dataset-label {color: var(--muted); font-size: .65rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase;}
    .dataset-value {color: var(--ink); font-size: .9rem; font-weight: 600; margin-top: .2rem;}
    [data-testid="stMetric"] {
        background: var(--paper); border: 1px solid var(--line);
        border-radius: 6px; padding: .85rem 1rem;
    }
    [data-testid="stMetricLabel"] {color: var(--muted);}
    [data-testid="stMetricValue"] {color: var(--ink);}
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--paper); border-color: var(--line); border-radius: 6px;
    }
    [data-testid="stAlert"] {border-radius: 5px;}
    .side-kicker {color: var(--teal); font-size: .68rem; font-weight: 700; letter-spacing: .12em;}
    .side-note {color: var(--muted); font-size: .78rem; line-height: 1.5;}
    .small-note {color: var(--muted); font-size: .78rem;}
    @media (max-width: 800px) {
        .block-container {padding: 1rem 1rem 2rem;}
        .dataset-strip {grid-template-columns: 1fr 1fr;}
        .status-tag {font-size: .58rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="page-kicker">Project 05 &nbsp; / &nbsp; Training and validation</div>',
    unsafe_allow_html=True,
)
st.title("Experiment monitor")
st.markdown(
    '<p class="page-intro">Track learning curves and compare model settings on a held-out benchmark. Validation results are shown separately from the final test split.</p>',
    unsafe_allow_html=True,
)

data, _ = trainer.load_data(scale=True)
_, _, x_val, y_val, x_test, y_test = data
validation_positive = int(y_val.sum())
validation_total = len(y_val)
test_total = len(y_test)

st.markdown(
    f"""
    <div class="dataset-strip">
      <div><div class="dataset-label">Dataset</div><div class="dataset-value">Breast cancer (Wisconsin)</div></div>
      <div><div class="dataset-label">Records</div><div class="dataset-value">569 samples</div></div>
      <div><div class="dataset-label">Features</div><div class="dataset-value">30 numeric</div></div>
      <div><div class="dataset-label">Split</div><div class="dataset-value">60 / 20 / 20</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "runs" not in st.session_state:
    st.session_state.runs = []
if "latest" not in st.session_state:
    st.session_state.latest = None

with st.sidebar:
    st.markdown('<div class="side-kicker">EXPERIMENT SETUP</div>', unsafe_allow_html=True)
    st.header("Training settings")
    st.markdown(
        '<p class="side-note">Submit a configuration to train. Changing controls alone does not start a run.</p>',
        unsafe_allow_html=True,
    )
    with st.form("experiment_settings"):
        st.subheader("Model")
        hidden = st.selectbox("Hidden units", [16, 32, 64, 128, 256], index=4)
        dropout = st.slider("Dropout", 0.0, 0.8, 0.0, 0.1)

        st.subheader("Optimisation")
        weight_decay = st.number_input("Weight decay", 0.0, 1.0, 0.0, 0.01)
        learning_rate = st.selectbox("Learning rate", [0.0001, 0.001, 0.01], index=1)
        epochs = st.slider("Maximum epochs", 20, 300, 150)
        patience = st.slider("Early-stopping patience", 5, 50, 20)
        scaling = st.toggle("Standardise features", value=True)
        early_stopping = st.toggle("Use early stopping", value=False)
        train_button = st.form_submit_button(
            "Run experiment", type="primary", width="stretch"
        )

    st.markdown(
        '<p class="side-note">Source: public scikit-learn benchmark. No patient records are uploaded.</p>',
        unsafe_allow_html=True,
    )


def loss_figure(hist):
    epochs_axis = list(range(1, len(hist["train"]) + 1))
    best_epoch = min(range(len(hist["val"])), key=hist["val"].__getitem__) + 1
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=epochs_axis,
            y=hist["train"],
            mode="lines",
            name="Training",
            line={"color": "#345b78", "width": 2.5},
            hovertemplate="Epoch %{x}<br>Training loss %{y:.3f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=epochs_axis,
            y=hist["val"],
            mode="lines",
            name="Validation",
            line={"color": "#0f766e", "width": 2.5},
            hovertemplate="Epoch %{x}<br>Validation loss %{y:.3f}<extra></extra>",
        )
    )
    figure.add_vline(
        x=best_epoch,
        line={"color": "#9aa7b0", "width": 1, "dash": "dot"},
        annotation_text="Best validation",
        annotation_position="top right",
        annotation_font={"size": 10, "color": "#62717d"},
    )
    figure.update_layout(
        height=330,
        margin={"l": 8, "r": 18, "t": 28, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font={"family": "Arial, sans-serif", "color": "#52616d", "size": 12},
        legend={"orientation": "h", "y": 1.12, "x": 0, "title": None},
        hovermode="x unified",
    )
    figure.update_xaxes(
        title="Epoch",
        showgrid=False,
        linecolor="#dce3e8",
        ticks="outside",
        tickcolor="#dce3e8",
    )
    figure.update_yaxes(
        title="Cross-entropy loss",
        rangemode="tozero",
        gridcolor="#e8edf0",
        zeroline=False,
    )
    return figure


if train_button:
    cfg = {
        "hidden": hidden,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "patience": patience,
        "scaling": scaling,
        "early_stopping": early_stopping,
    }
    progress = st.progress(0.0, text="Training model…")
    chart_placeholder = st.empty()

    def redraw(epoch, hist):
        progress.progress(
            min((epoch + 1) / epochs, 1.0),
            text=f"Training · epoch {epoch + 1} of {epochs}",
        )
        if epoch % 5 == 0:
            chart_placeholder.plotly_chart(
                loss_figure(hist), width="stretch", config={"displayModeBar": False}
            )

    result = trainer.train(cfg, data=trainer.load_data(scaling)[0], on_epoch=redraw)
    st.session_state.latest = result
    st.session_state.runs.append({
        "Run": len(st.session_state.runs) + 1,
        "Hidden units": hidden,
        "Dropout": dropout,
        "Weight decay": weight_decay,
        "Learning rate": learning_rate,
        "Validation recall": f"{result['cancers_found']}/{result['cancers_total']}",
        "Missed positives": result["cancers_total"] - result["cancers_found"],
        "False alarms": result["false_alarms"],
        "Epochs": result["epochs_run"],
        "Curve assessment": result["diagnosis"].title(),
    })
    st.rerun()


latest = st.session_state.latest
if latest:
    recall = latest["cancers_found"] / latest["cancers_total"]
    missed = latest["cancers_total"] - latest["cancers_found"]
    best_epoch = min(range(len(latest["hist"]["val"])), key=latest["hist"]["val"].__getitem__) + 1

    st.subheader("Latest run")
    st.caption(
        f"Validation set · {validation_total} samples · "
        f"{validation_positive} positive cases"
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Recall",
        f"{recall:.1%}",
        help="Positive cases correctly identified / positive cases in validation set.",
    )
    metric_cols[1].metric("Missed", missed, help="Positive validation cases predicted as benign.")
    metric_cols[2].metric("False alarms", latest["false_alarms"])
    metric_cols[3].metric("Val. loss", f"{latest['best_val_loss']:.3f}")

    with st.container(border=True):
        st.markdown("#### Learning curve")
        st.plotly_chart(
            loss_figure(latest["hist"]),
            width="stretch",
            config={"displayModeBar": False},
        )
        st.caption(
            f"Curve assessment: {latest['diagnosis'].title()} · "
            f"Best validation loss at epoch {best_epoch} · "
            f"{latest['epochs_run']} epochs completed"
        )
    with st.container(border=True):
        st.markdown("#### Run configuration")
        config_cols = st.columns(4)
        config_values = [
            ("Hidden units", str(hidden)),
            ("Dropout", f"{dropout:.1f}"),
            ("Weight decay", f"{weight_decay:g}"),
            ("Learning rate", f"{learning_rate:g}"),
        ]
        for column, (label, value) in zip(config_cols, config_values):
            column.caption(label)
            column.markdown(f"**{value}**")
        st.caption(
            "Recall is measured on the validation split. False alarms are benign validation "
            "samples predicted as malignant."
        )
else:
    with st.container(border=True):
        st.subheader("No experiment run yet")
        st.write(
            "Set the model and optimisation controls in the left panel, then select "
            "**Run experiment** to generate a validation curve and performance summary."
        )

st.subheader("Experiment history")
if st.session_state.runs:
    history = pd.DataFrame(st.session_state.runs)
    st.dataframe(
        history,
        width="stretch",
        hide_index=True,
        column_config={
            "Run": st.column_config.NumberColumn(format="%02d"),
            "Validation recall": st.column_config.TextColumn(help="Positive cases found / positive cases in validation set"),
            "Missed positives": st.column_config.NumberColumn(help="Positive validation cases predicted as benign"),
        },
    )
    st.download_button(
        "Export run history",
        history.to_csv(index=False).encode("utf-8"),
        file_name="training-experiments.csv",
        mime="text/csv",
    )
else:
    st.caption("Completed runs will appear here for side-by-side comparison.")

if latest:
    with st.expander("Inspect a held-out test sample"):
        st.caption(
            f"Test split · {test_total} samples. Benchmark records are de-identified "
            "research data, not current patients or clinical assessments."
        )
        sample = st.slider("Sample index", 0, test_total - 1, 0)
        data, _ = trainer.load_data(scaling)
        x_test, y_test = data[4], data[5]
        with torch.no_grad():
            probability = torch.softmax(latest["model"](x_test), dim=1)[sample, 1].item()
        prediction = int(probability >= 0.5)
        sample_cols = st.columns(3)
        sample_cols[0].metric("Predicted malignant probability", f"{probability:.1%}")
        sample_cols[1].metric("Model label", "Malignant" if prediction else "Benign")
        sample_cols[2].metric("Benchmark label", "Malignant" if int(y_test[sample]) else "Benign")

st.divider()
st.caption(
    "Educational benchmark only · Not intended for diagnosis, screening, triage, or treatment decisions."
)
