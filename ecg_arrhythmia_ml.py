"""
Streamlit web app: ECG Arrhythmia Classification & Real-Time Alert System
Run locally:   streamlit run streamlit_app.py
Deploy free:   Streamlit Community Cloud (see STREAMLIT_DEPLOY.md)
"""
import io
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

import ecg_core as E

st.set_page_config(page_title="ECG Arrhythmia Alert System", page_icon="🫀", layout="wide")

SCENARIOS = {
    "Full demo (about 5 minutes)": E.DEMO_SCHEDULE,
    "Normal rhythm": [("normal", 45)],
    "PVC (extra ventricular beats)": [("normal", 20), ("pvc", 55), ("normal", 10)],
    "AFib (irregular rhythm)": [("normal", 20), ("afib", 50), ("normal", 10)],
    "Tachycardia (fast heart)": [("normal", 20), ("tachy", 80), ("normal", 10)],
    "Bradycardia (slow heart)": [("normal", 15), ("brady", 30), ("normal", 8)],
}
BANNER = {"NORMAL": "success", "PVC": "error", "AFIB": "error", "TACHYCARDIA": "warning", "BRADYCARDIA": "warning"}


@st.cache_resource(show_spinner="Training the beat classifier on synthetic patients (about 20 s, once)...")
def get_model(n_train: int, seed: int):
    rng = np.random.default_rng(seed)
    train = [E.make_record(rng, E.make_patient(rng), E.training_schedule(rng)) for _ in range(n_train)]
    test = [E.make_record(rng, E.make_patient(rng), E.training_schedule(rng)) for _ in range(8)]
    Xtr, ytr, _, _ = E.build_dataset(train, "detect")
    Xte, yte, _, st_te = E.build_dataset(test, "detect")
    model = RandomForestClassifier(n_estimators=150, min_samples_leaf=2, class_weight="balanced_subsample",
                                   random_state=seed, n_jobs=-1)
    model.fit(np.nan_to_num(Xtr), ytr)
    pred = model.predict(np.nan_to_num(Xte))
    return model, yte, pred, st_te


def live_figure(rec, beats, t_now, win=12.0):
    fs = rec["fs"]
    i1 = int(t_now * fs); i0 = max(0, i1 - int(win * fs))
    xp = E.preprocess(rec["signal"][i0:i1], fs)
    tt = np.arange(i0, i1) / fs
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.plot(tt, xp, color="#1f4e79", lw=0.9)
    colors = dict(N="#2ca02c", S="#e08a00", V="#d62728")
    for bt, bl in beats:
        if tt[0] <= bt <= tt[-1]:
            k = int(round((bt - tt[0]) * fs))
            if 0 <= k < len(xp):
                ax.scatter([bt], [xp[k] + 0.12], color=colors[bl], marker="v", s=30)
                ax.text(bt, xp[k] + 0.25, bl, ha="center", fontsize=8, color=colors[bl])
    ax.set_xlim(tt[0], tt[-1]); ax.set_xlabel("Time (s)"); ax.set_ylabel("ECG (a.u.)")
    fig.tight_layout()
    return fig


def timeline_image(rec, beats, timeline):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    E.plot_timeline(rec, beats, timeline, path)
    return path


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Settings")
n_train = st.sidebar.slider("Training patients", 8, 30, 16, help="More patients = slightly better model, slower start")
seed = st.sidebar.number_input("Random seed", 0, 999, 7)
st.sidebar.info("The model is trained on synthetic ECG made by this app. "
                "It is an educational project, not a medical device.")
model, yte, pred, det_stats = get_model(int(n_train), int(seed))

st.title("🫀 ECG Arrhythmia Classification & Real-Time Alert System")
tab1, tab2, tab3, tab4 = st.tabs(["How it works", "Model accuracy", "Live monitor", "Your own ECG (CSV)"])

# ---------------------------------------------------------------- tab 1
with tab1:
    st.markdown("""
**What this does:** it watches an ECG signal, finds every heartbeat, labels each beat, and raises an alert when the rhythm looks dangerous.

1. **Filter** the signal (0.5 to 40 Hz) to remove drift and noise.
2. **Find heartbeats** (R-peaks) with a Pan-Tompkins style detector.
3. **Cut a window** around each beat and measure its features (timing, width, shape).
4. **Classify** each beat with a Random Forest: **N** normal, **S** atrial premature, **V** ventricular premature.
5. **Check the rhythm** over the last 10 seconds:
   - 2 or more V beats gives a **PVC** alert
   - very uneven beat spacing gives an **AFib** alert
   - heart rate above 100 gives **tachycardia**, below 50 gives **bradycardia**

Use the **Live monitor** tab to watch it work, or **Your own ECG** to test a CSV file.
""")
    st.warning("Not for medical use. Trained on synthetic data only.")

# ---------------------------------------------------------------- tab 2
with tab2:
    classes = sorted(set(yte))
    rep = classification_report(yte, pred, labels=classes, output_dict=True, zero_division=0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Beat accuracy (unseen patients)", f"{rep['accuracy']*100:.1f}%")
    c2.metric("R-peak sensitivity", f"{det_stats['sensitivity']*100:.1f}%")
    c3.metric("R-peak precision", f"{det_stats['ppv']*100:.1f}%")
    left, right = st.columns(2)
    df = pd.DataFrame({c: {k: round(rep[c][k], 3) for k in ("precision", "recall", "f1-score", "support")} for c in classes}).T
    left.subheader("Per-class results")
    left.dataframe(df)
    left.caption("N normal, S atrial premature, V ventricular premature. S is the hardest class because it looks like a normal beat.")
    cm = confusion_matrix(yte, pred, labels=classes)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.tight_layout()
    right.subheader("Confusion matrix")
    right.pyplot(fig)

# ---------------------------------------------------------------- tab 3
with tab3:
    st.write("Pick a patient scenario and press start. The system sees the ECG one second at a time, like a live sensor.")
    c1, c2 = st.columns([2, 1])
    scen = c1.selectbox("Scenario", list(SCENARIOS))
    delay = c2.slider("Seconds of waiting per simulated second", 0.0, 0.5, 0.05, 0.05)
    if st.button("Start monitoring", type="primary"):
        rng = np.random.default_rng(int(seed) + 11)
        rec = E.make_record(rng, E.make_patient(rng), SCENARIOS[scen])
        total = len(rec["signal"]) / rec["fs"]
        prog = st.progress(0.0)
        banner = st.empty(); metrics = st.empty(); plot = st.empty()
        timeline, log, beats = [], [], []
        for t, beats, state, hr, nV, irr, entry in E.realtime_steps(model, rec):
            timeline.append((t, state)); entry and log.append(entry)
            getattr(banner, BANNER[state])(f"System status: {state}" + ("  (notification sent, simulated)" if state != "NORMAL" else ""))
            m1, m2, m3, m4 = metrics.columns(4)
            m1.metric("Time", f"{t:.0f} s"); m2.metric("Heart rate", "…" if hr is None else f"{hr:.0f} bpm")
            m3.metric("V beats in 10 s", nV); m4.metric("Irregularity", f"{irr:.2f}")
            fig = live_figure(rec, beats, t); plot.pyplot(fig); plt.close(fig)
            prog.progress(min(1.0, t / total))
            time.sleep(delay)
        prog.empty()
        st.success("Monitoring finished.")
        if log:
            st.subheader("Alert log")
            st.dataframe(pd.DataFrame(log))
        if rec["segments"]:
            st.subheader("Summary: real rhythm (shaded) vs what the system reported")
            st.image(timeline_image(rec, beats, timeline))
            st.dataframe(E.evaluate_alerts(rec, timeline))

# ---------------------------------------------------------------- tab 4
with tab4:
    st.write("Upload a CSV with one column of ECG samples (a header row is fine). Best for single-lead ECG, at least 15 seconds.")
    sample_rng = np.random.default_rng(3)
    sample = E.make_record(sample_rng, E.make_patient(sample_rng), [("normal", 25), ("pvc", 45)], artifacts=False)
    sample_csv = pd.DataFrame({"ecg": sample["signal"][:60 * E.FS]}).to_csv(index=False)
    st.download_button("Download a sample CSV (250 Hz)", sample_csv, "sample_ecg_250hz.csv", "text/csv")
    up = st.file_uploader("ECG CSV file", type=["csv", "txt"])
    fs_in = st.number_input("Sampling rate of your file (Hz)", 50, 1000, 250)
    if up is not None:
        try:
            raw = pd.read_csv(up)
            num = raw.select_dtypes(include=[np.number])
            if num.shape[1] == 0:
                raw = pd.read_csv(io.BytesIO(up.getvalue()), header=None); num = raw.select_dtypes(include=[np.number])
            col = st.selectbox("Column to use", list(num.columns)) if num.shape[1] > 1 else num.columns[0]
            x = num[col].dropna().to_numpy(float)[: int(300 * fs_in)]
            if len(x) < 15 * fs_in:
                st.error("Need at least 15 seconds of ECG.")
            else:
                if fs_in != E.FS:
                    n_new = int(len(x) / fs_in * E.FS)
                    x = np.interp(np.linspace(0, len(x) - 1, n_new), np.arange(len(x)), x)
                x = (x - np.median(x)) / (np.percentile(x, 99.5) - np.percentile(x, 0.5) + 1e-9)
                rec = dict(signal=x, fs=E.FS, r_samples=np.array([]), labels=np.array([]), segments=[])
                beats, timeline, log = E.realtime_stream(model, rec, verbose=False)
                labs = pd.Series([b[1] for b in beats]).value_counts()
                c1, c2, c3 = st.columns(3)
                c1.metric("Beats found", len(beats)); c2.metric("V beats", int(labs.get("V", 0))); c3.metric("S beats", int(labs.get("S", 0)))
                fig = live_figure(rec, beats, len(x) / E.FS, win=min(30.0, len(x) / E.FS))
                st.pyplot(fig); plt.close(fig)
                st.subheader("Alerts raised")
                st.dataframe(pd.DataFrame(log) if log else pd.DataFrame({"result": ["No alerts: rhythm looked normal"]}))
        except Exception as e:
            st.error(f"Could not read that file: {e}")
