#!/usr/bin/env python3
"""
Automated ECG Arrhythmia Classification & Real-Time Alert System
Pure software version (no hardware needed).

Pipeline
  ECG signal -> band-pass filter -> R-peak detection -> beat segmentation
  -> feature extraction -> Random Forest beat classifier (N / S / V)
  -> rhythm rules on a sliding 10 s window -> real-time alerts

Modes
  python ecg_arrhythmia_ml.py            synthetic ECG (works offline, default)
  python ecg_arrhythmia_ml.py --mitbih   real MIT-BIH data via wfdb (needs internet)
"""
import argparse
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

FS = 250                      # sampling rate of the synthetic ECG (Hz)
PRE_S, POST_S = 0.25, 0.40    # beat window around the R-peak (seconds)

# ----------------------------------------------------------------------------
# 1. Synthetic ECG generator (replaces the ECG machine)
# ----------------------------------------------------------------------------
def _g(t, c, a, s):
    return a * np.exp(-((t - c) / s) ** 2)


def make_patient(rng):
    """Every synthetic patient has a slightly different heart and noise level."""
    return dict(amp=rng.uniform(0.7, 1.4), rs=rng.uniform(0.85, 1.25),
                ta=rng.uniform(0.6, 1.4), pa=rng.uniform(0.7, 1.4),
                rr=rng.uniform(0.70, 1.00), noise=rng.uniform(0.02, 0.08),
                v_wide=rng.uniform(0.9, 1.3))


def beat_wave(t, kind, p, afib=False, j=None):
    """Shape of one heartbeat. t = seconds relative to the R-peak.
    j = small beat-to-beat random variation (amplitude, widths, P-wave size)."""
    j = j or dict(a=1.0, rs=1.0, ta=1.0, pf=1.0, vw=1.0, va=1.0)
    a = p["amp"] * j["a"]
    if kind == "V":      # premature ventricular beat: wide, no P wave, inverted T
        return (_g(t, -0.01, 1.1 * a * j["va"], 0.035 * p["v_wide"] * j["vw"])
                - _g(t, 0.22, 0.35 * a * p["ta"], 0.06))
    pa = 0.0 if afib else 0.12 * p["pa"] * a * j["pf"]
    pc = -0.11
    if kind == "S":      # premature atrial beat: narrow QRS, smaller / different P wave
        pa, pc = pa * 0.7, -0.12
    return (_g(t, pc, pa, 0.025) + _g(t, -0.025, -0.10 * a, 0.008)
            + _g(t, 0.0, 1.0 * a, 0.010 * p["rs"] * j["rs"]) + _g(t, 0.025, -0.20 * a, 0.010)
            + _g(t, 0.20, 0.25 * a * p["ta"] * j["ta"], 0.04))


def make_record(rng, patient, schedule, fs=FS, artifacts=True):
    """schedule = [(segment_name, n_beats), ...]
    segment names: normal, pvc, pac, tachy, brady, afib"""
    kinds, afibs, rr_pre, seg_of, jits = [], [], [], [], []
    base = patient["rr"]
    seg_info = []
    override = None
    tcum = 1.0
    for si, (name, nb) in enumerate(schedule):
        first = len(kinds)
        for _ in range(nb):
            kind, af = "N", False
            resp = 0.04 * np.sin(2 * np.pi * 0.25 * tcum)       # breathing changes the heart rate a little
            if name in ("normal", "pac", "pvc"):
                rr = base + resp + rng.normal(0, 0.02)
            elif name == "tachy":
                rr = 0.45 + 0.5 * resp + rng.normal(0, 0.012)
            elif name == "brady":
                rr = rng.uniform(1.35, 1.5) + resp
            else:                                               # afib: irregularly irregular
                rr, af = rng.uniform(0.35, 1.10), True
            j = dict(a=rng.normal(1, 0.06), rs=rng.normal(1, 0.08), ta=rng.normal(1, 0.15),
                     pf=rng.normal(1, 0.1), vw=rng.uniform(0.8, 1.4), va=rng.uniform(0.7, 1.3))
            if override is not None:
                rr, override = override, None
            elif name == "pvc" and rng.random() < 0.22 and (not kinds or kinds[-1] == "N"):
                kind, rr, override = "V", base * rng.uniform(0.5, 0.75), base * rng.uniform(1.2, 1.45)
            elif name == "pac" and rng.random() < 0.20 and (not kinds or kinds[-1] == "N"):
                kind, rr, override = "S", base * rng.uniform(0.6, 0.9), base * rng.uniform(1.0, 1.15)
                j["pf"] = rng.uniform(0.1, 0.9)
            kinds.append(kind); afibs.append(af); rr_pre.append(rr); seg_of.append(si); jits.append(j)
            tcum += rr
        seg_info.append(dict(name=name, first=first, last=len(kinds) - 1))
    rr_pre[0] = 0.0
    rt = 1.0 + np.cumsum(rr_pre)                    # R-peak times in seconds
    n = int((rt[-1] + 1.5) * fs)
    t = np.arange(n) / fs
    sig = np.zeros(n)
    for k, r in enumerate(rt):
        i0, i1 = max(0, int((r - 0.5) * fs)), min(n, int((r + 0.5) * fs))
        sig[i0:i1] += beat_wave(t[i0:i1] - r, kinds[k], patient, afibs[k], jits[k])
    ph = rng.uniform(0, 6.28, 2)                    # realistic noise
    sig += 0.15 * np.sin(2 * np.pi * 0.25 * t + ph[0]) + 0.08 * np.sin(2 * np.pi * 0.10 * t + ph[1])
    sig += 0.03 * np.sin(2 * np.pi * 50 * t)
    sig += rng.normal(0, patient["noise"], n)
    for _ in range(max(1, int(n / fs / 90)) if artifacts else 0):   # short motion-artifact bursts
        c = rng.integers(int(2 * fs), n - int(2 * fs)); w = int(0.5 * fs)
        sig[c:c + w] += rng.normal(0, 0.15, w) + 0.3 * np.sin(2 * np.pi * 3 * t[c:c + w])
    for s in seg_info:
        s["start_s"], s["end_s"] = float(rt[s["first"]]), float(rt[s["last"]])
    return dict(signal=sig, fs=fs, r_samples=np.round(rt * fs).astype(int),
                labels=np.array(kinds), segments=seg_info)


def training_schedule(rng):
    segs = [("normal", 60), ("pvc", 100), ("pac", 100), ("tachy", 40),
            ("afib", 60), ("brady", 30), ("normal", 40)]
    rng.shuffle(segs)
    return segs


DEMO_SCHEDULE = [("normal", 40), ("pvc", 50), ("normal", 15), ("pac", 35), ("tachy", 60),
                 ("normal", 15), ("afib", 70), ("brady", 35), ("normal", 20)]

# ----------------------------------------------------------------------------
# 2. Preprocessing and R-peak detection (Pan-Tompkins style)
# ----------------------------------------------------------------------------
def bandpass(x, lo, hi, fs, order=3):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def preprocess(x, fs):
    """0.5-40 Hz band-pass: removes baseline wander, 50 Hz mains and muscle noise."""
    return bandpass(x, 0.5, 40, fs)


def detect_r_peaks(xp, fs):
    f = bandpass(xp, 5, 15, fs, 2)
    s = np.gradient(f) ** 2
    w = max(1, int(0.15 * fs))
    integ = np.convolve(s, np.ones(w) / w, mode="same")
    thr = 0.12 * np.percentile(integ, 99.5)
    pk, _ = find_peaks(integ, height=thr, distance=int(0.22 * fs))
    half = int(0.10 * fs)
    out = []
    for p in pk:
        lo, hi = max(0, p - half), min(len(xp), p + half)
        out.append(lo + int(np.argmax(xp[lo:hi])))
    return np.unique(out)

# ----------------------------------------------------------------------------
# 3. Beat features
# ----------------------------------------------------------------------------
def beat_features(xp, peaks, fs):
    pre, post = int(PRE_S * fs), int(POST_S * fs)
    n = len(peaks)
    rr = np.diff(peaks) / fs
    rows = []
    nfeat = 13 + len(range(0, pre + post, 4))
    for i, p in enumerate(peaks):
        if n < 3 or p - pre < 0 or p + post > len(xp):
            rows.append(np.full(nfeat, np.nan)); continue
        b = xp[p - pre:p + post]
        bn = b / (np.max(np.abs(b)) + 1e-9)
        rp = rr[i - 1] if i > 0 else rr[0]
        rn = rr[i] if i < n - 1 else rr[-1]
        hist = rr[max(0, i - 8):i]
        lm = np.median(hist) if len(hist) else rp
        c = int(np.argmax(np.abs(bn)))
        m = np.abs(bn) >= 0.5
        l = r = c
        while l > 0 and m[l - 1]: l -= 1
        while r < len(m) - 1 and m[r + 1]: r += 1
        width = (r - l + 1) / fs
        tseg = bn[pre + int(0.12 * fs): pre + int(0.36 * fs)]
        pseg = bn[pre - int(0.20 * fs): pre - int(0.06 * fs)]
        f = [rp, rn, rp / lm, rn / lm, lm, width, b[pre], tseg.min(), tseg.max(),
             pseg.min(), pseg.max(), bn.std(), np.abs(bn).sum() / fs]
        rows.append(np.array(f + list(bn[::4])))
    return np.array(rows)


def build_dataset(records, mode="detect", tol=0.08):
    X, y, pid = [], [], []
    tp = fp = fn = 0
    for k, rec in enumerate(records):
        fs = rec["fs"]
        xp = preprocess(rec["signal"], fs)
        tr, labs = rec["r_samples"], rec["labels"]
        if mode == "detect":
            pk = detect_r_peaks(xp, fs)
            d = np.abs(tr[None, :] - pk[:, None])
            near, dist = d.argmin(axis=1), d.min(axis=1)
            ok = dist <= tol * fs
            tp += int(ok.sum()); fp += int((~ok).sum())
            fn += len(tr) - len(np.unique(near[ok]))
            lab = np.where(ok, labs[near], "")
        else:
            pk, lab, ok = tr, labs, np.ones(len(tr), bool)
        F = beat_features(xp, pk, fs)
        keep = ok & ~np.isnan(F).any(axis=1)
        X.append(F[keep]); y += list(np.asarray(lab)[keep]); pid += [k] * int(keep.sum())
    stats = dict(sensitivity=tp / max(1, tp + fn), ppv=tp / max(1, tp + fp)) if mode == "detect" else {}
    return np.vstack(X), np.array(y), np.array(pid), stats

# ----------------------------------------------------------------------------
# 4. Real-time alert engine
# ----------------------------------------------------------------------------
def assess(beats, tcur, span=10.0):
    """Rhythm rules over the last `span` seconds of classified beats."""
    B = [b for b in beats if tcur - span <= b[0] <= tcur]
    if len(B) < 4:
        return "NORMAL", None, 0, 0.0
    ts = np.array([b[0] for b in B]); lab = [b[1] for b in B]
    nV = lab.count("V")
    nn = np.array([ts[k + 1] - ts[k] for k in range(len(B) - 1) if lab[k] == "N" and lab[k + 1] == "N"])
    hr = 60.0 / nn.mean() if len(nn) >= 3 else None
    irr = float(np.median(np.abs(np.diff(nn))) / np.median(nn)) if len(nn) >= 6 else 0.0
    if irr > 0.12:
        state = "AFIB"
    elif nV >= 2:
        state = "PVC"
    elif hr is not None and hr > 100:
        state = "TACHYCARDIA"
    elif hr is not None and hr < 50:
        state = "BRADYCARDIA"
    else:
        state = "NORMAL"
    return state, hr, nV, irr


def realtime_steps(model, rec, win=12.0, step=1.0, margin=0.9):
    """Generator: feeds the record 1 second at a time, like a live sensor.
    Yields (t, beats, state, hr, n_v, irregularity, new_log_entry_or_None)."""
    fs, sig = rec["fs"], rec["signal"]
    beats, seen = [], []
    current, pending, count = "NORMAL", None, 0
    for tcur in np.arange(win, len(sig) / fs + 1e-9, step):
        i1 = int(tcur * fs); i0 = i1 - int(win * fs)
        xp = preprocess(sig[i0:i1], fs)
        pk = detect_r_peaks(xp, fs)
        if len(pk) >= 4:
            F = beat_features(xp, pk, fs)
            for j in range(1, len(pk) - 1):
                tj = pk[j] / fs
                if tj < margin or tj > win - margin or np.isnan(F[j]).any():
                    continue
                g = i0 + pk[j]
                if any(abs(g - s) < 0.15 * fs for s in seen[-30:]):
                    continue
                beats.append((g / fs, model.predict(F[j:j + 1])[0])); seen.append(g)
        state, hr, nV, irr = assess(beats, tcur)
        entry = None
        if state == current:
            pending, count = None, 0
        elif state == pending:                       # must hold for 2 checks (fewer false alarms)
            count += 1
            if count >= 2:
                current, pending, count = state, None, 0
                msg = (f"ALERT: {state} detected -> notification sent (simulated)"
                       if state != "NORMAL" else "Rhythm back to NORMAL")
                entry = dict(time_s=round(float(tcur), 1), state=state,
                             heart_rate=None if hr is None else round(hr), v_beats_10s=nV,
                             irregularity=round(irr, 2), message=msg)
        else:
            pending, count = state, 1
        yield float(tcur), beats, current, hr, nV, irr, entry


def realtime_stream(model, rec, win=12.0, step=1.0, margin=0.9, verbose=True):
    """Runs the whole record and returns (beats, timeline, alert_log)."""
    beats, timeline, log = [], [], []
    for t, beats, current, hr, nV, irr, entry in realtime_steps(model, rec, win, step, margin):
        timeline.append((t, current))
        if entry:
            log.append(entry)
            if verbose:
                print(f"[{t:6.1f}s] {entry['message']}   (HR={entry['heart_rate']}, V in 10s={nV}, irr={irr:.2f})")
    return beats, timeline, log

# ----------------------------------------------------------------------------
# 5. Plots and evaluation
# ----------------------------------------------------------------------------
EXPECT = dict(normal="NORMAL", pac="NORMAL", pvc="PVC", tachy="TACHYCARDIA",
              brady="BRADYCARDIA", afib="AFIB")


def evaluate_alerts(rec, timeline):
    rows = []
    for s in rec["segments"]:
        exp = EXPECT[s["name"]]
        inside = [(t, st) for t, st in timeline if s["start_s"] <= t <= s["end_s"]]
        hit = [t for t, st in inside if st == exp]
        settled = [st for t, st in inside if t >= s["start_s"] + 15]
        rows.append(dict(segment=s["name"], start_s=round(s["start_s"]), end_s=round(s["end_s"]),
                         expected=exp,
                         detection_delay_s=round(hit[0] - s["start_s"], 1) if hit else None,
                         correct_after_15s=round(100 * np.mean([st == exp for st in settled]), 1) if settled else None))
    return pd.DataFrame(rows)


def plot_confusion(y_true, y_pred, classes, path):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Beat classification on unseen patients")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]*100:.1f}%", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_timeline(rec, beats, timeline, path):
    colors = dict(normal="#cfe8dc", pac="#d9ebf7", pvc="#f9d9d9", tachy="#fde9c8",
                  brady="#e7defa", afib="#f6cfe6")
    fig, ax = plt.subplots(2, 1, figsize=(12, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    for s in rec["segments"]:
        for a in ax:
            a.axvspan(s["start_s"], s["end_s"], color=colors[s["name"]], alpha=0.9, lw=0)
        ax[0].text((s["start_s"] + s["end_s"]) / 2, 168, s["name"].upper(), ha="center", fontsize=8)
    bt = np.array([b[0] for b in beats]); bl = np.array([b[1] for b in beats])
    hr = 60 / np.diff(bt)
    ax[0].plot(bt[1:], hr, color="#444", lw=0.8, label="beat-to-beat heart rate")
    for lab, mk, c in (("V", "x", "#d62728"), ("S", "^", "#e08a00")):
        m = np.where(bl[1:] == lab)[0]
        ax[0].scatter(bt[1:][m], hr[m], marker=mk, color=c, s=28, label=f"{lab} beat", zorder=3)
    ax[0].axhline(100, ls=":", c="gray"); ax[0].axhline(50, ls=":", c="gray")
    ax[0].set_ylim(20, 180); ax[0].set_ylabel("Heart rate (bpm)")
    ax[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8, ncol=3, frameon=False)
    ax[0].set_title("Real-time monitoring of a simulated patient (shaded = true rhythm)", pad=26)
    order = ["NORMAL", "PVC", "TACHYCARDIA", "BRADYCARDIA", "AFIB"]
    t = [x[0] for x in timeline]; v = [order.index(x[1]) for x in timeline]
    ax[1].step(t, v, where="post", color="#b00020", lw=1.8)
    ax[1].set_yticks(range(len(order))); ax[1].set_yticklabels(order, fontsize=8)
    ax[1].set_ylabel("System alert"); ax[1].set_xlabel("Time (s)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_zoom(rec, beats, path, span=10.0):
    fs = rec["fs"]
    xp = preprocess(rec["signal"], fs)
    picks = [next(s for s in rec["segments"] if s["name"] == n) for n in ("normal", "pvc", "afib")]
    fig, ax = plt.subplots(3, 1, figsize=(12, 7))
    for a, s in zip(ax, picks):
        t0 = (s["start_s"] + s["end_s"]) / 2 - span / 2
        i0, i1 = int(t0 * fs), int((t0 + span) * fs)
        tt = np.arange(i0, i1) / fs
        a.plot(tt, xp[i0:i1], color="#1f4e79", lw=0.9)
        for bt, bl in beats:
            if t0 <= bt <= t0 + span:
                y = xp[int(round(bt * fs))]
                c = dict(N="#2ca02c", S="#e08a00", V="#d62728")[bl]
                a.scatter([bt], [y + 0.12], color=c, s=26, marker="v")
                a.text(bt, y + 0.25, bl, ha="center", fontsize=8, color=c)
        a.set_title(f"{s['name'].upper()} segment: filtered ECG with predicted beat labels", fontsize=10)
        a.set_ylabel("mV (a.u.)")
    ax[-1].set_xlabel("Time (s)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

# ----------------------------------------------------------------------------
# 6. Optional: real MIT-BIH data (needs internet + `pip install wfdb`)
# ----------------------------------------------------------------------------
AAMI = {**{s: "N" for s in "NLReje"}, **{s: "S" for s in "AaJS"}, **{s: "V" for s in "VE"}, "F": "F"}
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]


def load_mitbih(records, seconds=None):
    import wfdb
    out = []
    for r in records:
        sampto = None if seconds is None else int(seconds * 360)
        rec = wfdb.rdrecord(str(r), pn_dir="mitdb", sampto=sampto)
        ann = wfdb.rdann(str(r), "atr", pn_dir="mitdb", sampto=sampto)
        keep = [i for i, s in enumerate(ann.symbol) if s in AAMI]
        out.append(dict(signal=rec.p_signal[:, 0], fs=int(rec.fs),
                        r_samples=np.array(ann.sample)[keep],
                        labels=np.array([AAMI[ann.symbol[i]] for i in keep]), segments=[]))
        print(f"loaded record {r}: {len(keep)} beats")
    return out

# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-patients", type=int, default=24)
    ap.add_argument("--test-patients", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results")
    ap.add_argument("--mitbih", action="store_true", help="use real MIT-BIH data (inter-patient split)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    metrics = {}

    if args.mitbih:
        print("Loading MIT-BIH (DS1 = train, DS2 = test)...")
        train_recs, test_recs = load_mitbih(DS1), load_mitbih(DS2)
        mode, demo = "annot", None
    else:
        print("Generating synthetic patients...")
        train_recs = [make_record(rng, make_patient(rng), training_schedule(rng)) for _ in range(args.train_patients)]
        test_recs = [make_record(rng, make_patient(rng), training_schedule(rng)) for _ in range(args.test_patients)]
        mode = "detect"
        demo = make_record(np.random.default_rng(args.seed + 1), make_patient(np.random.default_rng(args.seed + 1)), DEMO_SCHEDULE)

    Xtr, ytr, _, st_tr = build_dataset(train_recs, mode)
    Xte, yte, _, st_te = build_dataset(test_recs, mode)
    classes = sorted(set(ytr))
    print(f"Train beats: {len(ytr)}  Test beats (unseen patients): {len(yte)}")
    print("Train class counts:", dict(zip(*np.unique(ytr, return_counts=True))))
    if st_te:
        print(f"R-peak detector on test patients: sensitivity {st_te['sensitivity']*100:.1f}%, precision {st_te['ppv']*100:.1f}%")
        metrics["r_peak_detector"] = {k: round(v, 4) for k, v in st_te.items()}

    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                                   random_state=args.seed, n_jobs=-1)
    model.fit(np.nan_to_num(Xtr), ytr)
    pred = model.predict(np.nan_to_num(Xte))
    rep = classification_report(yte, pred, labels=classes, digits=3, output_dict=True, zero_division=0)
    print("\nPatient-wise test results (unseen patients):")
    print(classification_report(yte, pred, labels=classes, digits=3, zero_division=0))
    metrics["beat_classifier"] = {"accuracy": round(rep["accuracy"], 4),
                                  **{c: {k: round(rep[c][k], 3) for k in ("precision", "recall", "f1-score")} for c in classes}}
    plot_confusion(yte, pred, classes, os.path.join(args.out, "confusion_matrix.png"))
    joblib.dump(model, os.path.join(args.out, "beat_classifier.joblib"))

    if demo is not None:
        print("\n--- Real-time alert simulation (1 s of ECG at a time) ---")
        beats, timeline, log = realtime_stream(model, demo)
        pd.DataFrame(log).to_csv(os.path.join(args.out, "alert_log.csv"), index=False)
        ev = evaluate_alerts(demo, timeline)
        ev.to_csv(os.path.join(args.out, "alert_evaluation.csv"), index=False)
        print("\nAlert evaluation per rhythm segment:")
        print(ev.to_string(index=False))
        plot_timeline(demo, beats, timeline, os.path.join(args.out, "alert_timeline.png"))
        plot_zoom(demo, beats, os.path.join(args.out, "beats_zoom.png"))
        metrics["alerts"] = ev.to_dict(orient="records")
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nAll outputs saved in ./{args.out}/")


if __name__ == "__main__":
    main()
