"""Student Result Early-Warning System — Streamlit + XGBoost mini project.

Run:  streamlit run app.py
"""
import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from generate_data import generate

DATA_PATH = "students.csv"
FEATURES = ["attendance", "assignment_marks", "previous_exam_marks"]
LABELS = {
    "attendance": "Attendance (%)",
    "assignment_marks": "Assignment marks (/100)",
    "previous_exam_marks": "Previous exam marks (/100)",
}

st.set_page_config(page_title="Student Early-Warning System", page_icon="🎓", layout="wide")


# ---------- data ----------
@st.cache_data
def load_default() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        generate().to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)


def validate(df: pd.DataFrame) -> str | None:
    missing = [c for c in FEATURES + ["passed"] if c not in df.columns]
    if missing:
        return f"CSV is missing columns: {', '.join(missing)}"
    if df["passed"].nunique() < 2:
        return "The 'passed' column needs both 0 (failed) and 1 (passed) values."
    return None


# ---------- model ----------
def train(df: pd.DataFrame, n_estimators: int, max_depth: int, learning_rate: float):
    X, y = df[FEATURES], df["passed"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return {
        "model": model,
        "accuracy": accuracy_score(y_te, pred),
        "cm": confusion_matrix(y_te, pred),
        "n_train": len(X_tr),
        "n_test": len(X_te),
    }


# ---------- sidebar ----------
st.sidebar.header("Data & model settings")
upload = st.sidebar.file_uploader("Upload your own CSV (optional)", type="csv")
st.sidebar.caption(
    "Columns needed: attendance, assignment_marks, previous_exam_marks, passed (1/0)"
)
n_estimators = st.sidebar.slider("Number of trees", 20, 400, 150, 10)
max_depth = st.sidebar.slider("Tree depth", 1, 8, 3)
learning_rate = st.sidebar.select_slider(
    "Learning rate", options=[0.01, 0.05, 0.1, 0.2, 0.3], value=0.1
)

df = load_default()
if upload is not None:
    custom = pd.read_csv(upload)
    err = validate(custom)
    if err:
        st.sidebar.error(err)
    else:
        df = custom
        st.sidebar.success(f"Using uploaded data ({len(df)} rows)")

# ---------- header ----------
st.title("🎓 Student Result Early-Warning System")
st.write(
    "Learns from past student records with **XGBoost** and flags students who may "
    "need extra help. It's an estimate based on data — not a guarantee."
)

tab_data, tab_train, tab_predict = st.tabs(["1 · Past data", "2 · Train model", "3 · Predict"])

# ---------- tab 1: data ----------
with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Students", len(df))
    c2.metric("Passed", f"{df.passed.mean():.0%}")
    c3.metric("Need help (failed)", f"{1 - df.passed.mean():.0%}")

    st.dataframe(df, width="stretch", height=300)

    st.subheader("Average by outcome")
    avg = df.groupby(df["passed"].map({1: "Passed", 0: "Failed"}))[FEATURES].mean().round(1)
    st.dataframe(avg.rename(columns=LABELS), width="stretch")

    st.subheader("Attendance vs previous exam marks")
    plot_df = df.assign(Outcome=df["passed"].map({1: "Passed", 0: "Failed"}))
    st.scatter_chart(
        plot_df,
        x="attendance",
        y="previous_exam_marks",
        color="Outcome",
    )

# ---------- tab 2: train ----------
with tab_train:
    st.write(
        "XGBoost builds many small decision trees, each one correcting the mistakes "
        "of the trees before it. 80% of the data is used to train, 20% to test."
    )
    if st.button("Train model", type="primary"):
        with st.spinner("Training..."):
            st.session_state["result"] = train(df, n_estimators, max_depth, learning_rate)

    res = st.session_state.get("result")
    if res:
        a, b, c = st.columns(3)
        a.metric("Test accuracy", f"{res['accuracy']:.1%}")
        b.metric("Training rows", res["n_train"])
        c.metric("Test rows", res["n_test"])

        left, right = st.columns(2)
        with left:
            st.subheader("Confusion matrix")
            cm = pd.DataFrame(
                res["cm"],
                index=["Actual: Failed", "Actual: Passed"],
                columns=["Predicted: Failed", "Predicted: Passed"],
            )
            st.dataframe(cm, width="stretch")
        with right:
            st.subheader("What matters most")
            imp = pd.Series(
                res["model"].feature_importances_,
                index=[LABELS[f] for f in FEATURES],
            ).sort_values()
            st.bar_chart(imp, horizontal=True)
    else:
        st.info("Click **Train model** to continue.")

# ---------- tab 3: predict ----------
with tab_predict:
    res = st.session_state.get("result")
    if not res:
        st.info("Train the model in tab 2 first.")
    else:
        st.write("Enter a new student's details:")
        c1, c2, c3 = st.columns(3)
        att = c1.slider(LABELS["attendance"], 0, 100, 70)
        asg = c2.slider(LABELS["assignment_marks"], 0, 100, 60)
        prev = c3.slider(LABELS["previous_exam_marks"], 0, 100, 55)

        if st.button("Predict", type="primary"):
            row = pd.DataFrame([[att, asg, prev]], columns=FEATURES)
            p_pass = float(res["model"].predict_proba(row)[0][1])

            if p_pass >= 0.7:
                st.success(f"✅ Likely to pass — {p_pass:.0%} chance of passing")
            elif p_pass >= 0.4:
                st.warning(f"⚠️ Borderline — {p_pass:.0%} chance of passing. Keep an eye on this student.")
            else:
                st.error(f"🚩 At risk — only {p_pass:.0%} chance of passing. Needs extra help.")

            st.progress(p_pass)
            st.caption("This is a statistical estimate from past records, not a guarantee.")
