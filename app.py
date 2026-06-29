"""
CreditExplain – Explainable AI for Credit Decisions
Production-grade Streamlit dashboard (NumPy/Pandas/Matplotlib only; no SHAP/sklearn).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditExplain – XAI Credit Platform",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURE_NAMES = [
    "age", "annual_income", "employment_years", "credit_score",
    "debt_to_income", "loan_amount", "loan_purpose", "num_credit_lines",
    "delinquencies_2yr", "revolving_util", "months_employed", "savings_balance",
]

FEATURE_DISPLAY = {
    "age": "Age (years)",
    "annual_income": "Annual Income ($)",
    "employment_years": "Employment Years",
    "credit_score": "Credit Score",
    "debt_to_income": "Debt-to-Income Ratio",
    "loan_amount": "Loan Amount ($)",
    "loan_purpose": "Loan Purpose (encoded)",
    "num_credit_lines": "Num Credit Lines",
    "delinquencies_2yr": "Delinquencies (2yr)",
    "revolving_util": "Revolving Utilisation %",
    "months_employed": "Months Employed",
    "savings_balance": "Savings Balance ($)",
}

# ──────────────────────────────────────────────────────────────────────────────
# PURE-NUMPY PRIMITIVES
# ──────────────────────────────────────────────────────────────────────────────
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


class Standardizer:
    def fit(self, X):
        X = np.asarray(X, float)
        self.mu_ = X.mean(0)
        self.sd_ = X.std(0) + 1e-8
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.mu_) / self.sd_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return np.asarray(X, float) * self.sd_ + self.mu_


class LogisticRegressionSGD:
    def __init__(self, lr=0.15, epochs=600, l2=1e-3, seed=42):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        self.w_ = rng.normal(0, 0.01, d)
        self.b_ = 0.0
        pos = max(y.sum(), 1.0)
        neg = max((1 - y).sum(), 1.0)
        sw = np.where(y == 1, n / (2 * pos), n / (2 * neg))
        for _ in range(self.epochs):
            p = sigmoid(X @ self.w_ + self.b_)
            err = (p - y) * sw
            self.w_ -= self.lr * (X.T @ err / n + self.l2 * self.w_)
            self.b_ -= self.lr * err.mean()
        return self

    def predict_proba(self, X):
        return sigmoid(np.asarray(X, float) @ self.w_ + self.b_)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ──────────────────────────────────────────────────────────────────────────────
# METRICS (manual)
# ──────────────────────────────────────────────────────────────────────────────
def roc_auc_score(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    npos = (y == 1).sum()
    nneg = (y == 0).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def precision_recall_curve_manual(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    thresholds = np.linspace(s.min(), s.max(), 300)
    precisions, recalls = [], []
    for t in thresholds:
        pred = (s >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(pr)
        recalls.append(rc)
    return np.array(precisions), np.array(recalls), thresholds


def pr_auc_score(y, s):
    p, r, _ = precision_recall_curve_manual(y, s)
    order = np.argsort(r)
    return float(np.trapz(p[order], r[order]))


def roc_curve_manual(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    thresholds = np.unique(s)[::-1]
    tprs, fprs = [0.0], [0.0]
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    for t in thresholds:
        pred = (s >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tprs.append(tp / pos if pos else 0)
        fprs.append(fp / neg if neg else 0)
    tprs.append(1.0)
    fprs.append(1.0)
    return np.array(fprs), np.array(tprs)


def accuracy_score(y, p):
    return float((np.asarray(y) == np.asarray(p)).mean())


def gini_coefficient(y, s):
    return 2 * roc_auc_score(y, s) - 1


def ks_statistic(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    order = np.argsort(s)
    y_sorted = y[order]
    cum_pos = np.cumsum(y_sorted) / max(y.sum(), 1)
    cum_neg = np.cumsum(1 - y_sorted) / max((1 - y).sum(), 1)
    return float(np.max(np.abs(cum_pos - cum_neg)))


# ──────────────────────────────────────────────────────────────────────────────
# DATA GENERATION
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Generating synthetic dataset…")
def generate_dataset(n=20000, seed=7):
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 75, n).astype(float)
    annual_income = np.clip(rng.lognormal(10.8, 0.6, n), 15000, 500000)
    employment_years = np.clip(rng.exponential(6, n), 0, 40)
    credit_score = np.clip(rng.normal(680, 80, n), 300, 850)
    debt_to_income = np.clip(rng.beta(2, 5, n) * 0.9, 0.01, 0.90)
    loan_amount = np.clip(rng.lognormal(10.0, 0.8, n), 1000, 100000)
    loan_purpose = rng.integers(0, 5, n).astype(float)
    num_credit_lines = np.clip(rng.poisson(6, n), 0, 30).astype(float)
    delinquencies_2yr = np.clip(rng.poisson(0.3, n), 0, 10).astype(float)
    revolving_util = np.clip(rng.beta(2, 3, n) * 100, 0, 100)
    months_employed = np.clip(employment_years * 12 + rng.normal(0, 3, n), 0, 480)
    savings_balance = np.clip(rng.lognormal(9.5, 1.2, n), 0, 300000)

    # Realistic logit with correlations
    logit = (
        -3.5
        + 0.012 * (credit_score - 680)
        - 3.5 * debt_to_income
        + 0.008 * np.log1p(annual_income - 15000)
        - 0.4 * delinquencies_2yr
        + 0.15 * employment_years
        - 0.015 * revolving_util
        + 0.004 * num_credit_lines
        + 0.002 * np.log1p(savings_balance)
        - 0.002 * np.log1p(loan_amount)
        + 0.005 * age
        + 0.001 * months_employed
        + rng.normal(0, 0.4, n)
    )
    prob = sigmoid(logit)
    y = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "age": age,
        "annual_income": annual_income,
        "employment_years": employment_years,
        "credit_score": credit_score,
        "debt_to_income": debt_to_income,
        "loan_amount": loan_amount,
        "loan_purpose": loan_purpose,
        "num_credit_lines": num_credit_lines,
        "delinquencies_2yr": delinquencies_2yr,
        "revolving_util": revolving_util,
        "months_employed": months_employed,
        "savings_balance": savings_balance,
        "credit_approved": y,
    })

    # Protected attributes
    df["age_group"] = pd.cut(
        df["age"], bins=[17, 35, 55, 75],
        labels=["18-35", "36-55", "56+"]
    ).astype(str)
    df["gender_proxy"] = (rng.random(n) < 0.52).astype(int)  # 0=F,1=M (proxy)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training logistic regression…")
def train_model(seed=7):
    df = generate_dataset(seed=seed)
    X = df[FEATURE_NAMES].values.astype(float)
    y = df["credit_approved"].values.astype(int)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * 0.25)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = Standardizer().fit(X_train)
    X_tr_s = scaler.transform(X_train)
    X_te_s = scaler.transform(X_test)

    model = LogisticRegressionSGD(lr=0.15, epochs=600, l2=1e-3, seed=seed)
    model.fit(X_tr_s, y_train)

    proba_test = model.predict_proba(X_te_s)
    proba_train = model.predict_proba(X_tr_s)

    auc = roc_auc_score(y_test, proba_test)
    prauc = pr_auc_score(y_test, proba_test)
    acc = accuracy_score(y_test, (proba_test >= 0.5).astype(int))
    gini = gini_coefficient(y_test, proba_test)
    ks = ks_statistic(y_test, proba_test)

    return {
        "model": model,
        "scaler": scaler,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "proba_test": proba_test,
        "proba_train": proba_train,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "metrics": {
            "auc": auc,
            "pr_auc": prauc,
            "accuracy": acc,
            "gini": gini,
            "ks": ks,
        },
    }


def predict_proba(bundle, X):
    Xs = bundle["scaler"].transform(np.atleast_2d(np.asarray(X, float)))
    return bundle["model"].predict_proba(Xs)


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bank-card-back-side.png", width=64)
    st.title("CreditExplain")
    st.caption("Explainable AI · Credit Decisions")
    st.divider()

    st.subheader("Global Controls")
    decision_threshold = st.slider(
        "Decision Threshold", 0.10, 0.90, 0.50, 0.01,
        help="Probability cutoff for approval decision"
    )
    protected_attr = st.selectbox(
        "Protected Attribute", ["age_group", "gender_proxy"],
        help="Attribute used for fairness analysis"
    )
    fairness_metric = st.selectbox(
        "Primary Fairness Metric",
        ["Demographic Parity", "Equal Opportunity", "Predictive Parity", "Disparate Impact"]
    )
    st.divider()
    st.caption("Only NumPy · Pandas · Matplotlib · Streamlit")

# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA & MODEL
# ──────────────────────────────────────────────────────────────────────────────
df = generate_dataset()
bundle = train_model()
metrics = bundle["metrics"]

# Approval rate at current threshold
all_proba = predict_proba(bundle, df[FEATURE_NAMES].values)
approvals = (all_proba >= decision_threshold).mean()

# Disparate impact (quick compute for header)
def _quick_disparate_impact(df, proba, threshold, attr):
    preds = (proba >= threshold).astype(int)
    groups = df[attr].unique()
    rates = {}
    for g in groups:
        mask = df[attr].values == g
        rates[str(g)] = preds[mask].mean()
    if len(rates) < 2:
        return 1.0
    mn, mx = min(rates.values()), max(rates.values())
    return mn / mx if mx > 0 else 1.0

di_ratio = _quick_disparate_impact(df, all_proba, decision_threshold, protected_attr)
fairness_status = "PASS ✅" if di_ratio >= 0.8 else "FAIL ❌"

# ──────────────────────────────────────────────────────────────────────────────
# HEADER KPIs
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("## 💳 CreditExplain – Explainable AI Credit Platform")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Applications", f"{len(df):,}")
k2.metric("Approval Rate", f"{approvals:.1%}")
k3.metric("Model AUC", f"{metrics['auc']:.4f}")
k4.metric("Fairness (DI Ratio)", f"{di_ratio:.3f}", delta=fairness_status, delta_color="normal")

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dataset & Model Overview",
    "🔍 SHAP-Style Feature Importance",
    "👤 Individual Prediction Explainer",
    "⚖ Fairness & Bias Analysis",
    "📋 Regulatory Compliance",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – DATASET & MODEL OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Dataset & Model Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Training Samples", f"{len(bundle['X_train']):,}")
    c2.metric("Test Samples", f"{len(bundle['X_test']):,}")
    c3.metric("Base Approval Rate", f"{df['credit_approved'].mean():.1%}")

    col_a, col_b = st.columns(2)

    # Feature distributions
    with col_a:
        st.subheader("Feature Distributions")
        feat_sel = st.selectbox("Select feature", FEATURE_NAMES, key="feat_dist")
        fig, ax = plt.subplots(figsize=(5, 3))
        vals_approved = df.loc[df["credit_approved"] == 1, feat_sel]
        vals_declined = df.loc[df["credit_approved"] == 0, feat_sel]
        bins = 40
        ax.hist(vals_declined, bins=bins, alpha=0.6, color="#e74c3c", label="Declined", density=True)
        ax.hist(vals_approved, bins=bins, alpha=0.6, color="#27ae60", label="Approved", density=True)
        ax.set_xlabel(FEATURE_DISPLAY.get(feat_sel, feat_sel))
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.set_title(f"Distribution: {FEATURE_DISPLAY.get(feat_sel, feat_sel)}")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Approval by feature decile
    with col_b:
        st.subheader("Approval Rate by Decile")
        feat_dec = st.selectbox("Select feature", FEATURE_NAMES, index=3, key="feat_decile")
        df["_decile"] = pd.qcut(df[feat_dec], q=10, duplicates="drop")
        decile_rate = df.groupby("_decile", observed=True)["credit_approved"].mean()
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.bar(range(len(decile_rate)), decile_rate.values, color="#3498db", edgecolor="white")
        ax2.axhline(df["credit_approved"].mean(), color="red", linestyle="--", linewidth=1, label="Overall rate")
        ax2.set_xticks(range(len(decile_rate)))
        ax2.set_xticklabels([f"D{i+1}" for i in range(len(decile_rate))], fontsize=7)
        ax2.set_ylabel("Approval Rate")
        ax2.set_title(f"Approval Rate by {feat_dec} Decile")
        ax2.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
        df.drop(columns=["_decile"], inplace=True)

    # Correlation matrix
    st.subheader("Correlation Matrix")
    corr_df = df[FEATURE_NAMES + ["credit_approved"]].corr()
    fig3, ax3 = plt.subplots(figsize=(11, 7))
    im = ax3.imshow(corr_df.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax3.set_xticks(range(len(corr_df.columns)))
    ax3.set_yticks(range(len(corr_df.columns)))
    ax3.set_xticklabels(corr_df.columns, rotation=45, ha="right", fontsize=8)
    ax3.set_yticklabels(corr_df.columns, fontsize=8)
    for i in range(len(corr_df)):
        for j in range(len(corr_df.columns)):
            v = corr_df.values[i, j]
            ax3.text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=6, color="black" if abs(v) < 0.6 else "white")
    plt.colorbar(im, ax=ax3, shrink=0.8)
    ax3.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # Model performance
    st.subheader("Model Performance")
    pm1, pm2, pm3, pm4, pm5 = st.columns(5)
    pm1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    pm2.metric("ROC-AUC", f"{metrics['auc']:.4f}")
    pm3.metric("PR-AUC", f"{metrics['pr_auc']:.4f}")
    pm4.metric("Gini", f"{metrics['gini']:.4f}")
    pm5.metric("KS Statistic", f"{metrics['ks']:.4f}")

    col_roc, col_pr = st.columns(2)

    with col_roc:
        fpr_arr, tpr_arr = roc_curve_manual(bundle["y_test"], bundle["proba_test"])
        fig_r, ax_r = plt.subplots(figsize=(4.5, 3.5))
        ax_r.plot(fpr_arr, tpr_arr, color="#3498db", lw=2, label=f"AUC={metrics['auc']:.4f}")
        ax_r.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax_r.fill_between(fpr_arr, tpr_arr, alpha=0.1, color="#3498db")
        ax_r.set_xlabel("False Positive Rate")
        ax_r.set_ylabel("True Positive Rate")
        ax_r.set_title("ROC Curve")
        ax_r.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_r)
        plt.close(fig_r)

    with col_pr:
        prec_arr, rec_arr, _ = precision_recall_curve_manual(bundle["y_test"], bundle["proba_test"])
        fig_p, ax_p = plt.subplots(figsize=(4.5, 3.5))
        order = np.argsort(rec_arr)
        ax_p.plot(rec_arr[order], prec_arr[order], color="#e67e22", lw=2,
                  label=f"PR-AUC={metrics['pr_auc']:.4f}")
        ax_p.axhline(bundle["y_test"].mean(), color="gray", linestyle="--", lw=1, label="Baseline")
        ax_p.fill_between(rec_arr[order], prec_arr[order], alpha=0.1, color="#e67e22")
        ax_p.set_xlabel("Recall")
        ax_p.set_ylabel("Precision")
        ax_p.set_title("Precision-Recall Curve")
        ax_p.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_p)
        plt.close(fig_p)

    # Score distribution
    st.subheader("Score Distribution")
    fig_sd, ax_sd = plt.subplots(figsize=(8, 3))
    ax_sd.hist(bundle["proba_test"][bundle["y_test"] == 0], bins=60, alpha=0.6,
               color="#e74c3c", label="Declined (actual)", density=True)
    ax_sd.hist(bundle["proba_test"][bundle["y_test"] == 1], bins=60, alpha=0.6,
               color="#27ae60", label="Approved (actual)", density=True)
    ax_sd.axvline(decision_threshold, color="navy", linestyle="--", lw=2,
                  label=f"Threshold={decision_threshold:.2f}")
    ax_sd.set_xlabel("Predicted Probability")
    ax_sd.set_ylabel("Density")
    ax_sd.set_title("Model Score Distribution")
    ax_sd.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_sd)
    plt.close(fig_sd)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – SHAP-STYLE FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("SHAP-Style Feature Importance")

    st.latex(r"\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!\,(|F|-|S|-1)!}{|F|!}\bigl[f(S\cup\{i\})-f(S)\bigr]")
    st.caption("Implemented via Permutation Feature Importance (efficient approximation).")

    @st.cache_data(show_spinner="Computing permutation importances…")
    def compute_permutation_importance(n_repeats=10, n_boot=50, seed=99):
        X_te = bundle["X_test"]
        y_te = bundle["y_test"]
        base_auc = roc_auc_score(y_te, bundle["proba_test"])
        rng = np.random.default_rng(seed)
        n_feats = X_te.shape[1]

        importances = np.zeros((n_feats, n_repeats))
        for i in range(n_feats):
            for r in range(n_repeats):
                X_perm = X_te.copy()
                X_perm[:, i] = rng.permutation(X_perm[:, i])
                p_perm = predict_proba(bundle, X_perm)
                importances[i, r] = base_auc - roc_auc_score(y_te, p_perm)

        # Bootstrap CI
        boot_means = np.zeros((n_feats, n_boot))
        for b in range(n_boot):
            rep_idx = rng.integers(0, n_repeats, n_repeats)
            boot_means[:, b] = importances[:, rep_idx].mean(axis=1)

        mean_imp = importances.mean(axis=1)
        ci_lo = np.percentile(boot_means, 2.5, axis=1)
        ci_hi = np.percentile(boot_means, 97.5, axis=1)
        return mean_imp, ci_lo, ci_hi, base_auc

    mean_imp, ci_lo, ci_hi, base_auc = compute_permutation_importance()
    order = np.argsort(mean_imp)[::-1]
    feat_names_sorted = [FEATURE_NAMES[i] for i in order]

    # Global feature importance bar chart
    st.subheader("Global Permutation Feature Importance")
    fig_fi, ax_fi = plt.subplots(figsize=(9, 5))
    colors = ["#e74c3c" if v >= 0 else "#95a5a6" for v in mean_imp[order]]
    bars = ax_fi.barh(feat_names_sorted[::-1], mean_imp[order][::-1], color=colors[::-1],
                      xerr=[mean_imp[order][::-1] - ci_lo[order][::-1],
                             ci_hi[order][::-1] - mean_imp[order][::-1]],
                      capsize=4, error_kw={"elinewidth": 1.5, "ecolor": "#555"})
    ax_fi.axvline(0, color="black", linewidth=0.8)
    ax_fi.set_xlabel("Mean AUC Decrease (importance)")
    ax_fi.set_title("Permutation Feature Importance (10 repeats, 95% CI bootstrap)")
    plt.tight_layout()
    st.pyplot(fig_fi)
    plt.close(fig_fi)

    # Partial Dependence Plots
    st.subheader("Partial Dependence Plots (Top 3 Features)")

    @st.cache_data(show_spinner="Computing partial dependence…")
    def compute_partial_dependence():
        top3_idx = np.argsort(mean_imp)[::-1][:3]
        X_te = bundle["X_test"]
        scaler = bundle["scaler"]
        results = {}
        for idx in top3_idx:
            feat = FEATURE_NAMES[idx]
            grid = np.linspace(X_te[:, idx].min(), X_te[:, idx].max(), 60)
            pd_vals = []
            for v in grid:
                X_mod = X_te.copy()
                X_mod[:, idx] = v
                pd_vals.append(predict_proba(bundle, X_mod).mean())
            results[feat] = (grid, np.array(pd_vals))
        return results

    pd_results = compute_partial_dependence()
    top3_feats = list(pd_results.keys())
    fig_pd, axes_pd = plt.subplots(1, 3, figsize=(13, 4))
    for ax, feat in zip(axes_pd, top3_feats):
        grid, pdp = pd_results[feat]
        ax.plot(grid, pdp, color="#3498db", lw=2.5)
        ax.fill_between(grid, pdp - 0.02, pdp + 0.02, alpha=0.15, color="#3498db", label="±0.02 band")
        ax.axhline(0.5, color="red", linestyle="--", lw=1, label="Decision boundary")
        ax.set_xlabel(FEATURE_DISPLAY.get(feat, feat), fontsize=9)
        ax.set_ylabel("Mean P(Approve)")
        ax.set_title(f"PDP: {feat}")
        ax.legend(fontsize=7)
    plt.suptitle("Marginal Effect (Partial Dependence Plots)", fontsize=11, y=1.02)
    plt.tight_layout()
    st.pyplot(fig_pd)
    plt.close(fig_pd)

    # LIME-style local explanation
    st.subheader("LIME-Style Local Explanation")
    st.caption("Select an applicant from the test set to see a local linear approximation of the model.")

    lime_idx = st.slider("Test set applicant index", 0, len(bundle["X_test"]) - 1, 42, key="lime_idx")
    instance = bundle["X_test"][lime_idx]
    true_label = bundle["y_test"][lime_idx]
    pred_prob = predict_proba(bundle, instance)[0]

    @st.cache_data(show_spinner="Computing LIME approximation…")
    def compute_lime(instance_tuple, n_samples=200, seed=55):
        instance = np.array(instance_tuple)
        rng = np.random.default_rng(seed)
        X_te = bundle["X_test"]
        std = X_te.std(axis=0) + 1e-8

        # Sample perturbations
        noise = rng.normal(0, 1, (n_samples, len(instance)))
        X_perturbed = instance + noise * std * 0.3
        probs = predict_proba(bundle, X_perturbed)

        # Kernel: inverse distance weighting
        dists = np.sqrt(((X_perturbed - instance) ** 2 / (std ** 2)).sum(axis=1))
        kernel_width = np.percentile(dists, 75)
        weights = np.exp(-dists ** 2 / (2 * kernel_width ** 2))

        # Weighted ridge regression
        Xb = np.hstack([np.ones((n_samples, 1)), (X_perturbed - instance) / std])
        W = np.diag(weights)
        alpha = 1.0
        A = Xb.T @ W @ Xb + alpha * np.eye(Xb.shape[1])
        A[0, 0] -= alpha
        coef = np.linalg.solve(A, Xb.T @ W @ probs)
        return coef[1:]  # feature contributions

    lime_coefs = compute_lime(tuple(instance.tolist()))
    lime_order = np.argsort(np.abs(lime_coefs))[::-1]

    fig_lime, ax_lime = plt.subplots(figsize=(8, 4))
    cols_lime = ["#27ae60" if c > 0 else "#e74c3c" for c in lime_coefs[lime_order]]
    ax_lime.barh(
        [FEATURE_NAMES[i] for i in lime_order][::-1],
        lime_coefs[lime_order][::-1],
        color=cols_lime[::-1]
    )
    ax_lime.axvline(0, color="black", lw=0.8)
    ax_lime.set_xlabel("LIME Local Contribution")
    ax_lime.set_title(f"LIME: Applicant #{lime_idx} | P(Approve)={pred_prob:.3f} | True={true_label}")
    plt.tight_layout()
    st.pyplot(fig_lime)
    plt.close(fig_lime)

    with st.expander("LIME Algorithm Details"):
        st.markdown("""
        **Local Interpretable Model-agnostic Explanations (LIME)**
        1. Sample 200 perturbed instances around the selected applicant
        2. Query the black-box model for each perturbation
        3. Weight samples by kernel similarity: `w = exp(-d²/2σ²)`
        4. Fit weighted ridge regression to approximate local decision boundary
        5. Report coefficients as feature contributions
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – INDIVIDUAL PREDICTION EXPLAINER
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Individual Prediction Explainer")

    with st.expander("Enter Applicant Details", expanded=True):
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            inp_age = st.slider("Age", 18, 75, 35, key="inp_age")
            inp_income = st.number_input("Annual Income ($)", 15000, 500000, 55000, step=1000, key="inp_income")
            inp_emp_years = st.slider("Employment Years", 0.0, 40.0, 5.0, 0.5, key="inp_emp")
            inp_credit_score = st.slider("Credit Score", 300, 850, 680, key="inp_cs")
        with r1c2:
            inp_dti = st.slider("Debt-to-Income Ratio", 0.01, 0.90, 0.35, 0.01, key="inp_dti")
            inp_loan = st.number_input("Loan Amount ($)", 1000, 100000, 20000, step=500, key="inp_loan")
            inp_purpose = st.selectbox("Loan Purpose", [0, 1, 2, 3, 4],
                                        format_func=lambda x: ["Debt Consolidation", "Home Improvement",
                                                                "Auto", "Education", "Other"][x], key="inp_purpose")
            inp_lines = st.slider("Num Credit Lines", 0, 30, 6, key="inp_lines")
        with r1c3:
            inp_delinq = st.slider("Delinquencies (2yr)", 0, 10, 0, key="inp_delinq")
            inp_util = st.slider("Revolving Utilisation %", 0.0, 100.0, 30.0, 1.0, key="inp_util")
            inp_months = st.slider("Months Employed", 0, 480, 60, key="inp_months")
            inp_savings = st.number_input("Savings Balance ($)", 0, 300000, 15000, step=500, key="inp_savings")

    applicant = np.array([
        inp_age, inp_income, inp_emp_years, inp_credit_score, inp_dti,
        inp_loan, float(inp_purpose), inp_lines, inp_delinq, inp_util,
        inp_months, inp_savings
    ], dtype=float)

    prob = predict_proba(bundle, applicant)[0]
    decision = "APPROVED" if prob >= decision_threshold else "DECLINED"
    decision_color = "#27ae60" if decision == "APPROVED" else "#e74c3c"

    col_gauge, col_water = st.columns([1, 2])

    with col_gauge:
        st.markdown(f"### Decision: <span style='color:{decision_color};font-weight:bold'>{decision}</span>",
                    unsafe_allow_html=True)
        st.metric("Approval Probability", f"{prob:.1%}")

        # Gauge chart
        fig_g, ax_g = plt.subplots(figsize=(4, 2.5), subplot_kw={"projection": "polar"})
        theta = np.linspace(0, np.pi, 300)
        cmap = LinearSegmentedColormap.from_list("rg", ["#e74c3c", "#f39c12", "#27ae60"])
        for i in range(len(theta) - 1):
            ax_g.plot([theta[i], theta[i + 1]], [1, 1], lw=8,
                      color=cmap(i / (len(theta) - 1)), solid_capstyle="butt")
        needle_angle = np.pi * (1 - prob)
        ax_g.annotate("", xy=(needle_angle, 0.9), xytext=(0, 0),
                      arrowprops=dict(arrowstyle="->", lw=2.5, color="navy"))
        ax_g.set_ylim(0, 1.2)
        ax_g.set_theta_zero_location("W")
        ax_g.set_theta_direction(1)
        ax_g.set_axis_off()
        ax_g.text(np.pi / 2, -0.3, f"{prob:.1%}", ha="center", va="center",
                  fontsize=18, fontweight="bold", color=decision_color,
                  transform=ax_g.transData)
        ax_g.set_title("Approval Probability Gauge", pad=10)
        st.pyplot(fig_g)
        plt.close(fig_g)

    with col_water:
        # Waterfall chart using model weights contribution
        st.subheader("Feature Contribution Waterfall")
        scaler = bundle["scaler"]
        model = bundle["model"]
        base_rate = bundle["y_train"].mean()

        x_scaled = scaler.transform(applicant.reshape(1, -1))[0]
        contributions = model.w_ * x_scaled
        base_logit = np.log(base_rate / (1 - base_rate)) if 0 < base_rate < 1 else 0.0

        # Sort by absolute contribution
        cont_order = np.argsort(np.abs(contributions))[::-1]

        running = base_rate
        fig_wf, ax_wf = plt.subplots(figsize=(7, 5))
        bars_y = [base_rate]
        bar_labels = ["Base Rate"]
        bar_colors = ["#95a5a6"]

        cumulative = base_rate
        feat_contributions_norm = contributions / max(contributions.sum() + model.b_, 1e-6)
        total_shift = prob - base_rate

        # Scale contributions proportionally
        abs_sum = np.abs(contributions).sum()
        scaled_contribs = (contributions / (abs_sum + 1e-8)) * abs(total_shift)

        bottoms = []
        tops = []
        running_val = base_rate
        for fi in cont_order[:8]:
            bottoms.append(running_val)
            running_val += scaled_contribs[fi]
            tops.append(running_val)
            bar_labels.append(FEATURE_NAMES[fi])
            bar_colors.append("#27ae60" if scaled_contribs[fi] >= 0 else "#e74c3c")

        bottoms.append(running_val)
        tops.append(prob)
        bar_labels.append("Final Score")
        bar_colors.append("#3498db")

        all_bottoms = [base_rate] + bottoms
        all_heights = [0] + [t - b for t, b in zip(tops, bottoms[:-1])]

        ax_wf.bar(["Base Rate"] + bar_labels[1:],
                  [abs(t - b) for t, b in zip([base_rate] + tops, [0] + bottoms)],
                  bottom=[0] + bottoms,
                  color=bar_colors, edgecolor="white", linewidth=0.5)

        ax_wf.axhline(decision_threshold, color="navy", linestyle="--", lw=1.5,
                      label=f"Threshold={decision_threshold:.2f}")
        ax_wf.set_ylabel("Probability")
        ax_wf.set_title("Waterfall: Feature Contributions to Decision")
        ax_wf.set_xticklabels(["Base Rate"] + bar_labels[1:], rotation=45, ha="right", fontsize=8)
        ax_wf.legend(fontsize=8)
        ax_wf.set_ylim(0, 1.05)
        plt.tight_layout()
        st.pyplot(fig_wf)
        plt.close(fig_wf)

    # Counterfactual explanation
    st.subheader("Counterfactual Explanation")
    if decision == "DECLINED":
        st.info("Finding minimal changes to get APPROVED…")

        @st.cache_data(show_spinner="Computing counterfactual…")
        def compute_counterfactual(applicant_tuple, target_prob=0.65, lr=0.05, steps=300):
            x = np.array(applicant_tuple, dtype=float)
            x_cf = x.copy()
            scaler = bundle["scaler"]
            model = bundle["model"]
            # Only move mutable features (not loan_purpose, delinquencies)
            mutable = [0, 1, 2, 3, 4, 5, 7, 9, 10, 11]
            x_s = scaler.transform(x_cf.reshape(1, -1))[0]
            for _ in range(steps):
                p = sigmoid(x_s @ model.w_ + model.b_)
                if p >= target_prob:
                    break
                grad = (p - target_prob) * model.w_
                x_s[mutable] -= lr * grad[mutable]
            x_cf_orig = scaler.inverse_transform(x_s.reshape(1, -1))[0]
            return x_cf_orig

        cf = compute_counterfactual(tuple(applicant.tolist()))
        delta = cf - applicant
        significant = [(FEATURE_NAMES[i], applicant[i], cf[i], delta[i])
                       for i in np.argsort(np.abs(delta))[::-1] if abs(delta[i]) > 0.01][:5]

        cf_rows = []
        for feat, orig, new, d in significant:
            direction = "Increase" if d > 0 else "Decrease"
            cf_rows.append({
                "Feature": FEATURE_DISPLAY.get(feat, feat),
                "Current Value": f"{orig:.2f}",
                "Suggested Value": f"{new:.2f}",
                "Change": f"{direction} by {abs(d):.2f}",
            })
        if cf_rows:
            st.dataframe(pd.DataFrame(cf_rows), use_container_width=True)
            cf_prob = predict_proba(bundle, cf)[0]
            st.success(f"With these changes, approval probability would be **{cf_prob:.1%}**")
    else:
        st.success("Application is already APPROVED. No counterfactual needed.")

    # k-NN similar applicants
    st.subheader("Similar Applicants (k-NN, k=3)")
    X_tr = bundle["X_train"]
    scaler_std = bundle["scaler"]
    x_norm = scaler_std.transform(applicant.reshape(1, -1))[0]
    X_tr_norm = scaler_std.transform(X_tr)
    dists = np.sqrt(((X_tr_norm - x_norm) ** 2).sum(axis=1))
    knn_idx = np.argsort(dists)[:3]

    knn_rows = []
    for k, idx_ in enumerate(knn_idx):
        row = {f: f"{X_tr[idx_, j]:.2f}" for j, f in enumerate(FEATURE_NAMES)}
        row["Decision"] = "Approved" if bundle["y_train"][idx_] == 1 else "Declined"
        row["Distance"] = f"{dists[idx_]:.3f}"
        knn_rows.append(row)
    st.dataframe(pd.DataFrame(knn_rows)[["Decision", "Distance"] + FEATURE_NAMES[:6]],
                 use_container_width=True)

    # Decision boundary visualisation
    st.subheader("Decision Boundary (2D Projection)")
    db_feat_a = st.selectbox("Feature A", FEATURE_NAMES, index=3, key="db_fa")
    db_feat_b = st.selectbox("Feature B", FEATURE_NAMES, index=4, key="db_fb")

    fa_idx = FEATURE_NAMES.index(db_feat_a)
    fb_idx = FEATURE_NAMES.index(db_feat_b)

    grid_a = np.linspace(X_tr[:, fa_idx].min(), X_tr[:, fa_idx].max(), 80)
    grid_b = np.linspace(X_tr[:, fb_idx].min(), X_tr[:, fb_idx].max(), 80)
    GA, GB = np.meshgrid(grid_a, grid_b)

    X_grid = np.tile(applicant, (80 * 80, 1))
    X_grid[:, fa_idx] = GA.ravel()
    X_grid[:, fb_idx] = GB.ravel()
    Z = predict_proba(bundle, X_grid).reshape(80, 80)

    fig_db, ax_db = plt.subplots(figsize=(6, 4))
    cont = ax_db.contourf(GA, GB, Z, levels=30, cmap="RdYlGn", alpha=0.7, vmin=0, vmax=1)
    ax_db.contour(GA, GB, Z, levels=[decision_threshold], colors="navy", linewidths=2)
    plt.colorbar(cont, ax=ax_db, label="P(Approve)")
    ax_db.scatter([applicant[fa_idx]], [applicant[fb_idx]], s=150, color="blue",
                  zorder=5, marker="*", label="Current Applicant")
    ax_db.set_xlabel(FEATURE_DISPLAY.get(db_feat_a, db_feat_a))
    ax_db.set_ylabel(FEATURE_DISPLAY.get(db_feat_b, db_feat_b))
    ax_db.set_title(f"Decision Boundary: {db_feat_a} vs {db_feat_b}")
    ax_db.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_db)
    plt.close(fig_db)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – FAIRNESS & BIAS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Fairness & Bias Analysis")

    st.latex(r"\text{Demographic Parity Difference} = \left| P(\hat{Y}=1 \mid A=0) - P(\hat{Y}=1 \mid A=1) \right|")
    st.latex(r"\text{Disparate Impact Ratio} = \frac{\min_g P(\hat{Y}=1 \mid A=g)}{\max_g P(\hat{Y}=1 \mid A=g)}")
    st.latex(r"\text{Equal Opportunity} = \text{TPR}_g = P(\hat{Y}=1 \mid Y=1, A=g)")

    @st.cache_data(show_spinner="Computing fairness metrics…")
    def compute_fairness(attr, threshold):
        proba_all = predict_proba(bundle, df[FEATURE_NAMES].values)
        preds_all = (proba_all >= threshold).astype(int)
        y_all = df["credit_approved"].values
        groups = sorted(df[attr].unique())

        results = {}
        for g in groups:
            mask = df[attr].values == str(g) if df[attr].dtype == object else df[attr].values == g
            y_g = y_all[mask]
            p_g = preds_all[mask]
            pr_g = proba_all[mask]

            tp = int(((p_g == 1) & (y_g == 1)).sum())
            fp = int(((p_g == 1) & (y_g == 0)).sum())
            fn = int(((p_g == 0) & (y_g == 1)).sum())
            tn = int(((p_g == 0) & (y_g == 0)).sum())

            approval_rate = p_g.mean()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            auc_g = roc_auc_score(y_g, pr_g)

            results[str(g)] = {
                "approval_rate": approval_rate,
                "tpr": tpr,
                "fpr": fpr,
                "precision": precision,
                "auc": auc_g,
                "n": int(mask.sum()),
            }

        rates = [v["approval_rate"] for v in results.values()]
        tprs = [v["tpr"] for v in results.values()]
        mn_rate, mx_rate = min(rates), max(rates)
        di_ratio = mn_rate / mx_rate if mx_rate > 0 else 1.0
        dp_diff = mx_rate - mn_rate
        eo_diff = max(tprs) - min(tprs)

        return results, di_ratio, dp_diff, eo_diff

    fairness_results, di, dp_diff, eo_diff = compute_fairness(protected_attr, decision_threshold)
    groups_list = list(fairness_results.keys())

    # Traffic-light indicators
    st.subheader("Fairness Dashboard")

    def traffic_light(val, good_thresh, warn_thresh, higher_is_better=True):
        if higher_is_better:
            if val >= good_thresh:
                return "🟢"
            elif val >= warn_thresh:
                return "🟡"
            else:
                return "🔴"
        else:
            if val <= good_thresh:
                return "🟢"
            elif val <= warn_thresh:
                return "🟡"
            else:
                return "🔴"

    tl1, tl2, tl3, tl4 = st.columns(4)
    tl1.metric(
        f"{traffic_light(di, 0.8, 0.7)} Disparate Impact Ratio",
        f"{di:.3f}",
        help="≥0.8 pass (4/5 rule), 0.7-0.8 warning, <0.7 fail"
    )
    tl2.metric(
        f"{traffic_light(dp_diff, 0.05, 0.10, higher_is_better=False)} Dem. Parity Diff",
        f"{dp_diff:.3f}",
        help="<0.05 good, 0.05-0.10 warning, >0.10 fail"
    )
    tl3.metric(
        f"{traffic_light(eo_diff, 0.05, 0.10, higher_is_better=False)} Equal Opportunity Diff",
        f"{eo_diff:.3f}",
        help="Difference in TPR across groups"
    )
    overall_pass = di >= 0.8 and dp_diff <= 0.10
    tl4.metric("Overall Fairness", "PASS ✅" if overall_pass else "FAIL ❌")

    # Detailed group metrics
    st.subheader(f"Group Metrics by {protected_attr}")
    rows_fm = []
    for g, m in fairness_results.items():
        rows_fm.append({
            "Group": g,
            "Count": m["n"],
            "Approval Rate": f"{m['approval_rate']:.3f}",
            "TPR (Recall)": f"{m['tpr']:.3f}",
            "FPR": f"{m['fpr']:.3f}",
            "Precision": f"{m['precision']:.3f}",
            "AUC": f"{m['auc']:.3f}",
        })
    st.dataframe(pd.DataFrame(rows_fm), use_container_width=True)

    # Comparison charts
    fg1, fg2, fg3 = st.columns(3)

    with fg1:
        fig_ar, ax_ar = plt.subplots(figsize=(4, 3))
        vals = [fairness_results[g]["approval_rate"] for g in groups_list]
        colors_g = plt.cm.Set2(np.linspace(0, 1, len(groups_list)))
        ax_ar.bar(groups_list, vals, color=colors_g)
        ax_ar.axhline(np.mean(vals), color="red", linestyle="--", lw=1.5, label="Mean")
        ax_ar.set_ylabel("Approval Rate")
        ax_ar.set_title("Approval Rate by Group")
        ax_ar.set_ylim(0, 1)
        ax_ar.legend(fontsize=7)
        plt.tight_layout()
        st.pyplot(fig_ar)
        plt.close(fig_ar)

    with fg2:
        fig_tpr, ax_tpr = plt.subplots(figsize=(4, 3))
        tpr_vals = [fairness_results[g]["tpr"] for g in groups_list]
        fpr_vals = [fairness_results[g]["fpr"] for g in groups_list]
        x_ = np.arange(len(groups_list))
        ax_tpr.bar(x_ - 0.2, tpr_vals, 0.35, label="TPR", color="#3498db")
        ax_tpr.bar(x_ + 0.2, fpr_vals, 0.35, label="FPR", color="#e74c3c")
        ax_tpr.set_xticks(x_)
        ax_tpr.set_xticklabels(groups_list)
        ax_tpr.set_ylabel("Rate")
        ax_tpr.set_title("TPR & FPR by Group")
        ax_tpr.legend(fontsize=7)
        ax_tpr.set_ylim(0, 1)
        plt.tight_layout()
        st.pyplot(fig_tpr)
        plt.close(fig_tpr)

    with fg3:
        fig_prec, ax_prec = plt.subplots(figsize=(4, 3))
        prec_vals = [fairness_results[g]["precision"] for g in groups_list]
        ax_prec.bar(groups_list, prec_vals, color=colors_g)
        ax_prec.set_ylabel("Precision")
        ax_prec.set_title("Predictive Parity by Group")
        ax_prec.set_ylim(0, 1)
        plt.tight_layout()
        st.pyplot(fig_prec)
        plt.close(fig_prec)

    # Debiasing options
    st.subheader("Debiasing Options")
    debias_tab1, debias_tab2 = st.tabs(["Reweighing", "Threshold Calibration"])

    with debias_tab1:
        st.markdown("""
        **Reweighing** assigns different weights to training instances to equalise the
        expected positive rate across protected groups. The weight for group *g* is:

        `w(g) = P(Ŷ=1) / P(Ŷ=1 | A=g)`
        """)
        proba_all_arr = predict_proba(bundle, df[FEATURE_NAMES].values)
        global_rate = (proba_all_arr >= decision_threshold).mean()
        rw_rows = []
        for g, m in fairness_results.items():
            rw = global_rate / m["approval_rate"] if m["approval_rate"] > 0 else 1.0
            rw_rows.append({"Group": g, "Current Approval Rate": f"{m['approval_rate']:.3f}",
                            "Reweight Factor": f"{rw:.3f}",
                            "Post-reweight (target)": f"{global_rate:.3f}"})
        st.dataframe(pd.DataFrame(rw_rows), use_container_width=True)

    with debias_tab2:
        st.markdown("**Threshold Calibration** – set per-group thresholds to equalise TPR.")
        target_tpr = float(np.mean([fairness_results[g]["tpr"] for g in groups_list]))
        cal_rows = []
        proba_all_full = predict_proba(bundle, df[FEATURE_NAMES].values)
        y_all_full = df["credit_approved"].values

        for g in groups_list:
            mask = (df[protected_attr].astype(str) == str(g)).values
            y_g = y_all_full[mask]
            p_g = proba_all_full[mask]
            best_t, best_diff = decision_threshold, 1.0
            for t in np.linspace(0.1, 0.9, 81):
                pred_t = (p_g >= t).astype(int)
                tp_ = int(((pred_t == 1) & (y_g == 1)).sum())
                fn_ = int(((pred_t == 0) & (y_g == 1)).sum())
                tpr_t = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0.0
                if abs(tpr_t - target_tpr) < best_diff:
                    best_diff = abs(tpr_t - target_tpr)
                    best_t = t
            cal_rows.append({
                "Group": g,
                "Original TPR": f"{fairness_results[g]['tpr']:.3f}",
                "Target TPR": f"{target_tpr:.3f}",
                "Calibrated Threshold": f"{best_t:.2f}",
            })
        st.dataframe(pd.DataFrame(cal_rows), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – REGULATORY COMPLIANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Regulatory Compliance (ECOA / GDPR)")

    comp_tab1, comp_tab2, comp_tab3, comp_tab4 = st.tabs([
        "Adverse Action Notice",
        "Model Card",
        "Right to Explanation (GDPR Art.22)",
        "Model Monitoring (PSI)",
    ])

    # ── Adverse Action Notice ──────────────────────────────────────────────
    with comp_tab1:
        st.subheader("ECOA-Compliant Adverse Action Notice Generator")
        st.caption("US Equal Credit Opportunity Act (15 U.S.C. §1691) requires disclosure of principal reasons for adverse action.")

        declined_mask = predict_proba(bundle, df[FEATURE_NAMES].values) < decision_threshold
        declined_idx = np.where(declined_mask)[0]
        notice_idx = st.slider("Select declined applicant (dataset index)",
                               0, len(declined_idx) - 1, 0, key="ecoa_idx")
        real_idx = declined_idx[notice_idx]
        applicant_row = df.iloc[real_idx]
        app_prob = predict_proba(bundle, df[FEATURE_NAMES].values[real_idx])[0]

        # Compute per-feature negative contributions
        x_sc = bundle["scaler"].transform(
            df[FEATURE_NAMES].values[real_idx].reshape(1, -1))[0]
        contribs = bundle["model"].w_ * x_sc
        neg_contribs = [(FEATURE_NAMES[i], contribs[i]) for i in range(len(FEATURE_NAMES))
                        if contribs[i] < 0]
        neg_contribs.sort(key=lambda x: x[1])
        top3_neg = neg_contribs[:3]

        # Plain-English mapping
        ecoa_reason_map = {
            "credit_score": "Insufficient credit score",
            "debt_to_income": "Debt-to-income ratio too high",
            "delinquencies_2yr": "Derogatory marks or delinquencies on credit report",
            "revolving_util": "High revolving credit utilisation",
            "employment_years": "Insufficient employment history",
            "annual_income": "Insufficient income relative to loan amount",
            "savings_balance": "Inadequate savings or liquid assets",
            "loan_amount": "Requested loan amount exceeds qualification",
            "num_credit_lines": "Limited credit history",
            "months_employed": "Insufficient time with current employer",
            "age": "Applicant profile outside standard underwriting criteria",
            "loan_purpose": "Loan purpose risk assessment",
        }

        st.markdown("---")
        st.markdown(f"""
<div style="border:1px solid #ccc;padding:20px;border-radius:8px;background:#fafafa">
<h4>ADVERSE ACTION NOTICE</h4>
<p><strong>Date:</strong> {pd.Timestamp.now().strftime('%B %d, %Y')}<br>
<strong>Applicant Reference:</strong> #{real_idx:06d}<br>
<strong>Application Status:</strong> <span style="color:#e74c3c;font-weight:bold">DECLINED</span><br>
<strong>Approval Probability:</strong> {app_prob:.1%}</p>

<p>We regret to inform you that your application for credit has been declined.
This notice is provided pursuant to the Equal Credit Opportunity Act (ECOA),
15 U.S.C. §1691, and Regulation B (12 C.F.R. Part 1002).</p>

<h5>Principal Reasons for Adverse Action:</h5>
<ol>
{"".join(f"<li>{ecoa_reason_map.get(f, f.replace('_',' ').title())} (contribution score: {v:.4f})</li>" for f, v in top3_neg)}
</ol>

<p><em>You have the right to a free copy of your credit report and to dispute inaccurate
information with the credit bureaus. You also have the right to know whether a consumer
reporting agency was used in this decision.</em></p>

<p><strong>Model:</strong> Logistic Regression (NumPy SGD) | <strong>Version:</strong> 1.0.0</p>
</div>
""", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Applicant Feature Values")
        st.dataframe(applicant_row[FEATURE_NAMES].to_frame("Value").T, use_container_width=True)

    # ── Model Card ────────────────────────────────────────────────────────
    with comp_tab2:
        st.subheader("Model Card")
        st.markdown("""
| Field | Details |
|---|---|
| **Model Name** | CreditExplain Logistic Regression v1.0 |
| **Model Type** | Binary Classification (Logistic Regression, SGD) |
| **Training Data** | 15,000 synthetic applicants (stratified 75/25 split) |
| **Features** | 12 credit-relevant features (age, income, credit score, etc.) |
| **Target** | credit_approved (0/1) |
| **Training Framework** | Pure NumPy (no sklearn/SHAP) |
| **Intended Use** | Credit underwriting decision support |
| **Out-of-scope Use** | Sole automated decision-making without human review |
| **Primary Metric** | ROC-AUC |
| **Fairness Constraint** | Disparate Impact Ratio ≥ 0.80 (4/5 rule) |
        """)

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Gini Coefficient", f"{metrics['gini']:.4f}")
        mc2.metric("KS Statistic", f"{metrics['ks']:.4f}")
        mc3.metric("ROC-AUC", f"{metrics['auc']:.4f}")

        st.subheader("Known Limitations")
        st.warning("""
- Trained on **synthetic data** only; performance on real data may differ
- Linear model may not capture complex interactions between features
- Protected attribute proxies (age group, gender proxy) are simulated
- Model does not incorporate macroeconomic conditions or external credit bureau data
- Requires regular monitoring and retraining as population shifts occur
        """)

        st.subheader("Training Data Description")
        desc = df[FEATURE_NAMES].describe().round(3)
        st.dataframe(desc, use_container_width=True)

    # ── GDPR Right to Explanation ─────────────────────────────────────────
    with comp_tab3:
        st.subheader("Right to Explanation – GDPR Article 22")
        st.caption("Article 22 of GDPR requires meaningful information about the logic of automated decisions.")

        gdpr_idx = st.slider("Select applicant (test set)", 0, len(bundle["X_test"]) - 1, 10, key="gdpr_idx")
        gdpr_x = bundle["X_test"][gdpr_idx]
        gdpr_y = bundle["y_test"][gdpr_idx]
        gdpr_prob = predict_proba(bundle, gdpr_x)[0]
        gdpr_dec = "APPROVED" if gdpr_prob >= decision_threshold else "DECLINED"

        gdpr_x_sc = bundle["scaler"].transform(gdpr_x.reshape(1, -1))[0]
        gdpr_contribs = bundle["model"].w_ * gdpr_x_sc
        gdpr_order = np.argsort(np.abs(gdpr_contribs))[::-1]

        plain_english = {
            "credit_score": "your credit score",
            "debt_to_income": "your debt-to-income ratio",
            "annual_income": "your annual income",
            "employment_years": "your employment history",
            "delinquencies_2yr": "your recent payment history",
            "revolving_util": "your credit card utilisation",
            "savings_balance": "your savings balance",
            "loan_amount": "the loan amount requested",
            "num_credit_lines": "your number of credit accounts",
            "months_employed": "your tenure with your current employer",
            "age": "your age",
            "loan_purpose": "the purpose of your loan",
        }

        pos_factors = [(FEATURE_NAMES[i], gdpr_contribs[i]) for i in gdpr_order if gdpr_contribs[i] > 0][:3]
        neg_factors = [(FEATURE_NAMES[i], gdpr_contribs[i]) for i in gdpr_order if gdpr_contribs[i] < 0][:3]

        st.markdown(f"""
<div style="border:1px solid #3498db;padding:20px;border-radius:8px;background:#f0f7ff">
<h4>Human-Readable Explanation (GDPR Art. 22)</h4>

<p>Your credit application was <strong>{'approved' if gdpr_dec == 'APPROVED' else 'declined'}</strong>
with an approval probability of <strong>{gdpr_prob:.1%}</strong>.</p>

<p>Our automated system evaluated your application using 12 financial factors.
Here is a plain-language summary of the main factors that influenced this decision:</p>

<h5>Factors Working In Your Favour:</h5>
<ul>
{"".join(f'<li><strong>{plain_english.get(f, f)}</strong> (positive contribution: {v:+.3f})</li>' for f, v in pos_factors) if pos_factors else '<li>None identified above threshold</li>'}
</ul>

<h5>Factors Working Against You:</h5>
<ul>
{"".join(f'<li><strong>{plain_english.get(f, f)}</strong> (negative contribution: {v:+.3f})</li>' for f, v in neg_factors) if neg_factors else '<li>None identified above threshold</li>'}
</ul>

<p><em>This decision was made by an automated system. You have the right to request
human review of this decision, to express your point of view, and to contest this decision.
Contact our Data Protection Officer at dpo@creditexplain.example.com.</em></p>
</div>
""", unsafe_allow_html=True)

        # Feature importance in plain English bar chart
        fig_gdpr, ax_gdpr = plt.subplots(figsize=(8, 4))
        cols_gdpr = ["#27ae60" if c > 0 else "#e74c3c" for c in gdpr_contribs[gdpr_order]]
        ax_gdpr.barh(
            [plain_english.get(FEATURE_NAMES[i], FEATURE_NAMES[i]) for i in gdpr_order][::-1],
            gdpr_contribs[gdpr_order][::-1],
            color=cols_gdpr[::-1]
        )
        ax_gdpr.axvline(0, color="black", lw=0.8)
        ax_gdpr.set_xlabel("Contribution to Decision")
        ax_gdpr.set_title("GDPR-Compliant Feature Explanation (Plain English)")
        plt.tight_layout()
        st.pyplot(fig_gdpr)
        plt.close(fig_gdpr)

    # ── Model Monitoring / PSI ────────────────────────────────────────────
    with comp_tab4:
        st.subheader("Model Monitoring – Population Stability Index (PSI)")

        st.latex(r"\text{PSI} = \sum_{i=1}^{B}\left(\text{Actual}_i\% - \text{Expected}_i\%\right) \times \ln\!\left(\frac{\text{Actual}_i\%}{\text{Expected}_i\%}\right)")

        st.markdown("""
| PSI Value | Interpretation |
|---|---|
| < 0.10 | **Stable** – no significant shift |
| 0.10 – 0.20 | **Slight shift** – monitor closely |
| > 0.20 | **Major shift** – model retraining required |
        """)

        def compute_psi(expected_scores, actual_scores, n_bins=10):
            bins = np.percentile(expected_scores, np.linspace(0, 100, n_bins + 1))
            bins[0] = -np.inf
            bins[-1] = np.inf
            exp_counts = np.histogram(expected_scores, bins=bins)[0]
            act_counts = np.histogram(actual_scores, bins=bins)[0]
            exp_pct = (exp_counts + 1e-6) / len(expected_scores)
            act_pct = (act_counts + 1e-6) / len(actual_scores)
            psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
            return float(psi), exp_pct, act_pct, bins

        drift_level = st.slider("Simulate Score Drift", 0.0, 0.3, 0.05, 0.01,
                                help="Shift applied to test scores to simulate population drift")

        train_scores = bundle["proba_train"]
        rng_psi = np.random.default_rng(123)
        test_scores_drifted = np.clip(
            bundle["proba_test"] + drift_level + rng_psi.normal(0, 0.02, len(bundle["proba_test"])),
            0.001, 0.999
        )

        psi_val, exp_pct, act_pct, bins_psi = compute_psi(train_scores, test_scores_drifted)

        if psi_val < 0.1:
            psi_status = "🟢 Stable"
            psi_color = "#27ae60"
        elif psi_val < 0.2:
            psi_status = "🟡 Slight Shift"
            psi_color = "#f39c12"
        else:
            psi_status = "🔴 Major Shift"
            psi_color = "#e74c3c"

        st.metric(f"PSI Status: {psi_status}", f"{psi_val:.4f}")

        col_psi1, col_psi2 = st.columns(2)

        with col_psi1:
            fig_psi, ax_psi = plt.subplots(figsize=(5, 3.5))
            x_bins = np.arange(len(exp_pct))
            ax_psi.bar(x_bins - 0.2, exp_pct * 100, 0.35, label="Training (Expected)", color="#3498db", alpha=0.8)
            ax_psi.bar(x_bins + 0.2, act_pct * 100, 0.35, label="Test/Live (Actual)", color="#e74c3c", alpha=0.8)
            ax_psi.set_xlabel("Score Bin")
            ax_psi.set_ylabel("Population %")
            ax_psi.set_title(f"PSI Score Distribution | PSI={psi_val:.4f}")
            ax_psi.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_psi)
            plt.close(fig_psi)

        with col_psi2:
            fig_drift, ax_drift = plt.subplots(figsize=(5, 3.5))
            ax_drift.hist(train_scores, bins=50, alpha=0.6, density=True,
                          color="#3498db", label="Training scores")
            ax_drift.hist(test_scores_drifted, bins=50, alpha=0.6, density=True,
                          color="#e74c3c", label="Drifted test scores")
            ax_drift.set_xlabel("Predicted Probability")
            ax_drift.set_ylabel("Density")
            ax_drift.set_title("Score Distribution Drift")
            ax_drift.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_drift)
            plt.close(fig_drift)

        # Monitoring over time (simulated)
        st.subheader("Simulated PSI Over Time")
        rng_time = np.random.default_rng(77)
        months = list(range(1, 13))
        drifts = np.cumsum(rng_time.uniform(0.005, 0.025, 12))
        psi_over_time = []
        for d in drifts:
            drifted = np.clip(
                bundle["proba_test"] + d + rng_time.normal(0, 0.01, len(bundle["proba_test"])),
                0.001, 0.999
            )
            pv, _, _, _ = compute_psi(train_scores, drifted)
            psi_over_time.append(pv)

        fig_psi_t, ax_psi_t = plt.subplots(figsize=(8, 3.5))
        ax_psi_t.plot(months, psi_over_time, marker="o", color="#3498db", lw=2, label="PSI")
        ax_psi_t.axhline(0.1, color="#f39c12", linestyle="--", lw=1.5, label="0.10 (slight shift)")
        ax_psi_t.axhline(0.2, color="#e74c3c", linestyle="--", lw=1.5, label="0.20 (major shift)")
        ax_psi_t.fill_between(months, 0, [min(p, 0.1) for p in psi_over_time], alpha=0.15, color="#27ae60")
        ax_psi_t.fill_between(months,
                              [min(p, 0.1) for p in psi_over_time],
                              [min(p, 0.2) for p in psi_over_time], alpha=0.15, color="#f39c12")
        ax_psi_t.fill_between(months, [min(p, 0.2) for p in psi_over_time],
                              psi_over_time, alpha=0.15, color="#e74c3c")
        ax_psi_t.set_xlabel("Month")
        ax_psi_t.set_ylabel("PSI")
        ax_psi_t.set_title("PSI Trend (Simulated 12-Month Window)")
        ax_psi_t.set_xticks(months)
        ax_psi_t.set_xticklabels([f"M{m}" for m in months])
        ax_psi_t.legend(fontsize=8)
        ax_psi_t.set_ylim(0)
        plt.tight_layout()
        st.pyplot(fig_psi_t)
        plt.close(fig_psi_t)

# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "CreditExplain · Built with Streamlit · Pure NumPy/Pandas/Matplotlib · "
    "No SHAP/sklearn · All XAI methods implemented from scratch. "
    "For educational and demonstration purposes only. "
    "Not for use in production credit decisioning without regulatory review."
)
