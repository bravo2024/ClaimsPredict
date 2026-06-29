"""
ClaimsPredict | Allstate P&C Insurance Claims Severity Dashboard
Production-grade actuarial analytics platform.
Dependencies: numpy, pandas, matplotlib, scipy, streamlit only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
import streamlit as st

matplotlib.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1f2e",
    "axes.edgecolor":   "#2d3748",
    "axes.labelcolor":  "#e2e8f0",
    "xtick.color":      "#94a3b8",
    "ytick.color":      "#94a3b8",
    "text.color":       "#e2e8f0",
    "grid.color":       "#2d3748",
    "legend.facecolor": "#1a1f2e",
    "legend.edgecolor": "#2d3748",
})

PALETTE = ["#22d3ee", "#f43f5e", "#f97316", "#a855f7", "#22c55e", "#eab308"]

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ClaimsPredict | P&C Insurance Analytics",
    layout="wide",
    page_icon="🏦",
)

# ── sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/ClaimsPredict-P%26C%20Analytics-22d3ee?style=for-the-badge", use_container_width=True)
    st.markdown("---")
    st.header("⚙️ Dashboard Controls")

    policy_filter = st.multiselect(
        "Policy Type Filter",
        ["auto", "home", "commercial"],
        default=["auto", "home", "commercial"],
    )
    dev_periods = st.slider("Development Periods (Chain Ladder)", 4, 8, 6)
    confidence_level = st.slider("Confidence Level (%)", 90, 99, 95)
    min_claim_threshold = st.number_input(
        "Min Claim Threshold ($)", min_value=0, max_value=10000, value=0, step=500
    )
    expense_load_pct = st.slider("Expense Load (%)", 5, 40, 28)

    st.markdown("---")
    st.caption("Allstate P&C | IFRS 17 | Chain Ladder | Solvency II")
    st.caption("ClaimsPredict v2.0 — Production Dashboard")

# ── synthetic data generation ────────────────────────────────────────────────
@st.cache_data(show_spinner="Generating 15,000 claims portfolio…")
def generate_claims_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 15_000

    policy_type  = rng.choice(["auto", "home", "commercial"], p=[0.50, 0.30, 0.20], size=n)
    accident_sev = rng.integers(1, 5, size=n)           # 1-4
    vehicle_age  = rng.uniform(0, 20, size=n)
    driver_age   = rng.integers(18, 86, size=n)
    annual_mileage = rng.uniform(1_000, 50_000, size=n)
    num_vehicles = rng.integers(1, 6, size=n)           # 1-5
    claim_hour   = rng.integers(0, 24, size=n)
    weather      = rng.choice(["clear", "rain", "snow", "fog"], p=[0.55, 0.25, 0.12, 0.08], size=n)
    road_type    = rng.choice(["highway", "urban", "rural"], p=[0.40, 0.40, 0.20], size=n)
    prior_claims = rng.integers(0, 6, size=n)           # 0-5
    deductible   = rng.choice([500, 1000, 2500, 5000], p=[0.20, 0.40, 0.30, 0.10], size=n)
    coverage_amt = rng.uniform(10_000, 500_000, size=n)

    # 30% zero-payout
    has_payout = rng.random(n) > 0.30

    # log-normal positive amounts with feature influence
    log_mu = (
        8.5
        + 0.4  * accident_sev
        + 0.15 * (vehicle_age / 20)
        - 0.008 * (driver_age - 40)
        + 0.3  * (policy_type == "commercial").astype(float)
        + 0.15 * (weather == "snow").astype(float)
        + 0.10 * (weather == "fog").astype(float)
        + 0.08 * prior_claims
        - 0.5  * np.log1p(deductible / 1000)
    )
    claim_amount = np.where(
        has_payout,
        rng.lognormal(mean=log_mu - 8.5 + np.log(8_000), sigma=1.1, size=n),
        0.0,
    )

    # simulate 24-month timeline
    start = pd.Timestamp("2024-01-01")
    claim_dates = [start + pd.Timedelta(days=int(d)) for d in rng.integers(0, 730, size=n)]
    claim_month = [d.month for d in claim_dates]
    claim_year  = [d.year  for d in claim_dates]

    # synthetic premium (coverage-based with tier loading)
    base_premium = (
        coverage_amt * 0.012
        + accident_sev * 80
        + prior_claims * 150
        + (policy_type == "commercial").astype(float) * 400
    )

    df = pd.DataFrame({
        "policy_type":   policy_type,
        "accident_sev":  accident_sev,
        "vehicle_age":   vehicle_age,
        "driver_age":    driver_age,
        "annual_mileage": annual_mileage,
        "num_vehicles":  num_vehicles,
        "claim_hour":    claim_hour,
        "weather":       weather,
        "road_type":     road_type,
        "prior_claims":  prior_claims,
        "deductible":    deductible.astype(float),
        "coverage_amt":  coverage_amt,
        "has_payout":    has_payout.astype(int),
        "claim_amount":  claim_amount,
        "claim_month":   claim_month,
        "claim_year":    claim_year,
        "base_premium":  base_premium,
        "claim_date":    claim_dates,
    })
    return df

# ── two-part model (no sklearn) ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Training two-part severity model…")
def train_two_part_model(df: pd.DataFrame):
    rng = np.random.default_rng(0)

    # feature matrix (numeric encoding)
    X = np.column_stack([
        df["accident_sev"].values,
        df["vehicle_age"].values / 20,
        (df["driver_age"].values - 18) / (85 - 18),
        df["annual_mileage"].values / 50_000,
        df["num_vehicles"].values / 5,
        df["prior_claims"].values / 5,
        np.log1p(df["deductible"].values / 1000),
        df["coverage_amt"].values / 500_000,
        (df["policy_type"] == "commercial").astype(float).values,
        (df["policy_type"] == "home").astype(float).values,
        (df["weather"] == "snow").astype(float).values,
        (df["weather"] == "rain").astype(float).values,
        (df["road_type"] == "highway").astype(float).values,
        np.sin(2 * np.pi * df["claim_hour"].values / 24),
        np.cos(2 * np.pi * df["claim_hour"].values / 24),
    ])
    y = df["claim_amount"].values
    y_binary = (y > 0).astype(float)

    # 80/20 split
    idx = rng.permutation(len(X))
    split = int(0.8 * len(X))
    tr, te = idx[:split], idx[split:]
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y[tr], y[te]
    yb_tr, yb_te = y_binary[tr], y_binary[te]

    # ── Part 1: logistic regression (class-weighted gradient descent) ────────
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

    Xb = np.column_stack([np.ones(len(X_tr)), X_tr])
    w  = np.zeros(Xb.shape[1])
    pos_w = (1 - yb_tr.mean()) / yb_tr.mean()   # class weight for positives
    sample_w = np.where(yb_tr == 1, pos_w, 1.0)
    lr, iters = 0.05, 400
    for _ in range(iters):
        p   = sigmoid(Xb @ w)
        err = sample_w * (p - yb_tr)
        grad = Xb.T @ err / len(Xb)
        w -= lr * grad
    Xb_te    = np.column_stack([np.ones(len(X_te)), X_te])
    p_te     = sigmoid(Xb_te @ w)
    p_tr_hat = sigmoid(Xb @ w)

    # ── Part 2: log-linear OLS on positive claims ────────────────────────────
    pos_mask_tr = yb_tr == 1
    Xp = np.column_stack([np.ones(pos_mask_tr.sum()), X_tr[pos_mask_tr]])
    logy = np.log(y_tr[pos_mask_tr] + 1)
    beta, *_ = np.linalg.lstsq(Xp, logy, rcond=None)

    pos_mask_te  = yb_te == 1
    Xp_te        = np.column_stack([np.ones(pos_mask_te.sum()), X_te[pos_mask_te]])
    log_pred_te  = Xp_te @ beta
    y_pred_pos   = np.exp(log_pred_te) - 1

    # combined prediction: E[amount] = P(>0) × exp(log_pred)
    log_all_te   = np.column_stack([np.ones(len(X_te)), X_te]) @ beta
    y_combined   = p_te * (np.exp(log_all_te) - 1)

    # ── metrics on positive claims ────────────────────────────────────────────
    y_act_pos = y_te[pos_mask_te]
    rmse = np.sqrt(np.mean((y_pred_pos - y_act_pos) ** 2))
    mae  = np.mean(np.abs(y_pred_pos - y_act_pos))
    mape = np.mean(np.abs((y_pred_pos - y_act_pos) / (y_act_pos + 1))) * 100

    # permutation feature importance (on positive claims subset)
    feature_names = [
        "accident_sev", "vehicle_age", "driver_age", "annual_mileage",
        "num_vehicles", "prior_claims", "deductible", "coverage_amt",
        "commercial", "home", "weather_snow", "weather_rain",
        "highway", "hour_sin", "hour_cos",
    ]
    base_rmse = rmse
    importances = []
    for j in range(X_te.shape[1]):
        X_perm = X_te[pos_mask_te].copy()
        X_perm[:, j] = rng.permutation(X_perm[:, j])
        Xpp = np.column_stack([np.ones(pos_mask_te.sum()), X_perm])
        yp  = np.exp(Xpp @ beta) - 1
        imp = np.sqrt(np.mean((yp - y_act_pos) ** 2)) - base_rmse
        importances.append(max(imp, 0))

    return {
        "w_logit": w, "beta_ols": beta,
        "X_te": X_te, "y_te": y_te,
        "p_te": p_te, "y_combined": y_combined,
        "y_pred_pos": y_pred_pos, "y_act_pos": y_act_pos,
        "pos_mask_te": pos_mask_te,
        "rmse": rmse, "mae": mae, "mape": mape,
        "importances": importances, "feature_names": feature_names,
        "tr_idx": tr, "te_idx": te,
    }

# ── loss development triangle ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_development_triangle(n_periods: int = 6, seed: int = 7):
    rng  = np.random.default_rng(seed)
    tri  = np.zeros((n_periods, n_periods))
    base = np.array([4.2, 3.8, 3.5, 3.0, 2.5, 2.1]) * 1e6
    factors = np.array([2.8, 1.6, 1.25, 1.10, 1.04, 1.0])

    for i in range(n_periods):
        cum = base[i]
        for j in range(n_periods - i):
            noise = rng.normal(1.0, 0.04)
            cum   = cum * factors[j] * noise if j > 0 else cum
            tri[i, j] = round(cum, 0)
        # blank future cells (upper-right)
        for j in range(n_periods - i, n_periods):
            tri[i, j] = np.nan

    # age-to-age factors
    ldf = []
    for j in range(n_periods - 1):
        num   = sum(tri[i, j + 1] for i in range(n_periods - j - 1) if not np.isnan(tri[i, j + 1]))
        denom = sum(tri[i, j]     for i in range(n_periods - j - 1) if not np.isnan(tri[i, j + 1]))
        ldf.append(num / denom if denom else 1.0)
    ldf.append(1.0)  # tail factor

    # CDF = product of future factors
    cdf = np.ones(n_periods)
    for i in range(n_periods - 2, -1, -1):
        cdf[i] = cdf[i + 1] * ldf[i]

    # ultimate = latest diagonal × CDF
    diag      = [tri[i, n_periods - 1 - i] for i in range(n_periods)]
    ultimates = [d * c for d, c in zip(diag, cdf)]
    ibnr      = [u - d for u, d in zip(ultimates, diag)]

    return {
        "triangle": tri, "ldf": ldf, "cdf": cdf,
        "diag": diag, "ultimates": ultimates, "ibnr": ibnr,
        "n_periods": n_periods,
    }

# ── bootstrap chain ladder ───────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def bootstrap_chain_ladder(tri_data: dict, n_sim: int = 100, seed: int = 99):
    rng  = np.random.default_rng(seed)
    tri  = tri_data["triangle"]
    n    = tri_data["n_periods"]
    reserves = []
    for _ in range(n_sim):
        ldf_sim = []
        for j in range(n - 1):
            rows = [i for i in range(n - j - 1) if not np.isnan(tri[i, j + 1])]
            if not rows:
                ldf_sim.append(1.0); continue
            num   = sum(tri[i, j + 1] for i in rows)
            denom = sum(tri[i, j]     for i in rows)
            base  = num / denom if denom else 1.0
            ldf_sim.append(rng.lognormal(np.log(base), 0.05))
        ldf_sim.append(1.0)
        cdf_sim = np.ones(n)
        for i in range(n - 2, -1, -1):
            cdf_sim[i] = cdf_sim[i + 1] * ldf_sim[i]
        diag_sim = [tri[i, n - 1 - i] for i in range(n)]
        total_ibnr = sum((d * c - d) for d, c in zip(diag_sim, cdf_sim) if not np.isnan(d))
        reserves.append(total_ibnr)
    return np.array(reserves)

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA & MODEL
# ═══════════════════════════════════════════════════════════════════════════════
df_raw  = generate_claims_data()
tri_data = build_development_triangle(n_periods=dev_periods)

# Apply policy filter
df = df_raw[df_raw["policy_type"].isin(policy_filter)].copy()
if min_claim_threshold > 0:
    df = df[(df["claim_amount"] == 0) | (df["claim_amount"] >= min_claim_threshold)].copy()

model = train_two_part_model(df)

# ── header KPIs ───────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;color:#22d3ee;margin-bottom:4px'>"
    "🏦 ClaimsPredict — P&C Insurance Analytics Platform</h1>"
    "<p style='text-align:center;color:#94a3b8;margin-top:0'>Allstate-style "
    "actuarial intelligence · Chain Ladder · Tweedie Severity · Solvency II</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

total_claims    = len(df)
total_paid      = df["claim_amount"].sum()
avg_claim       = df[df["claim_amount"] > 0]["claim_amount"].mean()
zero_pct        = (df["claim_amount"] == 0).mean() * 100
total_premium   = df["base_premium"].sum()
loss_ratio      = total_paid / total_premium if total_premium > 0 else 0
ibnr_total      = sum(i for i in tri_data["ibnr"] if i > 0)
combined_ratio  = (loss_ratio + expense_load_pct / 100) * 100

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Claims",      f"{total_claims:,}")
k2.metric("Total Incurred",    f"${total_paid/1e6:.1f}M")
k3.metric("Avg Severity",      f"${avg_claim:,.0f}")
k4.metric("Zero-Payout Rate",  f"{zero_pct:.1f}%")
k5.metric("Loss Ratio",        f"{loss_ratio:.1%}")
k6.metric("Combined Ratio",    f"{combined_ratio:.1f}%",
          delta=f"{combined_ratio - 100:.1f}% vs 100%",
          delta_color="inverse")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
t1, t2, t3, t4, t5 = st.tabs([
    "📊 Claims Portfolio Explorer",
    "🔬 Actuarial Analysis",
    "🤖 Claims Severity Model",
    "📈 Risk Segmentation & Pricing",
    "💰 Reserving & Capital",
])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — Claims Portfolio Explorer
# ────────────────────────────────────────────────────────────────────────────
with t1:
    st.subheader("📊 Claims Portfolio Explorer")
    st.caption(f"Showing {len(df):,} claims · filters: policy={policy_filter}")

    col_a, col_b = st.columns(2)

    # claim amount distribution (log scale)
    with col_a:
        st.markdown("**Claim Amount Distribution (log scale)**")
        pos_amt = df[df["claim_amount"] > 0]["claim_amount"]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(pos_amt, bins=60, color=PALETTE[0], edgecolor="#0f1117", alpha=0.85)
        ax.set_xscale("log")
        ax.set_xlabel("Claim Amount ($) — log scale")
        ax.set_ylabel("Frequency")
        ax.set_title("Positive Claim Distribution (Heavy Right Tail)")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # zero vs non-zero (two-part model motivation)
    with col_b:
        st.markdown("**Zero vs Non-Zero Claims (Two-Part Model Motivation)**")
        z_ct = (df["claim_amount"] == 0).sum()
        nz_ct = (df["claim_amount"] > 0).sum()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(["Zero Payout\n(No Claim)", "Positive Claim"],
                      [z_ct, nz_ct], color=[PALETTE[1], PALETTE[0]], edgecolor="#0f1117")
        for bar, val in zip(bars, [z_ct, nz_ct]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 50, f"{val:,}\n({val/len(df):.1%})",
                    ha="center", fontsize=10, color="white")
        ax.set_ylabel("Number of Claims")
        ax.set_title("Claim Payout Split")
        ax.grid(True, alpha=0.3, axis="y")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    col_c, col_d = st.columns(2)

    # claims by policy type
    with col_c:
        st.markdown("**Avg Claim Amount by Policy Type**")
        pt_avg = df[df["claim_amount"] > 0].groupby("policy_type")["claim_amount"].mean()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(pt_avg.index, pt_avg.values, color=PALETTE[:len(pt_avg)], edgecolor="#0f1117")
        for i, (k, v) in enumerate(pt_avg.items()):
            ax.text(i, v + 100, f"${v:,.0f}", ha="center", fontsize=9, color="white")
        ax.set_ylabel("Avg Claim Amount ($)")
        ax.set_title("Severity by Policy Type")
        ax.grid(True, alpha=0.3, axis="y")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # claims by accident severity
    with col_d:
        st.markdown("**Claim Count by Accident Severity Level**")
        sev_ct = df.groupby("accident_sev")["claim_amount"].count()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar([f"Sev {s}" for s in sev_ct.index], sev_ct.values,
               color=PALETTE[2], edgecolor="#0f1117")
        ax.set_ylabel("Number of Claims")
        ax.set_title("Accident Severity Distribution")
        ax.grid(True, alpha=0.3, axis="y")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # monthly claim volume time series
    st.markdown("**Monthly Claim Volume — 24-Month Time Series**")
    df_ts = df_raw.copy()
    df_ts["ym"] = pd.to_datetime(
        df_ts["claim_year"].astype(str) + "-" + df_ts["claim_month"].astype(str).str.zfill(2)
    )
    monthly = df_ts.groupby("ym").agg(
        count=("claim_amount", "count"),
        total=("claim_amount", "sum"),
    ).reset_index().sort_values("ym")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    ax1.fill_between(range(len(monthly)), monthly["count"], alpha=0.5, color=PALETTE[0])
    ax1.plot(range(len(monthly)), monthly["count"], color=PALETTE[0], lw=2)
    ax1.set_ylabel("Claim Count")
    ax1.set_title("Monthly Claim Volume")
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(range(len(monthly)), monthly["total"] / 1e6, alpha=0.5, color=PALETTE[2])
    ax2.plot(range(len(monthly)), monthly["total"] / 1e6, color=PALETTE[2], lw=2)
    ax2.set_ylabel("Total Loss ($M)")
    ax2.set_xlabel("Month Index (Jan-2024 → Dec-2025)")
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(len(monthly)))
    ax2.set_xticklabels(
        [m.strftime("%b-%y") if i % 3 == 0 else "" for i, m in enumerate(monthly["ym"])],
        rotation=45, ha="right", fontsize=7,
    )
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # seasonal pattern
    st.markdown("**Seasonal Pattern — Claim Count by Month of Year**")
    seasonal = df_raw.groupby("claim_month")["claim_amount"].count().reset_index()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(range(1, 13), seasonal["claim_amount"], color=PALETTE[4], edgecolor="#0f1117")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    ax.set_ylabel("Total Claims")
    ax.set_title("Seasonal Claim Frequency (winter/fog spike expected)")
    ax.grid(True, alpha=0.3, axis="y")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    with st.expander("Raw Claims Data Sample"):
        st.dataframe(df.head(200), use_container_width=True, height=300)

# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — Actuarial Analysis
# ────────────────────────────────────────────────────────────────────────────
with t2:
    st.subheader("🔬 Actuarial Analysis")

    # ── Chain Ladder ─────────────────────────────────────────────────────────
    st.markdown("### Loss Development Triangle (Chain Ladder)")
    st.latex(
        r"f_j = \frac{\sum_i C_{i,j+1}}{\sum_i C_{i,j}}, \quad"
        r"\text{CDF}_j = \prod_{k \geq j} f_k, \quad"
        r"\text{Ultimate}_i = C_{i,\text{latest}} \times \text{CDF}_i"
    )

    tri  = tri_data["triangle"]
    n_p  = tri_data["n_periods"]
    ldf  = tri_data["ldf"]
    cdf  = tri_data["cdf"]
    diag = tri_data["diag"]
    ults = tri_data["ultimates"]
    ibnr = tri_data["ibnr"]

    # display triangle as DataFrame
    tri_df = pd.DataFrame(
        tri / 1e6,
        index=[f"AY {2019+i}" for i in range(n_p)],
        columns=[f"Dev {j+1}" for j in range(n_p)],
    ).round(2)
    tri_df.fillna("—", inplace=True)
    st.dataframe(tri_df, use_container_width=True)

    col_e, col_f = st.columns(2)
    with col_e:
        st.markdown("**Age-to-Age Development Factors**")
        ldf_df = pd.DataFrame({
            "Period→Period": [f"{j+1}→{j+2}" for j in range(n_p - 1)] + ["Tail"],
            "LDF":           [f"{f:.4f}" for f in ldf],
            "CDF to Ult":    [f"{c:.4f}" for c in cdf],
        })
        st.dataframe(ldf_df, use_container_width=True, hide_index=True)

    with col_f:
        st.markdown("**Chain Ladder Ultimates & IBNR**")
        ult_df = pd.DataFrame({
            "Acc. Year":   [f"AY {2019+i}" for i in range(n_p)],
            "Reported ($M)": [f"{d/1e6:.2f}" for d in diag],
            "Ultimate ($M)": [f"{u/1e6:.2f}" for u in ults],
            "IBNR ($M)":     [f"{max(i,0)/1e6:.2f}" for i in ibnr],
        })
        st.dataframe(ult_df, use_container_width=True, hide_index=True)

    # ── Bornhuetter-Ferguson ──────────────────────────────────────────────────
    st.markdown("### Bornhuetter-Ferguson Method")
    st.latex(
        r"\text{BF Ultimate}_i = R_i + E_i \times \left(1 - \frac{1}{\text{CDF}_i}\right)"
    )
    expected_loss_ratio = 0.62
    premium_by_ay = np.linspace(6e6, 3.5e6, n_p)  # declining premium exposure
    bf_ults, bf_ibnr = [], []
    for i in range(n_p):
        exp_loss = expected_loss_ratio * premium_by_ay[i]
        bf_u = diag[i] + exp_loss * (1 - 1 / cdf[i])
        bf_ults.append(bf_u)
        bf_ibnr.append(bf_u - diag[i])
    bf_df = pd.DataFrame({
        "Acc. Year":     [f"AY {2019+i}" for i in range(n_p)],
        "Reported ($M)": [f"{d/1e6:.2f}"  for d in diag],
        "CL Ult ($M)":   [f"{u/1e6:.2f}"  for u in ults],
        "BF Ult ($M)":   [f"{u/1e6:.2f}"  for u in bf_ults],
        "BF IBNR ($M)":  [f"{max(i,0)/1e6:.2f}" for i in bf_ibnr],
    })
    st.dataframe(bf_df, use_container_width=True, hide_index=True)

    # ── Frequency-Severity Decomposition ──────────────────────────────────────
    st.markdown("### Frequency-Severity Decomposition")
    st.latex(
        r"\text{Pure Premium} = \text{Frequency} \times \text{Severity}"
        r"\qquad \text{where Frequency} = \frac{\text{Claims Count}}{\text{Exposure}}"
    )
    pt_df_pos = df[df["claim_amount"] > 0]
    freq_sev = df.groupby("policy_type").apply(
        lambda g: pd.Series({
            "Exposure (policies)": len(g),
            "Claim Count":         (g["claim_amount"] > 0).sum(),
            "Frequency":           f"{(g['claim_amount'] > 0).mean():.3f}",
            "Avg Severity ($)":    f"${g[g['claim_amount']>0]['claim_amount'].mean():,.0f}" if (g["claim_amount"] > 0).any() else "$0",
            "Pure Premium ($)":    f"${(g['claim_amount'] > 0).mean() * g[g['claim_amount']>0]['claim_amount'].mean():,.0f}" if (g["claim_amount"] > 0).any() else "$0",
        })
    ).reset_index()
    st.dataframe(freq_sev, use_container_width=True, hide_index=True)

    # ── Combined Ratio ────────────────────────────────────────────────────────
    st.markdown("### Combined Ratio")
    st.latex(
        r"\text{Combined Ratio} = \left(\text{Loss Ratio} + \text{Expense Ratio}\right) \times 100\%"
        r"\quad \text{where } \text{Loss Ratio} = \frac{\text{Incurred Losses}}{\text{Earned Premium}}"
    )
    c1r, c2r, c3r, c4r = st.columns(4)
    c1r.metric("Loss Ratio",    f"{loss_ratio:.1%}")
    c2r.metric("Expense Ratio", f"{expense_load_pct}%")
    c3r.metric("Combined Ratio",f"{combined_ratio:.1f}%")
    c4r.metric("Underwriting Profit/Loss",
               f"${(total_premium * (1 - combined_ratio/100))/1e6:.2f}M",
               delta="Profit" if combined_ratio < 100 else "Loss",
               delta_color="normal" if combined_ratio < 100 else "inverse")

# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — Claims Severity Model
# ────────────────────────────────────────────────────────────────────────────
with t3:
    st.subheader("🤖 Two-Part Claims Severity Model")

    st.latex(
        r"\underbrace{P(\text{claim}>0|\mathbf{x})}_{\text{Part 1: Logistic}}"
        r"\times \underbrace{\exp\!\left(\boldsymbol{\beta}^\top \mathbf{x}\right)}_{\text{Part 2: Log-linear OLS}}"
        r"= \mathbb{E}[\text{Amount}|\mathbf{x}]"
    )

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("RMSE (positive claims)", f"${model['rmse']:,.0f}")
    col_m2.metric("MAE  (positive claims)", f"${model['mae']:,.0f}")
    col_m3.metric("MAPE (positive claims)", f"{model['mape']:.1f}%")

    col_p, col_q = st.columns(2)

    # Actual vs Predicted scatter (log scale)
    with col_p:
        st.markdown("**Actual vs Predicted — Positive Claims (log scale)**")
        y_act  = model["y_act_pos"]
        y_pred = model["y_pred_pos"]
        sample = min(2000, len(y_act))
        idx_s  = np.random.default_rng(1).choice(len(y_act), sample, replace=False)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_pred[idx_s], y_act[idx_s],
                   alpha=0.3, s=8, color=PALETTE[0], label="Claims")
        lims = [max(1, min(y_pred.min(), y_act.min())),
                max(y_pred.max(), y_act.max())]
        ax.plot(lims, lims, "r--", lw=1.5, label="Perfect Fit")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Predicted Claim Amount ($)")
        ax.set_ylabel("Actual Claim Amount ($)")
        ax.set_title("Actual vs Predicted (log scale)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Residuals by predicted decile (calibration)
    with col_q:
        st.markdown("**Calibration: Mean Actual vs Mean Predicted by Decile**")
        deciles = pd.qcut(y_pred, 10, labels=False, duplicates="drop")
        cal_df  = pd.DataFrame({"actual": y_act, "pred": y_pred, "decile": deciles})
        cal_grp = cal_df.groupby("decile").agg(
            mean_actual=("actual", "mean"), mean_pred=("pred", "mean")
        ).reset_index()
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(cal_grp["decile"], cal_grp["mean_actual"],
               alpha=0.7, color=PALETTE[0], label="Mean Actual", width=0.4, align="center")
        ax.bar(cal_grp["decile"] + 0.4, cal_grp["mean_pred"],
               alpha=0.7, color=PALETTE[2], label="Mean Predicted", width=0.4, align="center")
        ax.set_xlabel("Predicted Decile (1=lowest, 10=highest)")
        ax.set_ylabel("Average Claim Amount ($)")
        ax.set_title("Model Calibration by Decile")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    col_r, col_s = st.columns(2)

    # Gini / Lorenz curve
    with col_r:
        st.markdown("**Gini Coefficient — Ranking Performance**")
        sort_idx    = np.argsort(-y_pred)
        cum_pred    = np.cumsum(y_act[sort_idx]) / y_act.sum()
        rand_line   = np.linspace(0, 1, len(cum_pred))
        gini        = 1 - 2 * np.trapz(cum_pred, rand_line)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(rand_line, cum_pred, color=PALETTE[0], lw=2, label=f"Model (Gini={gini:.3f})")
        ax.plot([0, 1], [0, 1], "r--", lw=1.5, label="Random")
        ax.fill_between(rand_line, cum_pred, rand_line, alpha=0.2, color=PALETTE[0])
        ax.set_xlabel("Cumulative % of Claims (sorted by predicted)")
        ax.set_ylabel("Cumulative % of Actual Losses")
        ax.set_title("Lorenz / Gini Curve")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Feature importance
    with col_s:
        st.markdown("**Permutation Feature Importance (RMSE increase)**")
        imp_arr  = np.array(model["importances"])
        feat_nms = model["feature_names"]
        order    = np.argsort(imp_arr)[::-1]
        fig, ax  = plt.subplots(figsize=(5.5, 4.5))
        ax.barh(
            [feat_nms[i] for i in order[:10]],
            imp_arr[order[:10]],
            color=PALETTE[3], edgecolor="#0f1117",
        )
        ax.set_xlabel("RMSE Increase on Permutation")
        ax.set_title("Top-10 Feature Importance")
        ax.grid(True, alpha=0.3, axis="x")
        ax.invert_yaxis()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────
# TAB 4 — Risk Segmentation & Pricing
# ────────────────────────────────────────────────────────────────────────────
with t4:
    st.subheader("📈 Risk Segmentation & Pricing")

    # score all claims in test set using combined prediction
    y_te     = model["y_te"]
    y_comb   = model["y_combined"]
    te_idx   = model["te_idx"]
    df_te    = df.iloc[te_idx].copy()
    df_te["predicted_loss"] = np.maximum(y_comb, 0)
    df_te["actual_loss"]    = y_te

    # risk quintiles
    quintile_labels = ["Very Low", "Low", "Medium", "High", "Very High"]
    df_te["risk_tier"] = pd.qcut(
        df_te["predicted_loss"], 5,
        labels=quintile_labels, duplicates="drop"
    )

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("**Actual Loss Ratio by Risk Tier**")
        tier_grp = df_te.groupby("risk_tier", observed=True).agg(
            actual_loss=("actual_loss", "sum"),
            pred_loss=("predicted_loss", "sum"),
            premium=("base_premium", "sum"),
            count=("actual_loss", "count"),
        )
        tier_grp["actual_lr"]    = tier_grp["actual_loss"] / tier_grp["premium"]
        tier_grp["predicted_lr"] = tier_grp["pred_loss"]   / tier_grp["premium"]
        fig, ax = plt.subplots(figsize=(6, 4))
        x       = np.arange(len(tier_grp))
        ax.bar(x - 0.2, tier_grp["actual_lr"],    width=0.4, color=PALETTE[1],
               alpha=0.85, label="Actual LR",    edgecolor="#0f1117")
        ax.bar(x + 0.2, tier_grp["predicted_lr"], width=0.4, color=PALETTE[0],
               alpha=0.85, label="Predicted LR", edgecolor="#0f1117")
        ax.axhline(1.0, color="white", ls="--", lw=1, label="Break-even")
        ax.set_xticks(x)
        ax.set_xticklabels(quintile_labels, fontsize=8)
        ax.set_ylabel("Loss Ratio")
        ax.set_title("Loss Ratio by Risk Tier")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_t2:
        st.markdown("**Premium Adequacy by Risk Tier**")
        tier_grp["adequacy_ratio"] = tier_grp["premium"] / (tier_grp["pred_loss"] + 1)
        fig, ax = plt.subplots(figsize=(6, 4))
        colors  = [PALETTE[4] if v >= 1 else PALETTE[1] for v in tier_grp["adequacy_ratio"]]
        ax.bar(quintile_labels, tier_grp["adequacy_ratio"],
               color=colors, edgecolor="#0f1117")
        ax.axhline(1.0, color="white", ls="--", lw=1.5, label="Adequacy = 1.0")
        ax.set_ylabel("Premium / Predicted Loss")
        ax.set_title("Premium Adequacy Ratio by Tier")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Relativities table from OLS coefficients
    st.markdown("### Pricing Relativities (Multiplicative Factors from Log-Linear Model)")
    st.latex(
        r"\text{Relativity}_j = e^{\hat{\beta}_j}"
        r"\quad \Rightarrow \quad"
        r"\text{Premium} = \text{Base Rate} \times \prod_j \text{Relativity}_j \times (1 + \text{Expense\%})"
    )
    feat_nms  = model["feature_names"]
    beta      = model["beta_ols"][1:]   # skip intercept
    rel_df    = pd.DataFrame({
        "Feature":    feat_nms,
        "Coefficient β": [f"{b:.4f}" for b in beta],
        "Relativity eᵝ": [f"{np.exp(b):.4f}" for b in beta],
        "Direction":  ["↑ higher risk" if b > 0 else "↓ lower risk" for b in beta],
    })
    st.dataframe(rel_df, use_container_width=True, hide_index=True)

    # ── Pricing Simulator ──────────────────────────────────────────────────────
    st.markdown("### Real-Time Premium Pricing Simulator")
    with st.expander("⚙️ Configure Risk Inputs", expanded=True):
        sc1, sc2, sc3, sc4 = st.columns(4)
        sim_acc_sev  = sc1.slider("Accident Severity (1-4)", 1, 4, 2)
        sim_veh_age  = sc1.slider("Vehicle Age (years)", 0, 20, 5)
        sim_drv_age  = sc2.slider("Driver Age", 18, 85, 35)
        sim_mileage  = sc2.slider("Annual Mileage (k miles)", 1, 50, 12)
        sim_prior    = sc3.slider("Prior Claims (0-5)", 0, 5, 0)
        sim_ded      = sc3.selectbox("Deductible ($)", [500, 1000, 2500, 5000], index=1)
        sim_cov      = sc4.number_input("Coverage Amount ($)", 10000, 500000, 100000, 10000)
        sim_policy   = sc4.selectbox("Policy Type", ["auto", "home", "commercial"])
        sim_weather  = sc4.selectbox("Primary Weather", ["clear", "rain", "snow", "fog"])

    # build feature vector
    sim_x = np.array([
        sim_acc_sev,
        sim_veh_age / 20,
        (sim_drv_age - 18) / 67,
        sim_mileage * 1000 / 50_000,
        1 / 5,
        sim_prior / 5,
        np.log1p(sim_ded / 1000),
        sim_cov / 500_000,
        float(sim_policy == "commercial"),
        float(sim_policy == "home"),
        float(sim_weather == "snow"),
        float(sim_weather == "rain"),
        0.0,   # highway assumed
        0.0,   # hour_sin mid-day
        1.0,   # hour_cos mid-day
    ])
    w_logit = model["w_logit"]
    beta_ols = model["beta_ols"]

    def sigmoid_single(z):
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

    p_claim  = sigmoid_single(np.dot(np.append(1, sim_x), w_logit))
    log_sev  = np.dot(np.append(1, sim_x), beta_ols)
    exp_loss = p_claim * max(np.exp(log_sev) - 1, 0)
    base_rate = exp_loss
    final_quote = base_rate * (1 + expense_load_pct / 100)

    sq1, sq2, sq3, sq4 = st.columns(4)
    sq1.metric("P(Claim > 0)",       f"{p_claim:.1%}")
    sq2.metric("Expected Severity",  f"${max(np.exp(log_sev)-1,0):,.0f}")
    sq3.metric("Expected Loss",      f"${exp_loss:,.0f}")
    sq4.metric("Final Premium Quote",f"${final_quote:,.0f}",
               delta=f"incl. {expense_load_pct}% expense load")

# ────────────────────────────────────────────────────────────────────────────
# TAB 5 — Reserving & Capital
# ────────────────────────────────────────────────────────────────────────────
with t5:
    st.subheader("💰 Reserving & Capital Management")

    # ── IBNR summary ─────────────────────────────────────────────────────────
    st.markdown("### Chain Ladder IBNR Reserve")
    st.latex(
        r"\text{IBNR}_i = \text{Ultimate}_i - \text{Reported}_i"
        r"\quad \Rightarrow \quad"
        r"\text{Total IBNR} = \sum_i \text{IBNR}_i"
    )
    total_ultimate   = sum(ults)
    total_reported   = sum(diag)
    total_ibnr_cl    = total_ultimate - total_reported
    total_ibnr_bf    = sum(max(i, 0) for i in bf_ibnr)

    rk1, rk2, rk3, rk4, rk5 = st.columns(5)
    rk1.metric("Total Reported ($M)",  f"${total_reported/1e6:.2f}M")
    rk2.metric("CL Ultimate ($M)",     f"${total_ultimate/1e6:.2f}M")
    rk3.metric("CL IBNR ($M)",         f"${total_ibnr_cl/1e6:.2f}M")
    rk4.metric("BF IBNR ($M)",         f"${total_ibnr_bf/1e6:.2f}M")
    rk5.metric("Reserve Adequacy",
               f"{(total_premium / (total_paid+1)):.1%}",
               delta="Premium vs Losses")

    # ── Bootstrap chain ladder ────────────────────────────────────────────────
    st.markdown("### Bootstrap Chain Ladder — Reserve Distribution (100 Simulations)")
    st.latex(
        r"\hat{f}_j^{(b)} \sim \text{LogNormal}\!\left(\ln f_j,\, \sigma\right), "
        r"\quad b = 1,\dots,100"
    )
    reserves_boot = bootstrap_chain_ladder(tri_data, n_sim=100)
    p75  = float(np.percentile(reserves_boot, 75))
    p90  = float(np.percentile(reserves_boot, 90))
    p99  = float(np.percentile(reserves_boot, 99))
    p995 = float(np.percentile(reserves_boot, 99.5))

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(reserves_boot / 1e6, bins=30, color=PALETTE[3],
                edgecolor="#0f1117", alpha=0.85)
        for pct, val, col in [(75, p75, PALETTE[2]), (90, p90, PALETTE[1]), (99, p99, PALETTE[5])]:
            ax.axvline(val / 1e6, color=col, ls="--", lw=1.5, label=f"P{pct}: ${val/1e6:.2f}M")
        ax.set_xlabel("Total IBNR Reserve ($M)")
        ax.set_ylabel("Simulation Count")
        ax.set_title("Bootstrap Reserve Distribution")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_b2:
        st.markdown("**Reserve Percentiles (Solvency II Capital)**")
        pct_df = pd.DataFrame({
            "Percentile":  ["P75 (Best Estimate +)", "P90 (Risk Margin)", "P99 (SCR proxy)", "P99.5 (Solvency II SCR)"],
            "Reserve ($M)": [f"${p75/1e6:.3f}M", f"${p90/1e6:.3f}M", f"${p99/1e6:.3f}M", f"${p995/1e6:.3f}M"],
            "Purpose":     ["Management best est.", "Risk margin buffer", "1-in-100 year stress", "Regulatory capital req."],
        })
        st.dataframe(pct_df, use_container_width=True, hide_index=True)

        st.markdown("**VaR & Solvency Capital Requirement**")
        st.latex(
            r"\text{SCR} = \text{VaR}_{99.5\%}(\text{1-year losses}) - \text{Best Estimate}"
        )
        scr  = p995 - float(np.median(reserves_boot))
        best = float(np.median(reserves_boot))
        s1, s2 = st.columns(2)
        s1.metric("Best Estimate Reserve", f"${best/1e6:.3f}M")
        s2.metric("SCR (Solvency II)",     f"${scr/1e6:.3f}M",
                  delta="VaR(99.5%) − Median")

    # ── Combined Ratio Trend ──────────────────────────────────────────────────
    st.markdown("### Combined Ratio Trend — 24 Months")
    rng_cr   = np.random.default_rng(55)
    months   = 24
    lr_trend = 0.60 + rng_cr.normal(0, 0.04, months).cumsum() * 0.01
    er_trend = np.full(months, expense_load_pct / 100)
    cr_trend = (lr_trend + er_trend) * 100

    fig, ax  = plt.subplots(figsize=(12, 4))
    ax.plot(range(months), cr_trend, color=PALETTE[0], lw=2.5, label="Combined Ratio")
    ax.fill_between(range(months), cr_trend, 100, where=(cr_trend > 100),
                    alpha=0.25, color=PALETTE[1], label="Unprofitable")
    ax.fill_between(range(months), cr_trend, 100, where=(cr_trend <= 100),
                    alpha=0.25, color=PALETTE[4], label="Profitable")
    ax.axhline(100, color="white", ls="--", lw=1.5, label="Break-even 100%")
    ax.set_xlabel("Month")
    ax.set_ylabel("Combined Ratio (%)")
    ax.set_title("Combined Ratio Trend (24-Month Simulation)")
    ax.set_xticks(range(0, months, 3))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Claims Leakage ────────────────────────────────────────────────────────
    st.markdown("### Claims Leakage Analysis")
    st.caption("Claims settled above model-predicted fair value represent leakage / overpayment risk.")
    y_act_te  = model["y_act_pos"]
    y_pred_te = model["y_pred_pos"]
    over_mask = y_act_te > (y_pred_te * 1.25)   # settled >25% above predicted
    leakage_count  = over_mask.sum()
    leakage_amount = (y_act_te[over_mask] - y_pred_te[over_mask]).sum()
    normal_count   = (~over_mask).sum()

    lk1, lk2, lk3, lk4 = st.columns(4)
    lk1.metric("Claims with Leakage (>25% over)",  f"{leakage_count:,}")
    lk2.metric("Leakage Rate",                     f"{leakage_count/len(y_act_te):.1%}")
    lk3.metric("Total Leakage Amount",             f"${leakage_amount:,.0f}")
    lk4.metric("Avg Leakage per Claim",
               f"${leakage_amount/max(leakage_count,1):,.0f}")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(["Normal Settlement", "Leakage (>25% over predicted)"],
           [normal_count, leakage_count],
           color=[PALETTE[4], PALETTE[1]], edgecolor="#0f1117")
    for i, (lab, val) in enumerate(
        zip(["Normal", "Leakage"], [normal_count, leakage_count])
    ):
        ax.text(i, val + 2, f"{val:,}\n({val/len(y_act_te):.1%})",
                ha="center", color="white", fontsize=10)
    ax.set_ylabel("Claim Count")
    ax.set_title("Claims Leakage vs Normal Settlement")
    ax.grid(True, alpha=0.3, axis="y")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # final KPI summary
    st.markdown("---")
    st.markdown("### Portfolio KPI Summary")
    fk1, fk2, fk3, fk4, fk5 = st.columns(5)
    fk1.metric("Total Reserves (CL)",    f"${total_ibnr_cl/1e6:.2f}M")
    fk2.metric("Total IBNR (CL)",        f"${max(total_ibnr_cl,0)/1e6:.2f}M")
    fk3.metric("Combined Ratio (curr.)", f"{combined_ratio:.1f}%")
    fk4.metric("Loss Ratio",             f"{loss_ratio:.1%}")
    fk5.metric("Reserve Adequacy",
               f"{min((total_reported+total_ibnr_cl)/max(total_reported,1), 2.0):.2f}×")
