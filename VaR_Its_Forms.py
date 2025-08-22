#!/usr/bin/env python
# coding: utf-8

# # Logging

# 
# # --Value-at-Risk (VaR) — Why it matters
# 
# """
# Value-at-Risk (VaR) is a distribution-aware, single-number summary of market risk.
# It answers: “Over a chosen horizon T, what is the maximum loss L I should not exceed with confidence α?”
# 
# 
# # Why finance uses VaR
# - Provides a common “risk budget” across assets, desks, and portfolios.
# - Links directly to limits, capital allocation, and backtesting/reporting.
# - Improves on variance/standard deviation by focusing on downside tail risk.
# 
# # Methods covered in this project
# - Parametric (variance–covariance; Normal/Student-t assumptions)
# - Historical (empirical quantiles from past returns)
# - Monte Carlo (simulate returns under a chosen model)
# - Modified (Cornish–Fisher adjustment to account for skew/kurtosis)
# 
# # Scope
# - Works for single assets and weighted portfolios.
# - Produces comparable 1-day VaR figures at configurable confidence levels.
# 

# import pandas as pd
# import yfinance as yf
# import logging
# from typing import Iterable
# from dataclasses import dataclass

# In[1]:


import pandas as pd
import yfinance as yf
import logging
from typing import Iterable
from dataclasses import dataclass
import numpy as np
from scipy import stats


# In[2]:


import logging

logging.basicConfig(
    level=logging.INFO,  # niveau minimal affiché
    format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


# # Global configuration parameters for market data retrieval
# 

# In[3]:


from dataclasses import dataclass

@dataclass
class DataConfig:
    period: str = "10y"
    interval: str = "1d"
    auto_adjust: bool = True
    backfill_limit: int = 5


# In[4]:


# -----------------------------------------------------------------------------
# 1) Download & preprocess market data
# -----------------------------------------------------------------------------
from typing import Iterable
def fetch_df(
    tickers: Iterable[str],
    cfg: DataConfig = DataConfig()
) -> pd.DataFrame:
    """
    Download OHLCV data via yfinance and return a DataFrame of 'Adj Close' prices.
    Handles multiple tickers gracefully, aligns dates, and keeps only business days.
    """
    
    # (1) Ensure tickers is a list + logging for traceability
    tickers = list(tickers)
    logger.info(f"Downloading: {tickers} | period={cfg.period} | interval={cfg.interval}")
    
    # (2) Fetch raw data from yfinance
    df = yf.download(
        tickers=" ".join(tickers),
        period=cfg.period,
        interval=cfg.interval,
        auto_adjust=cfg.auto_adjust,
        progress=False,
        threads=True,
    )
    price_col = "Close" if cfg.auto_adjust else "Adj Close"
    if df.empty:
        raise ValueError(f"Aucune donnée téléchargée pour {tickers}. Vérifie tickers/période/intervalle.")


    # (3) Extract adjusted prices depending on auto_adjust
    if isinstance(df.columns, pd.MultiIndex):  # Multiple tickers
        adj = df[price_col].copy()
    else:  # Single ticker
        adj = df.rename(columns={price_col: tickers[0]})[tickers[0]].to_frame()


    # (4) Enforce a clean business daily calendar
    adj = adj.sort_index().asfreq("B")

    # (5) Fill missing values forward/backward (with a max limit)
    adj = adj.ffill(limit=cfg.backfill_limit).bfill(limit=cfg.backfill_limit)

    # (6) Check for remaining NaN values and log a warning if needed
    missing = adj.isna().sum().sum()
    if missing > 0:
        logger.warning(f"Missing values after ffill/bfill: {missing}")

    # (7) Return cleaned and ready-to-use DataFrame
    return adj


# In[5]:


logger = logging.getLogger(__name__)


# In[6]:


tickers = ["MC.PA", "TTE.PA", "VOW3.DE", "NESN.SW"]
df = fetch_df(tickers)
print(df.tail())


# In[7]:


df.head()


# In[10]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(df["MC.PA"], color="blue", label="MC.PA")
plt.title("MC.PA Stock Price", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()


# In[11]:


plt.figure(figsize=(12,6))
for ticker in df.columns:
    plt.plot(df[ticker], label=ticker)

plt.title("Stock Prices of Selected Tickers", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()


# In[12]:


import numpy as np
df = np.log(df).diff().dropna(how="all")
df.head()


# In[13]:


plt.figure(figsize=(10,5))
plt.plot(df["MC.PA"], color="blue", label="MC.PA")
plt.title("MC.PA Stock return", fontsize=14)
plt.xlabel("Date")
plt.ylabel("return")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()


# In[14]:


plt.figure(figsize=(12,6))
for ticker in df.columns:
    plt.plot(df[ticker], label=ticker)

plt.title("Stock return of Selected Tickers", fontsize=14)
plt.xlabel("Date")
plt.ylabel("return")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()


# ### 1.1 Parametric VaR
# 
# The **variance–covariance (parametric)** method assumes returns are normally distributed.  
# We estimate the mean return μ and volatility σ, then use the Normal quantile zₐ for confidence level α.
# 
# A common (loss-positive) convention is:
# VaRₐ = V · (zₐ σ − μ)
# 
# **Confidence Level vs. Parametric VaR (per-unit):**
# 
# | Confidence Level | zₐ    | Per-unit VaR expression     |
# |:--:|:--:|:--|
# | 90% | 1.2816 | μ − 1.2816·σ |
# | 95% | 1.6449 | μ − 1.6449·σ |
# | 99% | 2.3263 | μ − 2.3263·σ |
# 
# **Where:** μ = mean return; σ = volatility (standard deviation of returns); zₐ = standard Normal quantile (number of standard deviations from the mean) for confidence level α.
# 

# In[15]:


alphas = [0.90, 0.95, 0.99]

def var_row(s):
    r = s.dropna().values
    m = np.mean(r)
    sd = np.std(r, ddof=0)

    return pd.Series({
        "VaR_90": stats.norm.ppf(1 - 0.90, loc=m, scale=sd),
        "VaR_95": stats.norm.ppf(1 - 0.95, loc=m, scale=sd),
        "VaR_99": stats.norm.ppf(1 - 0.99, loc=m, scale=sd),
    })

var_table = df.apply(var_row, axis=0).T  # index=tickers, columns=90/95/99
print(var_table)


# In[18]:


# --- Tests ---
from scipy.stats import chi2
def kupiec_test(returns, var, alpha):
    exc = (returns < var).astype(int)
    x = int(exc.sum())
    N = int(exc.count())
    pi_hat = x / N if N > 0 else np.nan
    # log-likelihood ratio (POF)
    LR = -2 * ((N-x)*np.log(1-alpha) + x*np.log(alpha)
               - (N-x)*np.log(1-pi_hat) - x*np.log(pi_hat))
    pval = 1 - chi2.cdf(LR, df=1)
    return {"exceptions": x, "N": N, "ratio": pi_hat, "LR_pof": LR, "p_value": pval}

def christoffersen_test(returns, var):
    exc = (returns < var).astype(int).dropna().values
    n00=n01=n10=n11=0
    for i in range(1, len(exc)):
        a, b = exc[i-1], exc[i]
        n00 += (a==0 and b==0)
        n01 += (a==0 and b==1)
        n10 += (a==1 and b==0)
        n11 += (a==1 and b==1)
    pi0 = n01 / (n00+n01) if (n00+n01)>0 else 0.0
    pi1 = n11 / (n10+n11) if (n10+n11)>0 else 0.0
    pi  = (n01+n11) / max(n00+n01+n10+n11, 1)
    L0 = ((1-pi)**(n00+n10)) * (pi**(n01+n11))
    L1 = ((1-pi0)**n00) * (pi0**n01) * ((1-pi1)**n10) * (pi1**n11)
    if L0<=0 or L1<=0:
        return {"LR_ind": np.nan, "p_value": np.nan, "counts": {"n00":n00,"n01":n01,"n10":n10,"n11":n11}}
    LR = -2*np.log(L0/L1)
    pval = 1 - chi2.cdf(LR, df=1)
    return {"LR_ind": LR, "p_value": pval, "counts": {"n00":n00,"n01":n01,"n10":n10,"n11":n11}}

# --- Backtest "fixe" sur ta var_table ---
levels = {"VaR_90":0.10, "VaR_95":0.05, "VaR_99":0.01}
results_fixed = []

for t in var_table.index:
    r = df[t].dropna()
    for col, alpha in levels.items():
        v = var_table.loc[t, col]                # seuil constant (retour négatif attendu)
        out_k = kupiec_test(r, v, alpha)
        out_c = christoffersen_test(r, v)
        results_fixed.append({
            "Ticker": t, "Level": col,
            "Exc": out_k["exceptions"], "N": out_k["N"], "Rate": out_k["ratio"],
            "Kupiec_LR": out_k["LR_pof"], "Kupiec_p": out_k["p_value"],
            "Christ_LR": out_c["LR_ind"], "Christ_p": out_c["p_value"],
        })

bt_fixed_df = pd.DataFrame(results_fixed)
print(bt_fixed_df)


# In[19]:


import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Convert to positive loss-style VaR
loss_var = -var_table  # columns: '90%', '95%', '99%'; index: tickers

ax = loss_var.plot(kind="bar")
ax.set_title("Parametric VaR by Ticker and Confidence")
ax.set_xlabel("Ticker")
ax.set_ylabel("1-day VaR")
ax.yaxis.set_major_formatter(PercentFormatter(1.0))  # 0.028 -> 2.8%
ax.legend(title="Confidence")
plt.tight_layout()
plt.show()


# In[20]:


import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

loss_var = -var_table.copy()
loss_var.index.name = loss_var.index.name or "Ticker"

long = loss_var.reset_index().melt(
    id_vars="Ticker", var_name="Confidence", value_name="VaR"
)
# Ensure the 90–95–99 order on the x-axis
long["Confidence"] = long["Confidence"].str.extract(r'(\d+)$')[0] + "%"
long["Confidence"] = pd.Categorical(long["Confidence"], ["90%", "95%", "99%"], ordered=True)


plt.figure()
for t, d in long.groupby("Ticker"):
    plt.plot(d["Confidence"], d["VaR"], marker="o", label=t)

plt.title("Parametric VaR vs Confidence")
plt.ylabel("1-day VaR")
plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
plt.legend(title="Ticker")
plt.tight_layout()
plt.show()


# # CDF vs PPF
# 
# cdf(z) gives the probability up to z.
# 
# ppf(p) does the reverse: gives the z that corresponds to cumulative probability p.

# In[21]:


# number of stdev from the mean
print(stats.norm.ppf(0.01))


# ### 1.2 Historical VaR
# 
# **Idea.** Historical VaR estimates downside risk directly from past returns, without assuming a distribution.
# 
# **How it works:**
# 1) Take a window of daily returns.  
# 2) Sort returns ascending (worst → best).  
# 3) Pick the left-tail percentile **p = 1 − α** (e.g., **α = 95% ⇒ p = 5%**).  
# 4) That percentile is the historical VaR in return space (often negative). Many reports flip the sign to show a positive loss.
# 
# **Interpretation.** A 1-day **95%** historical VaR is the empirical **5th percentile** of daily returns; losses exceed it on ~5% of days.
# 
# **Pros.** Captures fat tails/outliers; simple.  
# **Cons.** Backward-looking; sensitive to window choice; jumps when extremes enter/leave the sample.
# 

# In[22]:


R = df[tickers].dropna(how="all") 
# Historical VaR (left-tail empirical quantiles of returns; often negative)
hvar_table = pd.DataFrame({
    "hVaR_90": R.quantile(0.10),   # 1 - 90%
    "hVaR_95": R.quantile(0.05),   # 1 - 95%
    "hVaR_99": R.quantile(0.01),   # 1 - 99%
}).loc[tickers]                   # keep desired row order

hvar_table.index.name = "Ticker"
print(hvar_table.round(6))


# In[23]:


# Positive “loss-style” numbers
loss_hvar = (-hvar_table).rename(columns={
    "hVaR_90": "90%",
    "hVaR_95": "95%",
    "hVaR_99": "99%",
})

ax = loss_hvar.plot(kind="bar")
ax.set_title("Historical VaR by Ticker and Confidence")
ax.set_xlabel("Ticker")
ax.set_ylabel("1-day VaR")
ax.yaxis.set_major_formatter(PercentFormatter(1.0))  # 0.028 -> 2.8%
ax.legend(title="Confidence")
plt.tight_layout()
plt.show()


# In[24]:


from matplotlib.ticker import PercentFormatter
# Convert to positive “loss-style” and reshape to long format
loss_hvar = (-hvar_table).copy()                    # keep index = tickers
loss_hvar.index.name = loss_hvar.index.name or "Ticker"

# Melt with the original column names, then map to clean labels
long_h = loss_hvar.reset_index().melt(
    id_vars="Ticker", var_name="ConfidenceRaw", value_name="VaR"
)
label_map = {"hVaR_90": "90%", "hVaR_95": "95%", "hVaR_99": "99%"}
long_h["Confidence"] = long_h["ConfidenceRaw"].map(label_map)
long_h["Confidence"] = pd.Categorical(long_h["Confidence"], ["90%", "95%", "99%"], ordered=True)

plt.figure()
for t, d in long_h.groupby("Ticker"):
    plt.plot(d["Confidence"], d["VaR"], marker="o", label=t)

plt.title("Historical VaR vs Confidence")
plt.ylabel("1-day VaR")
plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
plt.legend(title="Ticker")
plt.tight_layout()
plt.show()


# ### 1.3 Monte Carlo VaR
# 
# **Idea.** Monte Carlo VaR estimates downside risk by **simulating many possible return outcomes** under a chosen model, then reading the loss at the left-tail confidence level.
# 
# **How it works (single asset or portfolio):**
# 1) Choose a model for returns (e.g., Normal with mean μ and volatility σ; or a richer model).
# 2) Calibrate μ and σ from historical daily returns.
# 3) Generate **N** simulated daily returns (or paths for a longer horizon).
# 4) Convert returns to P&L (apply position value and/or portfolio weights).
# 5) Sort simulated outcomes and take the **left-tail percentile** \(p = 1 − α\) (e.g., 5% for 95% VaR).
# 6) Report VaR as a **positive loss** (the magnitude of that percentile loss).
# 
# **Interpretation.** A 1-day 95% Monte Carlo VaR is the **5th percentile** of simulated daily P&L: under the model, losses exceed this level on ~5% of days.
# 
# **Why use it.**
# - **Pros:** flexible; can handle non-linear instruments, path dependence, and non-Normal dynamics; easy to extend to portfolios.
# - **Cons:** **model risk** (results depend on model/specification); **simulation error** (needs enough scenarios); can be computationally heavier.
# 
# **Practical tips.**
# - Set a random seed for reproducibility.
# - Use a sufficiently large number of simulations (e.g., 10k–100k for stable tails).
# - Validate the model (backtesting, stress scenarios) and compare with Historical/Parametric VaR.
# 

# In[25]:


# 1) Select tickers and returns matrix
tickers = ["MC.PA", "NESN.SW", "TTE.PA", "VOW3.DE"]
R = df[tickers].dropna(how="all")   # df must be DAILY RETURNS

# 2) Fit Normal(μ,σ) per ticker and simulate
mu = R.mean()
sd = R.std(ddof=0)

n_sims = 10000
rng = np.random.default_rng(42)     # seed for reproducibility
sims = {t: rng.normal(loc=mu[t], scale=sd[t], size=n_sims) for t in tickers}

# 3) Build the table of left-tail percentiles (return quantiles; often negative)
mcvar_table = pd.DataFrame({
    "MCVaR_90": pd.Series({t: np.percentile(sims[t], 10) for t in tickers}),  # 1 - 90%
    "MCVaR_95": pd.Series({t: np.percentile(sims[t],  5) for t in tickers}),  # 1 - 95%
    "MCVaR_99": pd.Series({t: np.percentile(sims[t],  1) for t in tickers}),  # 1 - 99%
}).loc[tickers]

mcvar_table.index.name = "Ticker"
print(mcvar_table.round(6))


# In[26]:


# Assumes mcvar_table exists with columns: MCVaR_90, MCVaR_95, MCVaR_99
loss_mcvar = (-mcvar_table).rename(columns={
    "MCVaR_90": "90%",
    "MCVaR_95": "95%",
    "MCVaR_99": "99%",
}).copy()
loss_mcvar.index.name = loss_mcvar.index.name or "Ticker"

# ---------- Plot 1: Grouped bars ----------
ax = loss_mcvar.plot(kind="bar")
ax.set_title("Monte Carlo VaR by Ticker and Confidence")
ax.set_xlabel("Ticker")
ax.set_ylabel("1-day VaR")
ax.yaxis.set_major_formatter(PercentFormatter(1.0))  # 0.028 -> 2.8%
ax.legend(title="Confidence")
plt.tight_layout()
plt.show()

# ---------- Plot 2: Lines vs confidence ----------
long = loss_mcvar.reset_index().melt(
    id_vars="Ticker", var_name="Confidence", value_name="VaR"
)
long["Confidence"] = pd.Categorical(long["Confidence"], ["90%","95%","99%"], ordered=True)

plt.figure()
for t, d in long.groupby("Ticker"):
    plt.plot(d["Confidence"], d["VaR"], marker="o", label=t)

plt.title("Monte Carlo VaR vs Confidence")
plt.ylabel("1-day VaR")
plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
plt.legend(title="Ticker")
plt.tight_layout()
plt.show()


# ### 1.4 Normality Test
# 
# In parametric VaR we assume returns are approximately Normal. Real-world returns often show skewness and fat tails.
# We perform a quick normality check using the **Shapiro–Wilk** test.
# 
# **Shapiro–Wilk (W, p-value).**  
# - **H₀:** the sample comes from a Normal distribution.  
# - **Decision rule:** if *p* < 0.05, reject Normality at the 5% level.
# 
# > Note: Shapiro–Wilk is most reliable for sample sizes up to ~5,000. For larger samples we test on a random subsample.
# 

# In[27]:


# If your returns DataFrame is named something else, replace `df` below.
tickers = ["MC.PA", "NESN.SW", "TTE.PA", "VOW3.DE"]
R = df[tickers].dropna(how="all")   # df must be DAILY RETURNS

def shapiro_for_series(s: pd.Series, max_n: int = 5000, seed: int = 42) -> pd.Series:
    """
    Shapiro–Wilk normality test.
    H0: data are drawn from a Normal distribution.
    Subsample to max_n for numerical stability (Shapiro is best ≤ ~5000 obs).
    """
    x = s.dropna().to_numpy()
    n = x.size
    if n < 3:
        return pd.Series({"W": np.nan, "p_value": np.nan, "n_used": n, "Decision@5%": "Insufficient data"})
    if n > max_n:
        rng = np.random.default_rng(seed)
        x = rng.choice(x, size=max_n, replace=False)
        n_used = max_n
    else:
        n_used = n
    W, p = stats.shapiro(x)
    return pd.Series({"W": float(W), "p_value": float(p), "n_used": int(n_used),
                      "Decision@5%": "Reject Normality" if p < 0.05 else "Do not reject"})

shapiro_table = R.apply(shapiro_for_series, axis=0).T
shapiro_table.index.name = "Ticker"
print(shapiro_table.round(6))


# ### 1.4 Normality Test — Interpretation
# 
# **Null hypothesis (H₀):** the stock’s daily returns follow a Normal distribution.  
# **Result:** since the p-value is **< 0.05**, we **reject H₀** at the 5% level.  
# **Conclusion:** there is sufficient evidence that the sample of daily returns **does not come from a Normal distribution**.
# 
# This outcome is not surprising: observed returns are drawn from an **empirical distribution** that often exhibits **skewness** and **fat tails**, so Normality is frequently violated in practice.
# 
# ---
# 
# ### Anderson–Darling (alternative check)
# 
# The **Anderson–Darling** test is a goodness-of-fit test that evaluates how well data follow a specified distribution (commonly Normal).  
# - It gives a **test statistic** and **critical values** at standard significance levels.  
# - **Decision rule:** if the statistic **exceeds** the critical value at 5%, **reject Normality**.  
# - Compared with general Normality tests, Anderson–Darling puts **more weight in the tails**, which is often desirable for risk analysis.
# 
# *Implication for VaR:* because Normality is rejected, rely on **Historical VaR**, **Modified VaR (Cornish–Fisher)**, or **t-distribution/robust parametric** approaches alongside (or instead of) the basic Normal parametric VaR.
# 

# In[28]:


# Jarque–Bera: H0 = Normal with unspecified mean/variance (tests skew & kurtosis)
def jb_for_series(s: pd.Series) -> pd.Series:
    x = s.dropna().to_numpy()
    if x.size < 3:
        return pd.Series({"JB_stat": np.nan, "JB_p_value": np.nan})
    jb_stat, jb_p = stats.jarque_bera(x)
    return pd.Series({"JB_stat": float(jb_stat), "JB_p_value": float(jb_p)})

jb_table = R.apply(jb_for_series, axis=0).T
jb_table.index.name = "Ticker"
print("\nJarque–Bera results:\n", jb_table.round(6))

# Anderson–Darling: compare statistic to 5% critical value
from scipy.stats import anderson

def ad_for_series(s: pd.Series) -> pd.Series:
    x = s.dropna().to_numpy()
    if x.size < 5:
        return pd.Series({"A2": np.nan, "crit_5%": np.nan, "Decision@5%": "Insufficient data"})
    res = anderson(x, dist="norm")
    # get the 5% critical value (or the closest available level)
    levels = np.array(res.significance_level, dtype=float)
    idx = int(np.argmin(np.abs(levels - 5.0)))
    crit = float(res.critical_values[idx])
    decision = "Reject Normality" if res.statistic > crit else "Do not reject"
    return pd.Series({"A2": float(res.statistic), "crit_5%": crit, "Decision@5%": decision})

ad_table = R.apply(ad_for_series, axis=0).T
ad_table.index.name = "Ticker"
print("\nAnderson–Darling results:\n", ad_table.round(6))


# In[29]:


import plotly.io as pio
import plotly.express as px
pio.renderers.default = "browser"   # or "vscode" if you're in VS Code
# partons des rendements journaliers R
long = R.reset_index().melt(
    id_vars="Date", var_name="Ticker", value_name="Return"
)

# your faceted histogram
fig = px.histogram(
    long, x="Return", facet_col="Ticker", facet_col_wrap=2,
    histnorm="probability density", nbins=60,
    title="Histogram of Daily Returns by Ticker (density)"
)
fig.update_layout(showlegend=False, xaxis_title="Daily return", yaxis_title="Density")
fig.show()  # opens in your default browser


# In[30]:


import pandas as pd
import plotly.express as px

tickers = ["MC.PA", "NESN.SW", "TTE.PA", "VOW3.DE"]

# If your returns DF is named `returns`, replace `df` by `returns` below.
R = df[tickers].dropna(how="all")

# Long (tidy) format for Plotly
long = R.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="Return")

fig = px.histogram(
    long, x="Return", color="Ticker",
    histnorm="probability density",  # plots density instead of counts
    nbins=60, opacity=0.6,
    title="Histogram of Daily Returns (density)"
)
fig.update_layout(
    barmode="overlay",            # draw histograms overlaid
    legend_title_text="Ticker",
    xaxis_title="Daily return",
    yaxis_title="Density"
)
fig.show()


# In[31]:


fig = px.histogram(
    long, x="Return", facet_col="Ticker", facet_col_wrap=2,
    histnorm="probability density", nbins=60,
    title="Histogram of Daily Returns by Ticker (density)"
)
fig.update_layout(
    showlegend=False,
    xaxis_title="Daily return",
    yaxis_title="Density"
)
fig.show()


# ### 1.5 Modified VaR (Cornish–Fisher)
# 
# Standard normal returns have mean 0, variance 1, skewness 0, and kurtosis 3. In practice, many assets show non-zero skewness and excess kurtosis, so Modified VaR uses all four moments (mean, variance, skewness, kurtosis) instead of only the first two.
# 
# **Definition (loss-style):**  
# mVaR = V · (μ − t · σ)
# 
# **Cornish–Fisher adjusted quantile (no LaTeX):**  
# t = z + (1/6)·(z<sup>2</sup> − 1)·s + (1/24)·(z<sup>3</sup> − 3z)·k − (1/36)·(2z<sup>3</sup> − 5z)·s<sup>2</sup>
# 
# **Where:**  
# - V = position value  
# - μ = mean return  
# - σ = volatility (standard deviation of returns)  
# - s = skewness  
# - k = excess kurtosis (kurtosis − 3)  
# - z = standard Normal quantile at left-tail probability p = 1 − α (e.g., p = 0.05 for 95% VaR)
# 

# In[32]:


# --- Cornish–Fisher helpers ---
def cornish_fisher_t(z, s, k):
    """Cornish–Fisher adjustment; k = EXCESS kurtosis."""
    return (
        z
        + (1/6)*(z**2 - 1)*s
        + (1/24)*(z**3 - 3*z)*k
        - (1/36)*(2*z**3 - 5*z)*(s**2)
    )

def modified_var_series(r: pd.Series, alpha: float = 0.99, as_loss: bool = True) -> float:
    """
    Modified VaR (Cornish–Fisher) for one return series.
    Returns a positive loss if as_loss=True, else the left-tail return quantile.
    """
    x = pd.Series(r).dropna().values
    if x.size < 3:
        return np.nan
    mu  = float(np.mean(x))
    sig = float(np.std(x, ddof=0))
    if not np.isfinite(sig) or sig <= 0:
        return np.nan

    s = float(stats.skew(x, bias=False))
    k = float(stats.kurtosis(x, fisher=True, bias=False))  # EXCESS kurtosis
    z = float(stats.norm.ppf(1 - alpha))                   # left-tail quantile (negative)
    t = cornish_fisher_t(z, s, k)
    q_left = mu + sig * t                                  # return quantile (often negative)
    return float(-q_left if as_loss else q_left)


# In[33]:


tickers = ["MC.PA", "NESN.SW", "TTE.PA", "VOW3.DE"]
R = df[tickers].dropna(how="all")

mvar_table = pd.DataFrame({
    "mVaR_90": [modified_var_series(R[t], 0.90, as_loss=False) for t in tickers],
    "mVaR_95": [modified_var_series(R[t], 0.95, as_loss=False) for t in tickers],
    "mVaR_99": [modified_var_series(R[t], 0.99, as_loss=False) for t in tickers],
}, index=tickers)
mvar_table.index.name = "Ticker"

print("Modified VaR (Cornish–Fisher) — return quantiles (negative = loss):")
print(mvar_table.round(6))

# If you prefer positive loss-style numbers:
print("\nModified VaR — positive loss style:")
print((-mvar_table).round(6))


# ### 1.6 Scaling VaR
# 
# To estimate VaR over a multi-day horizon **T**, we use the **square-root-of-time** rule (assuming daily returns are i.i.d.):
# 
# **Loss-style VaR over T days (per position value V):**  
# VaR_T = V · ( z_α · σ · √T − μ · T )
# 
# **Where:**  
# - V = position value  
# - μ = daily mean return  
# - σ = daily volatility (standard deviation of returns)  
# - z_α = Normal z-score for confidence level α (e.g., 1.645 for 95%)  
# - T = horizon in trading days (e.g., T = 5 for 1 week)
# 
# **Shortcut:** if μ is small relative to z_α·σ, a common approximation is  
# VaR_T ≈ VaR_1day · √T
# 
# **Note:** The √T rule relies on i.i.d./Normal assumptions; with autocorrelation or volatility clustering (GARCH-like behavior), it can under- or over-state risk.
# 

# In[34]:


forecast_days = 5
sqrtT = np.sqrt(forecast_days)

# --- Choose your 1-day table here ---
one_day = var_table.copy()       # or: hvar_table, mvar_table

# Normalize column names to a common set
col_map = {
    "VaR_90":"90%","VaR_95":"95%","VaR_99":"99%",
    "hVaR_90":"90%","hVaR_95":"95%","hVaR_99":"99%",
    "mVaR_90":"90%","mVaR_95":"95%","mVaR_99":"99%",
}
one_day = one_day.rename(columns={k:v for k,v in col_map.items() if k in one_day.columns})
one_day = one_day[["90%","95%","99%"]]   # keep order

# If your 1-day numbers are NEGATIVE return-quantiles (e.g., -2.8%), scale them directly:
scaled = one_day * sqrtT

# If your 1-day numbers are POSITIVE loss-style, scale those instead:
# scaled = one_day * sqrtT

# Combined table (all tickers × confidences)
scaled.index.name = "Ticker"
print(scaled.round(6))


# ### 1.7 Expected Shortfall (ES) — a.k.a. CVaR
# 
# **Why:** VaR is a threshold. It tells you the loss you should not exceed with confidence α, but it says nothing about the **average size of losses beyond that threshold**. If returns are non-Normal (fat tails), VaR can understate tail risk.
# 
# **Definition (historical view):**  
# Expected Shortfall at confidence level α is the **average loss in the worst (1 − α) fraction of days**.
# 
# - Return-space definition:  
#   `ES_α = average of returns R such that R ≤ hVaR_α`  
#   (This is typically a **negative** number, since returns in the tail are losses.)
# 
# - Loss-style definition (positive number):  
#   `ES_α(loss) = average of (−R) given R ≤ hVaR_α`
# 
# **Interpretation:**  
# At 95%, ES is the **mean loss** of the worst 5% days. ES is tail-sensitive and is a **coherent** risk measure; in loss terms it is usually ≥ VaR.
# 
# **Notation used here:**  
# - `R` = daily return  
# - `hVaR_α` = historical VaR at level α (left-tail quantile)  
# - `α` = confidence level (e.g., 90%, 95%, 99%)
# 

# In[35]:


# Your four tickers (daily returns in df)
tickers = ["MC.PA", "NESN.SW", "TTE.PA", "VOW3.DE"]
R = df[tickers].dropna(how="all")   # df must contain DAILY RETURNS

# Left-tail probs for α = 90/95/99%
levels = {"CVaR_90": 0.10, "CVaR_95": 0.05, "CVaR_99": 0.01}

def es_one(series: pd.Series, p: float) -> float:
    """Historical ES (return-space): mean of returns ≤ p-quantile."""
    s = series.dropna()
    if s.empty:
        return np.nan
    thr = s.quantile(p)
    tail = s[s <= thr]
    return float(tail.mean()) if not tail.empty else np.nan

# Build the table: rows=tickers, cols=CVaR_90/95/99 (return-space; usually negative)
cvar_return = pd.DataFrame({
    name: R.apply(lambda s, q=q: es_one(s, q), axis=0)
    for name, q in levels.items()
}).loc[tickers]
cvar_return.index.name = "Ticker"

print("Historical CVaR (return-space; negative = average tail return):")
print(cvar_return.round(6))

# If you prefer positive loss-style numbers:
cvar_loss = -cvar_return
print("\nHistorical CVaR (positive loss style):")
print(cvar_loss.round(6))

# (Optional) Scale to money for notional V per ticker
# V = 100_000
# print("\nHistorical CVaR in money (V = 100,000):")
# print((cvar_loss * V).round(2))


# ## GARCH
# 
# Asset price volatility is central to derivatives pricing. It is defined as a measure of price variability over a certain period of time. In essence, it describes the standard deviation of returns.  
# 
# There are different types of volatility: **Historical, Implied, Forward**.  
# 
# In most cases, we assume volatility to be constant, which is clearly not true, and numerous studies have been dedicated to estimate this variable, both in academia and industry.
# 

# ## Volatility  
# 
# Volatility estimation by statistical means assumes equal weights to all returns measured over the period.  
# We know that over 1-day, the mean return is small as compared to the standard deviation.  
# 
# If we consider a simple m-period moving average, where σₙ is the volatility of return on day n, then with μ ≈ 0, we have:  
# 
# σₙ² = (1/m) ∑ᵢ₌₁ᵐ uₙ₋ᵢ²  
# 
# where u is the return and σ² is the variance.  
# 

# In[36]:


# Import arch library
from arch import arch_model


# In[37]:


# Mean zero
stockreturn = df["MC.PA"].dropna()
g1 = arch_model(stockreturn, vol='GARCH', mean='Zero', p=1, q=1, dist='Normal')
model = g1.fit()


# In[38]:


# Model output
print(model)


# In[39]:


tickers = ["MC.PA", "NESN.SW", "TTE.PA", "VOW3.DE"]

# Sélection des rendements
R = df[tickers].dropna(how="any")

# Estimation GARCH(1,1) mean=Zero, dist=Normal (on met en % pour une meilleure convergence)
results = {}
for t in tickers:
    y = (R[t].dropna() * 100)  # rendements en %
    am = arch_model(y, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
    res = am.fit(disp="off")
    results[t] = res
    print(f"\n=== {t} ===")
    print(res.summary())


# In[40]:


# Extract the parameters from each fitted model and store them in a DataFrame
params_table = pd.DataFrame({t: results[t].params for t in tickers}).T

# Set the index name for clarity
params_table.index.name = "Ticker"

# Display the table
print("GARCH(1,1) parameters for each asset:")
print(params_table)


# In[41]:


# Collect 99% confidence intervals for all parameters across tickers
ci_table = {t: results[t].conf_int(alpha=0.01) for t in tickers}
ci_df = pd.concat(ci_table, axis=0)
ci_df.index.names = ["Ticker", "Parameter"]

print("99% Confidence Intervals for GARCH(1,1) parameters:")
print(ci_df)


# In[42]:


for t in tickers:
    print(f"\n=== Annualized volatility plot for {t} ===")
    fig = results[t].plot(annualize='D')
    plt.show()


# In[43]:


# Forecast horizon (e.g., 60 days)
horizon = 60

# Collect forecasts for each ticker
fcasts = {}
for t in tickers:
    fc = results[t].forecast(horizon=horizon)
    # Convert variance → daily volatility, annualize (252 days), scale to %
    vol = np.sqrt(fc.variance.values[-1, :] * 252) * 100
    fcasts[t] = vol

# Build a DataFrame with horizons as index
fdf = pd.DataFrame(fcasts, index=np.arange(1, horizon+1))
fdf.index.name = "Horizon"

# Convert to long format for Plotly
fdf_long = fdf.reset_index().melt(id_vars="Horizon", var_name="Ticker", value_name="Cond_Vol")

# Plot forecasted volatility paths
fig = px.line(
    fdf_long, x="Horizon", y="Cond_Vol", color="Ticker",
    title="GARCH(1,1) Volatility Forecast (Annualized, %)",
    labels={"Cond_Vol": "Conditional Volatility (%)"}
)
fig.show()


# In[ ]:




