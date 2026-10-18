"""
Walk-forward logistic-regression direction strategy.

For every date t the model sees only features known at the close of t and
predicts whether the return EARNED by the position taken under the chosen
execution lag is positive. With run_backtest's convention, position[t + lag]
earns the return from close t+lag-1 to close t+lag, so the target horizon
equals the execution lag.

Walk-forward: the model is refit every `retrain_every` trading days on a
rolling window of past rows only. Rows whose label is not yet realised at the
refit date are purged, so no training label uses information from the future.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Fixed before looking at any results. Changing any of these is a NEW variant
# and must be counted as one.
DEFAULT_CONFIG = {
    "return_lags": 5,
    "train_window": 504,     # rolling window, trading days (about 2 years)
    "retrain_every": 21,     # refit roughly monthly
    "min_train": 252,        # do not trade until this many training rows exist
    "C": 0.1,                # inverse L2 regularisation strength
    "threshold": 0.5,        # long if P(up) > threshold, else flat
}


def rsi(price, window=14):
    """Wilder's RSI in [0, 100]."""
    delta = price.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = gain / loss
    return 100 - 100 / (1 + rs)


def build_features(df, return_lags=5):
    """Features at row t use data up to and including the close of t only."""
    price = df["Adj Close"]
    ret = price.pct_change()
    feats = pd.DataFrame(index=df.index)
    for k in range(return_lags):
        feats[f"ret_lag{k}"] = ret.shift(k)          # ret_lag0 = today's return
    feats["vol_20"] = ret.rolling(20).std()
    feats["rsi_14"] = rsi(price, 14)
    feats["price_sma20"] = price / price.rolling(20).mean() - 1
    feats["price_sma50"] = price / price.rolling(50).mean() - 1
    return feats


def build_target(df, horizon):
    """1 if the return earned `horizon` days ahead is positive, NaN if unknown."""
    fwd = df["Adj Close"].pct_change().shift(-horizon)
    return (fwd > 0).astype(float).where(fwd.notna())


def walk_forward_probabilities(
    features, target, eval_start, horizon,
    train_window=504, retrain_every=21, min_train=252, C=0.1, **_ignored,
):
    """P(up) for each date from eval_start on; NaN before that or if untrainable."""
    X = features.to_numpy(dtype=float)
    y = target.to_numpy(dtype=float)
    valid_x = ~np.isnan(X).any(axis=1)
    valid_y = ~np.isnan(y)

    n = len(features)
    eval_pos = np.flatnonzero(features.index >= pd.Timestamp(eval_start))
    if len(eval_pos) == 0:
        raise ValueError("eval_start is after the last date")

    probs = np.full(n, np.nan)
    for block_start in range(eval_pos[0], n, retrain_every):
        block = np.arange(block_start, min(block_start + retrain_every, n))

        # The label of row s is realised at close s+horizon. It must be known at
        # block_start, so only rows s <= block_start - horizon may be used.
        last_train = block_start - horizon
        first_train = max(0, last_train - train_window + 1)
        idx = np.arange(first_train, last_train + 1)
        idx = idx[valid_x[idx] & valid_y[idx]]
        if len(idx) < min_train or len(np.unique(y[idx])) < 2:
            continue

        model = make_pipeline(
            StandardScaler(), LogisticRegression(C=C, max_iter=1000)
        )
        model.fit(X[idx], y[idx].astype(int))

        pred = block[valid_x[block]]
        if len(pred):
            probs[pred] = model.predict_proba(X[pred])[:, 1]

    return pd.Series(probs, index=features.index, name="p_up")


def position_state(probs, threshold=0.5):
    """1 (long) where P(up) > threshold, else 0. NaN probabilities mean flat."""
    return (probs > threshold).astype(int)


def signals_from_probabilities(probs, threshold=0.5):
    """
    Convert the long/flat state to the +1 / -1 / 0 event signal run_backtest expects.
    The first row carries the initial state, so a state that starts long is not lost.
    """
    state = position_state(probs, threshold)
    signal = state.diff()
    signal.iloc[0] = state.iloc[0]
    return signal