import pandas as pd

from src.strategy import generate_signals


def run(prices, short=2, long=3):
    df = pd.DataFrame({"Adj Close": prices})
    return generate_signals(df, SMA_short=short, SMA_long=long, verbose=False)


def test_golden_and_death_cross_detected_on_exact_days():
    # Hand-checked: row 6 SMA2=9.5 > SMA3=9.0 (row 5 was 8.5 < 8.67),
    # row 10 SMA2=10.5 < SMA3=11.0 (row 9 was 11.5 > 11.33).
    out = run([10, 10, 10, 9, 8, 9, 10, 11, 12, 11, 10, 9, 8])
    assert out["signal"].iloc[6] == 1
    assert out["signal"].iloc[10] == -1
    assert out["signal"].isin([1, -1]).sum() == 2


def test_signal_fires_only_on_the_crossover_day():
    out = run([10, 10, 10, 9, 8, 9, 10, 11, 12, 11, 10, 9, 8])
    # short stays above long for rows 6 to 9, but only row 6 is a signal
    assert (out["signal"].iloc[7:10] == 0).all()


def test_rising_warmup_does_not_fake_a_golden_cross():
    # Steadily rising prices never cross; the first valid long SMA must not
    # produce a buy signal just because short > long when it appears.
    out = run([1, 2, 3, 4, 5, 6])
    assert (out["signal"] == 0).all()


def test_no_signal_before_long_window_is_available():
    out = run([10, 10, 10, 9, 8, 9, 10, 11, 12, 11, 10, 9, 8], short=2, long=5)
    assert (out["signal"].iloc[:5] == 0).all()
    