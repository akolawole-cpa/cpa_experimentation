"""
Market Matching for Geo Experiments.

When running geographic experiments (e.g., testing a campaign in specific
markets), selecting appropriate control markets is crucial. This module
provides methods to find the best control markets based on:

1. Dynamic Time Warping (DTW) distance - captures similar patterns
2. Euclidean distance - simple distance measure
3. Correlation-based distance - captures linear relationships

Key concepts:
- Treatment markets: Where the intervention is applied
- Control markets: Similar markets without intervention (for comparison)
- Pre-period: Historical data used for matching
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class MarketMatchResult:
    """Results from market matching."""

    treatment_market: str
    control_markets: List[str]
    distances: Dict[str, float]
    weights: Dict[str, float]
    correlation_matrix: pd.DataFrame
    best_match: str
    match_quality: float  # 0-1, higher is better

    def __repr__(self) -> str:
        controls = ", ".join(self.control_markets[:3])
        if len(self.control_markets) > 3:
            controls += f", ... ({len(self.control_markets)} total)"
        return (
            f"MarketMatchResult\n"
            f"  Treatment: {self.treatment_market}\n"
            f"  Best control: {self.best_match} (distance: {self.distances[self.best_match]:.4f})\n"
            f"  Match quality: {self.match_quality:.2%}\n"
            f"  All controls: {controls}"
        )


def dtw_distance(
    series1: np.ndarray,
    series2: np.ndarray,
    window: Optional[int] = None,
) -> float:
    """
    Calculate Dynamic Time Warping distance between two time series.

    DTW finds the optimal alignment between two sequences by allowing
    "warping" (stretching/compressing) of the time axis.

    Parameters
    ----------
    series1 : array-like
        First time series
    series2 : array-like
        Second time series
    window : int, optional
        Maximum warping window. If None, no constraint.

    Returns
    -------
    float
        DTW distance (lower = more similar)

    Example
    -------
    >>> market_a = [100, 105, 110, 108, 115]
    >>> market_b = [102, 107, 112, 110, 118]
    >>> distance = dtw_distance(market_a, market_b)
    """
    s1 = np.asarray(series1)
    s2 = np.asarray(series2)
    n, m = len(s1), len(s2)

    # Initialize cost matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    # Apply window constraint if specified
    if window is None:
        window = max(n, m)

    # Fill cost matrix
    for i in range(1, n + 1):
        for j in range(max(1, i - window), min(m + 1, i + window + 1)):
            cost = (s1[i - 1] - s2[j - 1]) ** 2
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1]   # match
            )

    return np.sqrt(dtw_matrix[n, m])


def euclidean_distance(
    series1: np.ndarray,
    series2: np.ndarray,
) -> float:
    """
    Calculate Euclidean distance between two time series.

    Parameters
    ----------
    series1 : array-like
        First time series
    series2 : array-like
        Second time series (must be same length as series1)

    Returns
    -------
    float
        Euclidean distance
    """
    s1 = np.asarray(series1)
    s2 = np.asarray(series2)

    if len(s1) != len(s2):
        raise ValueError("Series must have the same length")

    return np.sqrt(np.sum((s1 - s2) ** 2))


def correlation_distance(
    series1: np.ndarray,
    series2: np.ndarray,
) -> float:
    """
    Calculate correlation-based distance between two time series.

    Distance = 1 - |correlation|, so perfectly correlated series
    have distance 0.

    Parameters
    ----------
    series1 : array-like
        First time series
    series2 : array-like
        Second time series

    Returns
    -------
    float
        Correlation distance (0-1, lower = more similar)
    """
    s1 = np.asarray(series1)
    s2 = np.asarray(series2)

    corr = np.corrcoef(s1, s2)[0, 1]
    return 1 - abs(corr)


def normalize_series(series: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize time series for comparison.

    Parameters
    ----------
    series : array-like
        Time series to normalize
    method : str
        'zscore' (mean=0, std=1) or 'minmax' (range 0-1)

    Returns
    -------
    np.ndarray
        Normalized series
    """
    series = np.asarray(series)

    if method == 'zscore':
        mean = np.mean(series)
        std = np.std(series)
        return (series - mean) / std if std > 0 else series - mean
    elif method == 'minmax':
        min_val = np.min(series)
        max_val = np.max(series)
        return (series - min_val) / (max_val - min_val) if max_val > min_val else series - min_val
    else:
        raise ValueError(f"Unknown normalization method: {method}")


class MarketMatcher:
    """
    Find optimal control markets for geo experiments.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data in long format with columns:
        - market: Market identifier
        - date/time: Time period
        - metric: The metric to match on
    market_col : str
        Column name for market identifier
    time_col : str
        Column name for time period
    metric_col : str
        Column name for the metric

    Example
    -------
    >>> df = pd.DataFrame({
    ...     'market': ['CA', 'CA', 'TX', 'TX', 'NY', 'NY'],
    ...     'week': [1, 2, 1, 2, 1, 2],
    ...     'conversions': [100, 105, 98, 103, 150, 155]
    ... })
    >>> matcher = MarketMatcher(df, 'market', 'week', 'conversions')
    >>> result = matcher.find_controls('CA', n_controls=2)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        market_col: str,
        time_col: str,
        metric_col: str,
    ):
        self.data = data.copy()
        self.market_col = market_col
        self.time_col = time_col
        self.metric_col = metric_col

        # Pivot to wide format for easy comparison
        self.wide_data = data.pivot(
            index=time_col,
            columns=market_col,
            values=metric_col
        )
        self.markets = list(self.wide_data.columns)

    def get_series(self, market: str) -> np.ndarray:
        """Get time series for a specific market."""
        return self.wide_data[market].values

    def calculate_distances(
        self,
        treatment_market: str,
        method: str = 'dtw',
        normalize: bool = True,
    ) -> Dict[str, float]:
        """
        Calculate distances from treatment market to all other markets.

        Parameters
        ----------
        treatment_market : str
            Market to find controls for
        method : str
            'dtw', 'euclidean', or 'correlation'
        normalize : bool
            Whether to normalize series before comparison

        Returns
        -------
        dict
            Market -> distance mapping
        """
        treatment_series = self.get_series(treatment_market)
        if normalize:
            treatment_series = normalize_series(treatment_series)

        distance_func = {
            'dtw': dtw_distance,
            'euclidean': euclidean_distance,
            'correlation': correlation_distance,
        }[method]

        distances = {}
        for market in self.markets:
            if market == treatment_market:
                continue

            control_series = self.get_series(market)
            if normalize:
                control_series = normalize_series(control_series)

            distances[market] = distance_func(treatment_series, control_series)

        return distances

    def find_controls(
        self,
        treatment_market: str,
        n_controls: int = 5,
        method: str = 'dtw',
        normalize: bool = True,
        exclude_markets: Optional[List[str]] = None,
    ) -> MarketMatchResult:
        """
        Find best control markets for a treatment market.

        Parameters
        ----------
        treatment_market : str
            Market where intervention will be applied
        n_controls : int
            Number of control markets to return
        method : str
            Distance method: 'dtw', 'euclidean', or 'correlation'
        normalize : bool
            Whether to normalize series before comparison
        exclude_markets : list, optional
            Markets to exclude from consideration

        Returns
        -------
        MarketMatchResult
            Matching results
        """
        distances = self.calculate_distances(treatment_market, method, normalize)

        # Exclude specified markets
        if exclude_markets:
            distances = {k: v for k, v in distances.items() if k not in exclude_markets}

        # Sort by distance
        sorted_markets = sorted(distances.items(), key=lambda x: x[1])
        control_markets = [m for m, _ in sorted_markets[:n_controls]]

        # Calculate weights (inverse distance)
        total_inv_dist = sum(1 / d for m, d in sorted_markets[:n_controls] if d > 0)
        weights = {}
        for market, dist in sorted_markets[:n_controls]:
            if dist > 0 and total_inv_dist > 0:
                weights[market] = (1 / dist) / total_inv_dist
            else:
                weights[market] = 1 / n_controls

        # Calculate correlation matrix
        selected_markets = [treatment_market] + control_markets
        corr_matrix = self.wide_data[selected_markets].corr()

        # Match quality: average correlation with best controls
        match_quality = corr_matrix.loc[treatment_market, control_markets].mean()

        return MarketMatchResult(
            treatment_market=treatment_market,
            control_markets=control_markets,
            distances={k: v for k, v in sorted_markets[:n_controls]},
            weights=weights,
            correlation_matrix=corr_matrix,
            best_match=control_markets[0],
            match_quality=match_quality,
        )

    def plot_comparison(
        self,
        treatment_market: str,
        control_markets: List[str],
        figsize: Tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        """
        Plot treatment market against control markets.

        Parameters
        ----------
        treatment_market : str
            Treatment market
        control_markets : list
            Control markets to plot
        figsize : tuple
            Figure size

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Normalize for comparison
        treatment_series = normalize_series(self.get_series(treatment_market))
        ax.plot(treatment_series, 'k-', linewidth=2, label=f'{treatment_market} (treatment)')

        colors = plt.cm.Set2(np.linspace(0, 1, len(control_markets)))
        for i, market in enumerate(control_markets):
            control_series = normalize_series(self.get_series(market))
            ax.plot(control_series, '--', color=colors[i], linewidth=1.5,
                    label=f'{market} (control)')

        ax.set_xlabel('Time Period')
        ax.set_ylabel('Normalized Metric')
        ax.set_title('Treatment vs Control Markets (Pre-Period)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


def find_best_control_markets(
    data: pd.DataFrame,
    treatment_market: str,
    market_col: str,
    time_col: str,
    metric_col: str,
    n_controls: int = 5,
    method: str = 'dtw',
    normalize: bool = True,
) -> MarketMatchResult:
    """
    Convenience function to find best control markets.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    treatment_market : str
        Market to find controls for
    market_col : str
        Column name for market identifier
    time_col : str
        Column name for time period
    metric_col : str
        Column name for the metric
    n_controls : int
        Number of control markets
    method : str
        Distance method
    normalize : bool
        Whether to normalize

    Returns
    -------
    MarketMatchResult
        Matching results

    Example
    -------
    >>> result = find_best_control_markets(
    ...     data=df,
    ...     treatment_market='California',
    ...     market_col='Region',
    ...     time_col='Week',
    ...     metric_col='Conversions',
    ...     n_controls=3
    ... )
    """
    matcher = MarketMatcher(data, market_col, time_col, metric_col)
    return matcher.find_controls(treatment_market, n_controls, method, normalize)


def validate_pre_period_balance(
    data: pd.DataFrame,
    treatment_markets: List[str],
    control_markets: List[str],
    market_col: str,
    time_col: str,
    metric_col: str,
) -> Tuple[float, float, bool]:
    """
    Validate that treatment and control groups are balanced in pre-period.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    treatment_markets : list
        Treatment market(s)
    control_markets : list
        Control market(s)
    market_col, time_col, metric_col : str
        Column names

    Returns
    -------
    tuple
        (t_statistic, p_value, is_balanced)
    """
    from scipy import stats

    treatment_data = data[data[market_col].isin(treatment_markets)][metric_col]
    control_data = data[data[market_col].isin(control_markets)][metric_col]

    t_stat, p_value = stats.ttest_ind(treatment_data, control_data)

    return t_stat, p_value, p_value > 0.05
