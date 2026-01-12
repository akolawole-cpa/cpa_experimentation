"""
Interrupted Time Series Analysis (ITSA).

ITSA is used to estimate the causal effect of an intervention when
randomization is not possible. It models the time series before and
after an intervention to estimate:
1. Level change: Immediate jump in the outcome
2. Slope change: Change in the trend over time

This is particularly useful for:
- Policy interventions
- Marketing campaign launches
- Product feature rollouts
- Geographic market tests
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Tuple, List
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class ITSAResult:
    """Results from Interrupted Time Series Analysis."""

    # Model coefficients
    intercept: float
    trend: float  # Pre-intervention slope
    level_change: float  # Immediate effect at intervention
    slope_change: float  # Change in slope post-intervention

    # Standard errors
    se_intercept: float
    se_trend: float
    se_level_change: float
    se_slope_change: float

    # Statistical tests
    t_level: float
    t_slope: float
    p_level: float
    p_slope: float

    # Model fit
    r_squared: float
    residual_std: float

    # Cumulative effect
    cumulative_effect: float
    counterfactual_values: np.ndarray
    actual_values: np.ndarray
    fitted_values: np.ndarray

    # Time information
    intervention_index: int
    n_pre: int
    n_post: int

    def __repr__(self) -> str:
        sig_level = "*" if self.p_level < 0.05 else ""
        sig_slope = "*" if self.p_slope < 0.05 else ""
        return (
            f"ITSAResult\n"
            f"  Pre-intervention trend: {self.trend:.4f} (SE: {self.se_trend:.4f})\n"
            f"  Level change: {self.level_change:.4f}{sig_level} (p={self.p_level:.4f})\n"
            f"  Slope change: {self.slope_change:.4f}{sig_slope} (p={self.p_slope:.4f})\n"
            f"  Cumulative effect: {self.cumulative_effect:.2f}\n"
            f"  R-squared: {self.r_squared:.4f}"
        )

    def summary(self) -> str:
        """Generate detailed summary."""
        lines = [
            "=" * 60,
            "           Interrupted Time Series Analysis Results",
            "=" * 60,
            "",
            f"  Sample: {self.n_pre} pre-intervention, {self.n_post} post-intervention",
            "",
            "  Coefficients:",
            f"    Intercept:     {self.intercept:>10.4f} (SE: {self.se_intercept:.4f})",
            f"    Time trend:    {self.trend:>10.4f} (SE: {self.se_trend:.4f})",
            f"    Level change:  {self.level_change:>10.4f} (SE: {self.se_level_change:.4f}, p={self.p_level:.4f})",
            f"    Slope change:  {self.slope_change:>10.4f} (SE: {self.se_slope_change:.4f}, p={self.p_slope:.4f})",
            "",
            f"  Model fit: R² = {self.r_squared:.4f}",
            f"  Residual std: {self.residual_std:.4f}",
            "",
            f"  Cumulative effect (total lift): {self.cumulative_effect:.2f}",
            "=" * 60,
        ]
        return "\n".join(lines)


class ITSAnalysis:
    """
    Interrupted Time Series Analysis class.

    Estimates causal effects using a segmented regression approach:

    Y_t = β0 + β1*time + β2*intervention + β3*time_after_intervention + ε

    Where:
    - β0: Baseline level
    - β1: Pre-intervention trend (slope)
    - β2: Level change at intervention (immediate effect)
    - β3: Slope change after intervention (gradual effect)

    Parameters
    ----------
    y : array-like
        Time series of outcomes
    intervention_index : int
        Index where intervention begins (0-indexed)
    dates : array-like, optional
        Date labels for plotting

    Example
    -------
    >>> # Monthly sales data with intervention at month 12
    >>> sales = [100, 102, 105, 103, 108, 110, 112, 115, 113, 118, 120, 122,
    ...          135, 138, 142, 145, 150, 155, 160, 165, 170, 175, 180, 185]
    >>> itsa = ITSAnalysis(sales, intervention_index=12)
    >>> result = itsa.fit()
    >>> print(result)
    """

    def __init__(
        self,
        y: np.ndarray,
        intervention_index: int,
        dates: Optional[np.ndarray] = None,
    ):
        self.y = np.asarray(y)
        self.intervention_index = intervention_index
        self.dates = dates
        self.n = len(y)
        self.n_pre = intervention_index
        self.n_post = self.n - intervention_index

        if intervention_index <= 1 or intervention_index >= self.n - 1:
            raise ValueError("Intervention index must leave at least 2 points on each side")

        self._result: Optional[ITSAResult] = None

    def _build_design_matrix(self) -> np.ndarray:
        """Build the design matrix for ITSA regression."""
        n = self.n
        idx = self.intervention_index

        # Time variable (centered at intervention for interpretability)
        time = np.arange(n) - idx

        # Intervention indicator (0 before, 1 after)
        intervention = np.zeros(n)
        intervention[idx:] = 1

        # Time after intervention (0 before, counts from 0 after)
        time_after = np.zeros(n)
        time_after[idx:] = np.arange(self.n_post)

        # Design matrix: [intercept, time, intervention, time_after]
        X = np.column_stack([
            np.ones(n),  # Intercept
            time,  # Overall time trend
            intervention,  # Level change
            time_after,  # Slope change
        ])

        return X

    def fit(self) -> ITSAResult:
        """
        Fit the ITSA model using OLS regression.

        Returns
        -------
        ITSAResult
            Model results
        """
        X = self._build_design_matrix()
        y = self.y

        # OLS: β = (X'X)^(-1) X'y
        XtX = X.T @ X
        Xty = X.T @ y
        beta = np.linalg.solve(XtX, Xty)

        # Fitted values and residuals
        fitted = X @ beta
        residuals = y - fitted

        # Residual variance and standard errors
        n, p = X.shape
        df = n - p
        mse = np.sum(residuals ** 2) / df
        var_beta = mse * np.linalg.inv(XtX)
        se = np.sqrt(np.diag(var_beta))

        # T-statistics and p-values
        t_stats = beta / se
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df))

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # Counterfactual: what would have happened without intervention
        X_cf = X.copy()
        X_cf[:, 2] = 0  # No level change
        X_cf[:, 3] = 0  # No slope change
        counterfactual = X_cf @ beta

        # Cumulative effect
        cumulative_effect = np.sum(y[self.intervention_index:] - counterfactual[self.intervention_index:])

        self._result = ITSAResult(
            intercept=beta[0],
            trend=beta[1],
            level_change=beta[2],
            slope_change=beta[3],
            se_intercept=se[0],
            se_trend=se[1],
            se_level_change=se[2],
            se_slope_change=se[3],
            t_level=t_stats[2],
            t_slope=t_stats[3],
            p_level=p_values[2],
            p_slope=p_values[3],
            r_squared=r_squared,
            residual_std=np.sqrt(mse),
            cumulative_effect=cumulative_effect,
            counterfactual_values=counterfactual,
            actual_values=y,
            fitted_values=fitted,
            intervention_index=self.intervention_index,
            n_pre=self.n_pre,
            n_post=self.n_post,
        )

        return self._result

    @property
    def result(self) -> Optional[ITSAResult]:
        """Get the fitted result."""
        return self._result

    def predict(self, n_future: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions for fitted period plus future periods.

        Parameters
        ----------
        n_future : int
            Number of future periods to predict

        Returns
        -------
        tuple
            (predicted_values, counterfactual_values)
        """
        if self._result is None:
            raise ValueError("Model must be fit first")

        if n_future == 0:
            return self._result.fitted_values, self._result.counterfactual_values

        # Extend for future periods
        n_total = self.n + n_future
        idx = self.intervention_index

        time = np.arange(n_total) - idx
        intervention = np.zeros(n_total)
        intervention[idx:] = 1
        time_after = np.zeros(n_total)
        time_after[idx:] = np.arange(n_total - idx)

        X = np.column_stack([np.ones(n_total), time, intervention, time_after])
        beta = np.array([
            self._result.intercept,
            self._result.trend,
            self._result.level_change,
            self._result.slope_change,
        ])

        predicted = X @ beta

        X_cf = X.copy()
        X_cf[:, 2] = 0
        X_cf[:, 3] = 0
        counterfactual = X_cf @ beta

        return predicted, counterfactual

    def plot(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot ITSA results.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self._result is None:
            raise ValueError("Model must be fit first")

        return plot_itsa(self._result, self.dates, figsize)


def run_itsa(
    y: np.ndarray,
    intervention_index: int,
    dates: Optional[np.ndarray] = None,
) -> ITSAResult:
    """
    Convenience function to run ITSA.

    Parameters
    ----------
    y : array-like
        Time series of outcomes
    intervention_index : int
        Index where intervention begins
    dates : array-like, optional
        Date labels

    Returns
    -------
    ITSAResult
        Analysis results
    """
    analysis = ITSAnalysis(y, intervention_index, dates)
    return analysis.fit()


def plot_itsa(
    result: ITSAResult,
    dates: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot ITSA results with actual, fitted, and counterfactual values.

    Parameters
    ----------
    result : ITSAResult
        Results from ITSA
    dates : array-like, optional
        Date labels for x-axis
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    n = len(result.actual_values)
    x = dates if dates is not None else np.arange(n)

    # Actual values
    ax.plot(x, result.actual_values, 'o-', color='#2c3e50', label='Actual', alpha=0.7)

    # Fitted values
    ax.plot(x, result.fitted_values, '-', color='#3498db', linewidth=2, label='Fitted')

    # Counterfactual
    ax.plot(x, result.counterfactual_values, '--', color='#e74c3c', linewidth=2,
            label='Counterfactual (no intervention)')

    # Intervention line
    if dates is not None:
        intervention_x = dates[result.intervention_index]
    else:
        intervention_x = result.intervention_index

    ax.axvline(x=intervention_x, color='gray', linestyle=':', linewidth=2,
               label='Intervention')

    # Shade the intervention period
    ax.axvspan(intervention_x, x[-1] if dates is not None else n-1,
               alpha=0.1, color='green')

    # Add effect annotation
    ax.annotate(
        f'Cumulative effect: {result.cumulative_effect:.1f}',
        xy=(0.98, 0.95), xycoords='axes fraction',
        ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    ax.set_xlabel('Time')
    ax.set_ylabel('Outcome')
    ax.set_title('Interrupted Time Series Analysis')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def itsa_with_covariates(
    y: np.ndarray,
    intervention_index: int,
    covariates: np.ndarray,
) -> ITSAResult:
    """
    ITSA with additional covariates for improved control.

    Parameters
    ----------
    y : array-like
        Time series of outcomes
    intervention_index : int
        Index where intervention begins
    covariates : 2D array
        Additional covariates (n_timepoints x n_covariates)

    Returns
    -------
    ITSAResult
        Analysis results (coefficients only for ITSA terms)

    Notes
    -----
    Including covariates can help control for confounders and
    reduce residual variance, leading to more precise estimates.
    """
    y = np.asarray(y)
    covariates = np.asarray(covariates)
    n = len(y)
    idx = intervention_index

    # Build ITSA design matrix
    time = np.arange(n) - idx
    intervention = np.zeros(n)
    intervention[idx:] = 1
    time_after = np.zeros(n)
    time_after[idx:] = np.arange(n - idx)

    # Combine with covariates
    X_itsa = np.column_stack([np.ones(n), time, intervention, time_after])
    X = np.column_stack([X_itsa, covariates])

    # OLS
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)

    # Extract ITSA coefficients
    beta_itsa = beta[:4]

    # Calculate statistics
    fitted = X @ beta
    residuals = y - fitted
    n_total, p = X.shape
    df = n_total - p
    mse = np.sum(residuals ** 2) / df
    var_beta = mse * np.linalg.inv(XtX)
    se = np.sqrt(np.diag(var_beta))[:4]

    t_stats = beta_itsa / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df))

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Counterfactual
    X_cf = X.copy()
    X_cf[:, 2] = 0
    X_cf[:, 3] = 0
    counterfactual = X_cf @ beta

    cumulative_effect = np.sum(y[idx:] - counterfactual[idx:])

    return ITSAResult(
        intercept=beta_itsa[0],
        trend=beta_itsa[1],
        level_change=beta_itsa[2],
        slope_change=beta_itsa[3],
        se_intercept=se[0],
        se_trend=se[1],
        se_level_change=se[2],
        se_slope_change=se[3],
        t_level=t_stats[2],
        t_slope=t_stats[3],
        p_level=p_values[2],
        p_slope=p_values[3],
        r_squared=r_squared,
        residual_std=np.sqrt(mse),
        cumulative_effect=cumulative_effect,
        counterfactual_values=counterfactual,
        actual_values=y,
        fitted_values=fitted,
        intervention_index=idx,
        n_pre=idx,
        n_post=n - idx,
    )
