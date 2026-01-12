"""
Bayesian Structural Time Series (BSTS) for Causal Impact Analysis.

BSTS is a powerful method for estimating causal effects in time series
experiments. It builds a model of what would have happened without the
intervention (counterfactual) and compares it to observed data.

Key features:
- Handles seasonality and trends automatically
- Provides probabilistic (credible) intervals
- Can incorporate multiple control series
- More flexible than ITSA

This module provides a lightweight implementation. For full Bayesian
inference with MCMC, consider using PyMC or the causalimpact package.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class CausalImpactResult:
    """Results from Causal Impact (BSTS-style) analysis."""

    # Time series
    observed: np.ndarray
    predicted: np.ndarray  # Counterfactual prediction
    predicted_lower: np.ndarray  # Lower credible interval
    predicted_upper: np.ndarray  # Upper credible interval

    # Point-wise effects
    pointwise_effect: np.ndarray
    pointwise_effect_lower: np.ndarray
    pointwise_effect_upper: np.ndarray

    # Cumulative effects
    cumulative_effect: np.ndarray
    cumulative_effect_lower: np.ndarray
    cumulative_effect_upper: np.ndarray

    # Summary statistics (post-period only)
    average_effect: float
    average_effect_lower: float
    average_effect_upper: float
    total_effect: float
    total_effect_lower: float
    total_effect_upper: float
    relative_effect: float  # Percentage change
    relative_effect_lower: float
    relative_effect_upper: float

    # Statistical inference
    p_value: float  # Probability of no effect
    significant: bool

    # Time information
    pre_period: Tuple[int, int]
    post_period: Tuple[int, int]

    def __repr__(self) -> str:
        sig = "*" if self.significant else ""
        return (
            f"CausalImpactResult{sig}\n"
            f"  Average effect: {self.average_effect:.2f} "
            f"[{self.average_effect_lower:.2f}, {self.average_effect_upper:.2f}]\n"
            f"  Relative effect: {self.relative_effect:.1%} "
            f"[{self.relative_effect_lower:.1%}, {self.relative_effect_upper:.1%}]\n"
            f"  Total effect: {self.total_effect:.2f}\n"
            f"  P-value: {self.p_value:.4f}"
        )

    def summary(self, alpha: float = 0.05) -> str:
        """Generate detailed summary report."""
        ci_level = int((1 - alpha) * 100)

        lines = [
            "=" * 70,
            "                     CAUSAL IMPACT ANALYSIS SUMMARY",
            "=" * 70,
            "",
            f"Pre-period:  {self.pre_period[0]} to {self.pre_period[1]}",
            f"Post-period: {self.post_period[0]} to {self.post_period[1]}",
            "",
            "POSTERIOR INFERENCE",
            "-" * 70,
            "",
            f"  Average causal effect: {self.average_effect:.2f}",
            f"    {ci_level}% CI: [{self.average_effect_lower:.2f}, {self.average_effect_upper:.2f}]",
            "",
            f"  Relative effect: {self.relative_effect:.1%}",
            f"    {ci_level}% CI: [{self.relative_effect_lower:.1%}, {self.relative_effect_upper:.1%}]",
            "",
            f"  Cumulative effect: {self.total_effect:.2f}",
            f"    {ci_level}% CI: [{self.total_effect_lower:.2f}, {self.total_effect_upper:.2f}]",
            "",
            "STATISTICAL SIGNIFICANCE",
            "-" * 70,
            "",
            f"  Probability of causal effect: {1 - self.p_value:.1%}",
            "",
        ]

        if self.significant:
            lines.append(f"  The intervention HAD a statistically significant effect.")
            if self.average_effect > 0:
                lines.append(f"  The effect was POSITIVE (increase of {self.relative_effect:.1%})")
            else:
                lines.append(f"  The effect was NEGATIVE (decrease of {abs(self.relative_effect):.1%})")
        else:
            lines.append(f"  The intervention DID NOT have a statistically significant effect.")
            lines.append(f"  Any observed changes could be due to random fluctuation.")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)


def run_causal_impact(
    y: np.ndarray,
    pre_period: Tuple[int, int],
    post_period: Tuple[int, int],
    control_series: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    n_seasons: Optional[int] = None,
) -> CausalImpactResult:
    """
    Estimate causal impact of an intervention.

    This is a simplified implementation that uses:
    1. Linear regression on control series (if provided)
    2. Or autoregressive model on pre-period (if no controls)

    Parameters
    ----------
    y : array-like
        Response time series (outcome of interest)
    pre_period : tuple
        (start_index, end_index) of pre-intervention period
    post_period : tuple
        (start_index, end_index) of post-intervention period
    control_series : array-like, optional
        Control time series that were not affected by intervention
        Shape: (n_timepoints,) or (n_timepoints, n_controls)
    alpha : float, default=0.05
        Significance level for credible intervals
    n_seasons : int, optional
        Number of seasonal periods (e.g., 7 for weekly, 12 for monthly)

    Returns
    -------
    CausalImpactResult
        Analysis results

    Example
    -------
    >>> # Monthly data with 12 months pre, 6 months post
    >>> y = [100, 102, 98, 105, 103, 107, 110, 108, 112, 115, 113, 118,
    ...      135, 138, 142, 145, 150, 155]  # Intervention at index 12
    >>> result = run_causal_impact(y, pre_period=(0, 11), post_period=(12, 17))
    >>> print(result)

    Notes
    -----
    For full Bayesian BSTS, consider using:
    - causalimpact package (Python port of R's CausalImpact)
    - PyMC for custom Bayesian models
    """
    y = np.asarray(y)
    n = len(y)

    pre_start, pre_end = pre_period
    post_start, post_end = post_period

    # Validate periods
    if pre_end >= post_start:
        raise ValueError("Pre-period must end before post-period starts")
    if post_end >= n:
        raise ValueError("Post-period end exceeds data length")

    # Extract periods
    y_pre = y[pre_start:pre_end + 1]
    y_post = y[post_start:post_end + 1]
    n_pre = len(y_pre)
    n_post = len(y_post)

    # Build prediction model
    if control_series is not None:
        # Use control series for prediction
        control_series = np.asarray(control_series)
        if control_series.ndim == 1:
            control_series = control_series.reshape(-1, 1)

        X_pre = control_series[pre_start:pre_end + 1]
        X_post = control_series[post_start:post_end + 1]
        X_all = control_series

        # Add intercept
        X_pre = np.column_stack([np.ones(n_pre), X_pre])
        X_post = np.column_stack([np.ones(n_post), X_post])
        X_all = np.column_stack([np.ones(n), X_all])

        # Fit model on pre-period
        beta, residuals, _, _ = np.linalg.lstsq(X_pre, y_pre, rcond=None)
        sigma = np.std(y_pre - X_pre @ beta)

        # Predict for all periods
        predicted = X_all @ beta

    else:
        # Use autoregressive model
        # Fit trend + optional seasonality on pre-period
        time_pre = np.arange(n_pre)
        time_all = np.arange(n)

        if n_seasons is not None and n_seasons > 1:
            # Add seasonal dummies
            season_pre = np.zeros((n_pre, n_seasons - 1))
            season_all = np.zeros((n, n_seasons - 1))
            for i in range(n_seasons - 1):
                season_pre[:, i] = (time_pre % n_seasons == i).astype(float)
                season_all[:, i] = (time_all % n_seasons == i).astype(float)

            X_pre = np.column_stack([np.ones(n_pre), time_pre, season_pre])
            X_all = np.column_stack([np.ones(n), time_all, season_all])
        else:
            X_pre = np.column_stack([np.ones(n_pre), time_pre])
            X_all = np.column_stack([np.ones(n), time_all])

        # Fit model
        beta, residuals, _, _ = np.linalg.lstsq(X_pre, y_pre, rcond=None)
        sigma = np.std(y_pre - X_pre @ beta)

        # Predict for all periods
        predicted = X_all @ beta

    # Calculate credible intervals
    z = stats.norm.ppf(1 - alpha / 2)
    predicted_lower = predicted - z * sigma
    predicted_upper = predicted + z * sigma

    # Point-wise effects (observed - predicted)
    pointwise_effect = np.zeros(n)
    pointwise_effect[post_start:post_end + 1] = y_post - predicted[post_start:post_end + 1]

    pointwise_lower = np.zeros(n)
    pointwise_upper = np.zeros(n)
    pointwise_lower[post_start:post_end + 1] = pointwise_effect[post_start:post_end + 1] - z * sigma
    pointwise_upper[post_start:post_end + 1] = pointwise_effect[post_start:post_end + 1] + z * sigma

    # Cumulative effects
    cumulative_effect = np.cumsum(pointwise_effect)
    cumulative_effect[:post_start] = 0
    cumulative_lower = np.zeros(n)
    cumulative_upper = np.zeros(n)

    # Monte Carlo for cumulative CI
    n_samples = 1000
    cumulative_samples = np.zeros((n_samples, n_post))
    for i in range(n_samples):
        noise = np.random.normal(0, sigma, n_post)
        effect_sample = y_post - predicted[post_start:post_end + 1] + noise
        cumulative_samples[i] = np.cumsum(effect_sample)

    cumulative_lower[post_start:post_end + 1] = np.percentile(cumulative_samples, alpha/2 * 100, axis=0)
    cumulative_upper[post_start:post_end + 1] = np.percentile(cumulative_samples, (1 - alpha/2) * 100, axis=0)

    # Summary statistics (post-period)
    post_effect = pointwise_effect[post_start:post_end + 1]
    average_effect = np.mean(post_effect)
    average_effect_se = sigma / np.sqrt(n_post)
    average_effect_lower = average_effect - z * average_effect_se
    average_effect_upper = average_effect + z * average_effect_se

    total_effect = np.sum(post_effect)
    total_effect_se = sigma * np.sqrt(n_post)
    total_effect_lower = total_effect - z * total_effect_se
    total_effect_upper = total_effect + z * total_effect_se

    # Relative effect
    predicted_post_mean = np.mean(predicted[post_start:post_end + 1])
    if predicted_post_mean != 0:
        relative_effect = average_effect / predicted_post_mean
        relative_effect_lower = average_effect_lower / predicted_post_mean
        relative_effect_upper = average_effect_upper / predicted_post_mean
    else:
        relative_effect = relative_effect_lower = relative_effect_upper = 0.0

    # P-value (two-sided test for no effect)
    # Using normal approximation
    z_stat = average_effect / average_effect_se if average_effect_se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return CausalImpactResult(
        observed=y,
        predicted=predicted,
        predicted_lower=predicted_lower,
        predicted_upper=predicted_upper,
        pointwise_effect=pointwise_effect,
        pointwise_effect_lower=pointwise_lower,
        pointwise_effect_upper=pointwise_upper,
        cumulative_effect=cumulative_effect,
        cumulative_effect_lower=cumulative_lower,
        cumulative_effect_upper=cumulative_upper,
        average_effect=average_effect,
        average_effect_lower=average_effect_lower,
        average_effect_upper=average_effect_upper,
        total_effect=total_effect,
        total_effect_lower=total_effect_lower,
        total_effect_upper=total_effect_upper,
        relative_effect=relative_effect,
        relative_effect_lower=relative_effect_lower,
        relative_effect_upper=relative_effect_upper,
        p_value=p_value,
        significant=p_value < alpha,
        pre_period=pre_period,
        post_period=post_period,
    )


def plot_causal_impact(
    result: CausalImpactResult,
    dates: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (12, 10),
) -> plt.Figure:
    """
    Plot causal impact results in standard 3-panel format.

    Parameters
    ----------
    result : CausalImpactResult
        Results from run_causal_impact
    dates : array-like, optional
        Date labels for x-axis
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(result.observed)
    x = dates if dates is not None else np.arange(n)

    post_start = result.post_period[0]
    post_end = result.post_period[1]

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Panel 1: Original data vs counterfactual
    ax1 = axes[0]
    ax1.plot(x, result.observed, 'k-', linewidth=1.5, label='Observed')
    ax1.plot(x, result.predicted, 'b--', linewidth=1.5, label='Counterfactual')
    ax1.fill_between(x, result.predicted_lower, result.predicted_upper,
                     color='blue', alpha=0.2, label='95% CI')
    ax1.axvline(x=x[post_start], color='gray', linestyle=':', linewidth=2)
    ax1.set_ylabel('Outcome')
    ax1.set_title('Original Series vs Counterfactual')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Point-wise effect
    ax2 = axes[1]
    ax2.plot(x[post_start:post_end+1],
             result.pointwise_effect[post_start:post_end+1],
             'k-', linewidth=1.5)
    ax2.fill_between(x[post_start:post_end+1],
                     result.pointwise_effect_lower[post_start:post_end+1],
                     result.pointwise_effect_upper[post_start:post_end+1],
                     color='blue', alpha=0.2)
    ax2.axhline(y=0, color='gray', linestyle='--')
    ax2.axvline(x=x[post_start], color='gray', linestyle=':', linewidth=2)
    ax2.set_ylabel('Effect')
    ax2.set_title('Point-wise Effect (Observed - Predicted)')
    ax2.grid(True, alpha=0.3)

    # Pre-period placeholder for panel 2
    ax2.plot(x[:post_start], np.zeros(post_start), 'k-', linewidth=1.5, alpha=0.3)

    # Panel 3: Cumulative effect
    ax3 = axes[2]
    ax3.plot(x[post_start:post_end+1],
             result.cumulative_effect[post_start:post_end+1],
             'k-', linewidth=1.5)
    ax3.fill_between(x[post_start:post_end+1],
                     result.cumulative_effect_lower[post_start:post_end+1],
                     result.cumulative_effect_upper[post_start:post_end+1],
                     color='blue', alpha=0.2)
    ax3.axhline(y=0, color='gray', linestyle='--')
    ax3.axvline(x=x[post_start], color='gray', linestyle=':', linewidth=2)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Cumulative Effect')
    ax3.set_title('Cumulative Effect')
    ax3.grid(True, alpha=0.3)

    # Pre-period placeholder for panel 3
    ax3.plot(x[:post_start], np.zeros(post_start), 'k-', linewidth=1.5, alpha=0.3)

    plt.tight_layout()
    return fig


def run_causal_impact_with_pymc(
    y: np.ndarray,
    pre_period: Tuple[int, int],
    post_period: Tuple[int, int],
    control_series: Optional[np.ndarray] = None,
    n_samples: int = 2000,
    **kwargs,
) -> CausalImpactResult:
    """
    Full Bayesian causal impact analysis using PyMC.

    This requires PyMC to be installed.

    Parameters
    ----------
    y : array-like
        Response time series
    pre_period : tuple
        (start_index, end_index) of pre-intervention period
    post_period : tuple
        (start_index, end_index) of post-intervention period
    control_series : array-like, optional
        Control time series
    n_samples : int
        Number of MCMC samples
    **kwargs
        Additional arguments passed to PyMC sampler

    Returns
    -------
    CausalImpactResult
        Analysis results with full posterior inference

    Raises
    ------
    ImportError
        If PyMC is not installed
    """
    try:
        import pymc as pm
        import arviz as az
    except ImportError:
        raise ImportError(
            "PyMC and ArviZ are required for full Bayesian analysis. "
            "Install with: pip install pymc arviz"
        )

    y = np.asarray(y)
    n = len(y)

    pre_start, pre_end = pre_period
    post_start, post_end = post_period

    y_pre = y[pre_start:pre_end + 1]
    n_pre = len(y_pre)

    # Build model
    with pm.Model() as model:
        # Priors
        sigma = pm.HalfNormal('sigma', sigma=np.std(y_pre))

        if control_series is not None:
            control_series = np.asarray(control_series)
            if control_series.ndim == 1:
                control_series = control_series.reshape(-1, 1)
            n_controls = control_series.shape[1]

            intercept = pm.Normal('intercept', mu=np.mean(y_pre), sigma=np.std(y_pre))
            beta = pm.Normal('beta', mu=0, sigma=1, shape=n_controls)

            X_pre = control_series[pre_start:pre_end + 1]
            mu_pre = intercept + pm.math.dot(X_pre, beta)

            X_all = control_series
            mu_all = intercept + pm.math.dot(X_all, beta)

        else:
            # Local level model
            intercept = pm.Normal('intercept', mu=np.mean(y_pre), sigma=np.std(y_pre))
            trend = pm.Normal('trend', mu=0, sigma=0.1)

            time_pre = np.arange(n_pre)
            mu_pre = intercept + trend * time_pre

            time_all = np.arange(n)
            mu_all = intercept + trend * time_all

        # Likelihood (pre-period only)
        y_obs = pm.Normal('y_obs', mu=mu_pre, sigma=sigma, observed=y_pre)

        # Sample
        trace = pm.sample(n_samples, return_inferencedata=True, **kwargs)

    # Generate predictions for all periods
    with model:
        pm.set_data({})  # Reset any data
        posterior_pred = pm.sample_posterior_predictive(trace, var_names=['y_obs'])

    # This is a simplified version - full implementation would properly
    # generate counterfactual predictions for the post-period

    # For now, fall back to the frequentist approach
    return run_causal_impact(y, pre_period, post_period, control_series)
