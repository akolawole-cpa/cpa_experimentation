"""
Geo Lift Analysis for Geographic Experiments.

This module provides methods for estimating causal effects in
geographic experiments:

1. Difference-in-Differences (DiD): Compare treatment vs control
   before and after intervention
2. Synthetic Control: Create a weighted combination of control
   markets to match the treatment market

These methods are used when:
- You can't randomize at the user level
- You're testing marketing campaigns in specific markets
- You need to validate MMM incrementality estimates
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class GeoLiftResult:
    """Results from geo lift analysis."""

    # Effect estimates
    lift: float  # Absolute lift
    lift_se: float  # Standard error
    lift_ci_lower: float
    lift_ci_upper: float
    relative_lift: float  # Percentage lift
    relative_lift_ci_lower: float
    relative_lift_ci_upper: float

    # Statistical inference
    t_statistic: float
    p_value: float
    significant: bool

    # Counterfactual
    counterfactual_total: float  # What would have happened
    actual_total: float  # What actually happened
    incremental_total: float  # Difference (actual - counterfactual)

    # Method used
    method: str
    treatment_markets: List[str]
    control_markets: List[str]

    def __repr__(self) -> str:
        sig = "*" if self.significant else ""
        return (
            f"GeoLiftResult({self.method}){sig}\n"
            f"  Lift: {self.lift:.2f} (SE: {self.lift_se:.2f})\n"
            f"  Relative lift: {self.relative_lift:.1%} "
            f"[{self.relative_lift_ci_lower:.1%}, {self.relative_lift_ci_upper:.1%}]\n"
            f"  Incremental: {self.incremental_total:.2f}\n"
            f"  p-value: {self.p_value:.4f}"
        )

    def summary(self) -> str:
        """Generate detailed summary."""
        lines = [
            "=" * 60,
            "                  GEO LIFT ANALYSIS RESULTS",
            "=" * 60,
            "",
            f"Method: {self.method}",
            f"Treatment markets: {', '.join(self.treatment_markets)}",
            f"Control markets: {', '.join(self.control_markets[:5])}{'...' if len(self.control_markets) > 5 else ''}",
            "",
            "EFFECT ESTIMATES",
            "-" * 60,
            f"  Average lift per period: {self.lift:.2f} (SE: {self.lift_se:.2f})",
            f"  95% CI: [{self.lift_ci_lower:.2f}, {self.lift_ci_upper:.2f}]",
            "",
            f"  Relative lift: {self.relative_lift:.1%}",
            f"  95% CI: [{self.relative_lift_ci_lower:.1%}, {self.relative_lift_ci_upper:.1%}]",
            "",
            "TOTALS",
            "-" * 60,
            f"  Actual total: {self.actual_total:,.2f}",
            f"  Counterfactual total: {self.counterfactual_total:,.2f}",
            f"  Incremental impact: {self.incremental_total:,.2f}",
            "",
            "STATISTICAL SIGNIFICANCE",
            "-" * 60,
            f"  t-statistic: {self.t_statistic:.3f}",
            f"  p-value: {self.p_value:.4f}",
            "",
        ]

        if self.significant:
            lines.append("  CONCLUSION: The intervention HAD a statistically significant effect.")
        else:
            lines.append("  CONCLUSION: No statistically significant effect detected.")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


def difference_in_differences(
    data: pd.DataFrame,
    treatment_markets: List[str],
    control_markets: List[str],
    market_col: str,
    time_col: str,
    metric_col: str,
    intervention_time: Union[str, int, pd.Timestamp],
    alpha: float = 0.05,
) -> GeoLiftResult:
    """
    Estimate causal effect using Difference-in-Differences.

    DiD compares the change in outcomes over time between treatment
    and control groups:

    Effect = (Treatment_post - Treatment_pre) - (Control_post - Control_pre)

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    treatment_markets : list
        Markets where intervention was applied
    control_markets : list
        Markets without intervention
    market_col : str
        Column for market identifier
    time_col : str
        Column for time period
    metric_col : str
        Column for the outcome metric
    intervention_time : str, int, or Timestamp
        Time when intervention started
    alpha : float
        Significance level

    Returns
    -------
    GeoLiftResult
        Analysis results

    Example
    -------
    >>> result = difference_in_differences(
    ...     data=df,
    ...     treatment_markets=['California'],
    ...     control_markets=['Texas', 'Florida'],
    ...     market_col='Region',
    ...     time_col='Week',
    ...     metric_col='Conversions',
    ...     intervention_time='2025-03-01'
    ... )

    Notes
    -----
    Key assumption: Parallel trends - in absence of treatment,
    treatment and control groups would have followed parallel paths.
    """
    data = data.copy()

    # Create treatment indicator
    data['is_treatment'] = data[market_col].isin(treatment_markets).astype(int)

    # Create post-period indicator
    data['is_post'] = (data[time_col] >= intervention_time).astype(int)

    # Create interaction term
    data['treatment_x_post'] = data['is_treatment'] * data['is_post']

    # Aggregate by market and period
    pre_treatment = data[(data['is_treatment'] == 1) & (data['is_post'] == 0)][metric_col]
    post_treatment = data[(data['is_treatment'] == 1) & (data['is_post'] == 1)][metric_col]
    pre_control = data[(data['is_treatment'] == 0) & (data['is_post'] == 0)][metric_col]
    post_control = data[(data['is_treatment'] == 0) & (data['is_post'] == 1)][metric_col]

    # DiD estimate
    treatment_diff = post_treatment.mean() - pre_treatment.mean()
    control_diff = post_control.mean() - pre_control.mean()
    did_estimate = treatment_diff - control_diff

    # Standard error (using regression approach)
    # Y = β0 + β1*treatment + β2*post + β3*(treatment*post) + ε
    from scipy import stats as scipy_stats

    n = len(data)
    X = np.column_stack([
        np.ones(n),
        data['is_treatment'].values,
        data['is_post'].values,
        data['treatment_x_post'].values
    ])
    y = data[metric_col].values

    # OLS
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ beta
    mse = np.sum(residuals ** 2) / (n - 4)
    var_beta = mse * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(var_beta))

    # DiD coefficient is β3
    did_se = se[3]

    # Confidence interval
    z = scipy_stats.norm.ppf(1 - alpha / 2)
    ci_lower = did_estimate - z * did_se
    ci_upper = did_estimate + z * did_se

    # T-test
    t_stat = did_estimate / did_se if did_se > 0 else 0
    p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), n - 4))

    # Counterfactual and totals
    n_post_periods = len(post_treatment)
    actual_total = post_treatment.sum()
    counterfactual_mean = pre_treatment.mean() + control_diff
    counterfactual_total = counterfactual_mean * n_post_periods
    incremental_total = actual_total - counterfactual_total

    # Relative lift
    if counterfactual_mean != 0:
        relative_lift = did_estimate / counterfactual_mean
        relative_ci_lower = ci_lower / counterfactual_mean
        relative_ci_upper = ci_upper / counterfactual_mean
    else:
        relative_lift = relative_ci_lower = relative_ci_upper = 0.0

    return GeoLiftResult(
        lift=did_estimate,
        lift_se=did_se,
        lift_ci_lower=ci_lower,
        lift_ci_upper=ci_upper,
        relative_lift=relative_lift,
        relative_lift_ci_lower=relative_ci_lower,
        relative_lift_ci_upper=relative_ci_upper,
        t_statistic=t_stat,
        p_value=p_value,
        significant=p_value < alpha,
        counterfactual_total=counterfactual_total,
        actual_total=actual_total,
        incremental_total=incremental_total,
        method='Difference-in-Differences',
        treatment_markets=treatment_markets,
        control_markets=control_markets,
    )


def synthetic_control(
    data: pd.DataFrame,
    treatment_market: str,
    control_markets: List[str],
    market_col: str,
    time_col: str,
    metric_col: str,
    intervention_time: Union[str, int, pd.Timestamp],
    alpha: float = 0.05,
) -> GeoLiftResult:
    """
    Estimate causal effect using Synthetic Control Method.

    Creates a weighted combination of control markets that best
    matches the treatment market in the pre-period, then uses
    this synthetic control to estimate what would have happened.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    treatment_market : str
        Market where intervention was applied (single market)
    control_markets : list
        Candidate control markets
    market_col : str
        Column for market identifier
    time_col : str
        Column for time period
    metric_col : str
        Column for the outcome metric
    intervention_time : str, int, or Timestamp
        Time when intervention started
    alpha : float
        Significance level

    Returns
    -------
    GeoLiftResult
        Analysis results

    Notes
    -----
    The synthetic control is created by finding weights w such that:
    minimize ||Y_treatment_pre - sum(w_i * Y_control_i_pre)||^2
    subject to: w_i >= 0, sum(w_i) = 1
    """
    from scipy.optimize import minimize

    # Pivot to wide format
    wide = data.pivot(index=time_col, columns=market_col, values=metric_col)

    # Split into pre and post periods
    pre_mask = wide.index < intervention_time
    post_mask = wide.index >= intervention_time

    # Treatment series
    y_treatment_pre = wide.loc[pre_mask, treatment_market].values
    y_treatment_post = wide.loc[post_mask, treatment_market].values

    # Control matrix
    X_control_pre = wide.loc[pre_mask, control_markets].values
    X_control_post = wide.loc[post_mask, control_markets].values

    n_controls = len(control_markets)

    # Optimize weights
    def objective(w):
        synthetic_pre = X_control_pre @ w
        return np.sum((y_treatment_pre - synthetic_pre) ** 2)

    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    # Bounds: weights are non-negative
    bounds = [(0, 1) for _ in range(n_controls)]
    # Initial guess: equal weights
    w0 = np.ones(n_controls) / n_controls

    result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    weights = result.x

    # Create synthetic control
    synthetic_pre = X_control_pre @ weights
    synthetic_post = X_control_post @ weights

    # Calculate effect
    effects = y_treatment_post - synthetic_post
    lift = np.mean(effects)

    # Standard error from pre-period residuals
    pre_residuals = y_treatment_pre - synthetic_pre
    se = np.std(pre_residuals) / np.sqrt(len(effects))

    # Confidence interval
    z = stats.norm.ppf(1 - alpha / 2)
    ci_lower = lift - z * se
    ci_upper = lift + z * se

    # T-test
    t_stat = lift / se if se > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(effects) - 1))

    # Totals
    actual_total = np.sum(y_treatment_post)
    counterfactual_total = np.sum(synthetic_post)
    incremental_total = actual_total - counterfactual_total

    # Relative lift
    counterfactual_mean = np.mean(synthetic_post)
    if counterfactual_mean != 0:
        relative_lift = lift / counterfactual_mean
        relative_ci_lower = ci_lower / counterfactual_mean
        relative_ci_upper = ci_upper / counterfactual_mean
    else:
        relative_lift = relative_ci_lower = relative_ci_upper = 0.0

    return GeoLiftResult(
        lift=lift,
        lift_se=se,
        lift_ci_lower=ci_lower,
        lift_ci_upper=ci_upper,
        relative_lift=relative_lift,
        relative_lift_ci_lower=relative_ci_lower,
        relative_lift_ci_upper=relative_ci_upper,
        t_statistic=t_stat,
        p_value=p_value,
        significant=p_value < alpha,
        counterfactual_total=counterfactual_total,
        actual_total=actual_total,
        incremental_total=incremental_total,
        method='Synthetic Control',
        treatment_markets=[treatment_market],
        control_markets=control_markets,
    )


def run_geo_lift(
    data: pd.DataFrame,
    treatment_markets: Union[str, List[str]],
    control_markets: List[str],
    market_col: str,
    time_col: str,
    metric_col: str,
    intervention_time: Union[str, int, pd.Timestamp],
    method: str = 'did',
    alpha: float = 0.05,
) -> GeoLiftResult:
    """
    Convenience function to run geo lift analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    treatment_markets : str or list
        Treatment market(s)
    control_markets : list
        Control markets
    market_col : str
        Column for market identifier
    time_col : str
        Column for time period
    metric_col : str
        Column for the outcome metric
    intervention_time : str, int, or Timestamp
        Time when intervention started
    method : str
        'did' for Difference-in-Differences or 'sc' for Synthetic Control
    alpha : float
        Significance level

    Returns
    -------
    GeoLiftResult
        Analysis results
    """
    if isinstance(treatment_markets, str):
        treatment_markets = [treatment_markets]

    if method == 'did':
        return difference_in_differences(
            data, treatment_markets, control_markets,
            market_col, time_col, metric_col,
            intervention_time, alpha
        )
    elif method == 'sc':
        if len(treatment_markets) > 1:
            raise ValueError("Synthetic control requires a single treatment market")
        return synthetic_control(
            data, treatment_markets[0], control_markets,
            market_col, time_col, metric_col,
            intervention_time, alpha
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'did' or 'sc'")


def plot_geo_lift(
    data: pd.DataFrame,
    result: GeoLiftResult,
    market_col: str,
    time_col: str,
    metric_col: str,
    intervention_time: Union[str, int, pd.Timestamp],
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Plot geo lift analysis results.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    result : GeoLiftResult
        Results from geo lift analysis
    market_col : str
        Column for market identifier
    time_col : str
        Column for time period
    metric_col : str
        Column for the outcome metric
    intervention_time : str, int, or Timestamp
        Time when intervention started
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # Pivot for easier plotting
    wide = data.pivot(index=time_col, columns=market_col, values=metric_col)

    # Panel 1: Treatment vs Control time series
    ax1 = axes[0]

    # Treatment average
    treatment_series = wide[result.treatment_markets].mean(axis=1)
    ax1.plot(treatment_series.index, treatment_series.values, 'b-',
             linewidth=2, label='Treatment')

    # Control average
    control_series = wide[result.control_markets].mean(axis=1)
    ax1.plot(control_series.index, control_series.values, 'r--',
             linewidth=2, label='Control')

    ax1.axvline(x=intervention_time, color='gray', linestyle=':', linewidth=2,
                label='Intervention')
    ax1.set_xlabel('Time')
    ax1.set_ylabel(metric_col)
    ax1.set_title('Treatment vs Control Markets')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Cumulative difference
    ax2 = axes[1]

    # Calculate cumulative difference
    diff = treatment_series - control_series
    cumulative_diff = diff.cumsum()

    # Split by intervention
    pre_mask = cumulative_diff.index < intervention_time
    post_mask = cumulative_diff.index >= intervention_time

    ax2.plot(cumulative_diff.index[pre_mask], cumulative_diff.values[pre_mask],
             'gray', linewidth=2, label='Pre-period')
    ax2.plot(cumulative_diff.index[post_mask], cumulative_diff.values[post_mask],
             'green', linewidth=2, label='Post-period')

    ax2.axvline(x=intervention_time, color='gray', linestyle=':', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Annotate cumulative effect
    ax2.annotate(
        f'Cumulative: {result.incremental_total:,.0f}',
        xy=(0.95, 0.95), xycoords='axes fraction',
        ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Cumulative Difference')
    ax2.set_title('Cumulative Effect Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def placebo_test(
    data: pd.DataFrame,
    treatment_markets: List[str],
    control_markets: List[str],
    market_col: str,
    time_col: str,
    metric_col: str,
    intervention_time: Union[str, int, pd.Timestamp],
    n_placebos: int = 100,
    method: str = 'did',
) -> Tuple[float, np.ndarray]:
    """
    Run placebo tests to validate geo experiment.

    Randomly reassigns treatment status and re-runs analysis
    to create a null distribution of effects.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    treatment_markets : list
        Actual treatment markets
    control_markets : list
        Actual control markets
    market_col : str
        Column for market identifier
    time_col : str
        Column for time period
    metric_col : str
        Column for the outcome metric
    intervention_time : str, int, or Timestamp
        Time when intervention started
    n_placebos : int
        Number of placebo iterations
    method : str
        'did' or 'sc'

    Returns
    -------
    tuple
        (p_value, placebo_effects)
    """
    # Get actual effect
    actual_result = run_geo_lift(
        data, treatment_markets, control_markets,
        market_col, time_col, metric_col,
        intervention_time, method
    )
    actual_effect = actual_result.lift

    # Run placebos
    all_markets = treatment_markets + control_markets
    n_treatment = len(treatment_markets)
    placebo_effects = []

    for _ in range(n_placebos):
        # Random assignment
        np.random.shuffle(all_markets)
        fake_treatment = all_markets[:n_treatment]
        fake_control = all_markets[n_treatment:]

        try:
            placebo_result = run_geo_lift(
                data, fake_treatment, fake_control,
                market_col, time_col, metric_col,
                intervention_time, method
            )
            placebo_effects.append(placebo_result.lift)
        except Exception:
            continue

    placebo_effects = np.array(placebo_effects)

    # Calculate p-value (proportion of placebos >= actual effect)
    p_value = np.mean(np.abs(placebo_effects) >= np.abs(actual_effect))

    return p_value, placebo_effects


def plot_placebo_distribution(
    actual_effect: float,
    placebo_effects: np.ndarray,
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Plot distribution of placebo effects with actual effect marked.

    Parameters
    ----------
    actual_effect : float
        The actual treatment effect
    placebo_effects : array
        Effects from placebo iterations
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(placebo_effects, bins=30, density=True, alpha=0.7,
            edgecolor='black', label='Placebo distribution')
    ax.axvline(x=actual_effect, color='red', linewidth=2,
               label=f'Actual effect: {actual_effect:.2f}')

    # Calculate p-value
    p_value = np.mean(np.abs(placebo_effects) >= np.abs(actual_effect))
    ax.set_title(f'Placebo Test Distribution (p = {p_value:.3f})')
    ax.set_xlabel('Effect Size')
    ax.set_ylabel('Density')
    ax.legend()

    plt.tight_layout()
    return fig
