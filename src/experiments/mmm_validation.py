"""
MMM (Marketing Mix Model) Validation.

This module provides tools to validate Marketing Mix Model outputs
against experimental results (incrementality tests).

Key validations:
1. Compare MMM-predicted incrementality vs experimental lift
2. Calibration assessment (how well MMM predictions match reality)
3. Bias detection and correction recommendations

Use this when:
- You have both MMM results and experimental results for the same channel
- You want to validate/calibrate your MMM
- You need to understand MMM reliability for budget decisions
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class MMMValidationResult:
    """Results from MMM validation against experiments."""

    # Core comparison
    mmm_estimate: float
    experiment_estimate: float
    experiment_ci_lower: float
    experiment_ci_upper: float

    # Calibration metrics
    absolute_error: float
    relative_error: float
    percentage_error: float
    within_ci: bool

    # Statistical comparison
    z_statistic: float
    p_value: float
    significantly_different: bool

    # Recommendations
    bias_direction: str  # 'overestimate', 'underestimate', 'aligned'
    calibration_factor: float  # Multiply MMM by this to align
    confidence_level: str  # 'high', 'medium', 'low'

    def __repr__(self) -> str:
        status = "ALIGNED" if not self.significantly_different else f"MISALIGNED ({self.bias_direction})"
        return (
            f"MMMValidationResult ({status})\n"
            f"  MMM estimate: {self.mmm_estimate:.2f}\n"
            f"  Experiment estimate: {self.experiment_estimate:.2f} "
            f"[{self.experiment_ci_lower:.2f}, {self.experiment_ci_upper:.2f}]\n"
            f"  Percentage error: {self.percentage_error:+.1%}\n"
            f"  Calibration factor: {self.calibration_factor:.2f}"
        )

    def summary(self) -> str:
        """Generate detailed summary report."""
        lines = [
            "=" * 70,
            "              MMM VALIDATION RESULTS",
            "=" * 70,
            "",
            "COMPARISON",
            "-" * 70,
            f"  MMM predicted lift:        {self.mmm_estimate:>12,.2f}",
            f"  Experimental lift:         {self.experiment_estimate:>12,.2f}",
            f"  95% CI:                    [{self.experiment_ci_lower:>10,.2f}, {self.experiment_ci_upper:>10,.2f}]",
            "",
            "CALIBRATION METRICS",
            "-" * 70,
            f"  Absolute error:            {self.absolute_error:>12,.2f}",
            f"  Relative error:            {self.relative_error:>12,.1%}",
            f"  Percentage error:          {self.percentage_error:>+12.1%}",
            f"  MMM within exp. CI:        {'Yes' if self.within_ci else 'No':>12}",
            "",
            "STATISTICAL TEST",
            "-" * 70,
            f"  Z-statistic:               {self.z_statistic:>12.3f}",
            f"  P-value:                   {self.p_value:>12.4f}",
            f"  Significantly different:   {'Yes' if self.significantly_different else 'No':>12}",
            "",
            "RECOMMENDATIONS",
            "-" * 70,
            f"  Bias direction:            {self.bias_direction:>12}",
            f"  Calibration factor:        {self.calibration_factor:>12.2f}",
            f"  Confidence level:          {self.confidence_level:>12}",
            "",
        ]

        # Add interpretation
        lines.append("INTERPRETATION")
        lines.append("-" * 70)

        if self.within_ci:
            lines.append("  The MMM estimate falls within the experimental confidence interval.")
            lines.append("  This suggests reasonable alignment between models.")
        elif self.bias_direction == 'overestimate':
            lines.append(f"  The MMM is OVERESTIMATING by {abs(self.percentage_error):.1%}.")
            lines.append(f"  Consider applying a calibration factor of {self.calibration_factor:.2f}.")
        else:
            lines.append(f"  The MMM is UNDERESTIMATING by {abs(self.percentage_error):.1%}.")
            lines.append(f"  Consider applying a calibration factor of {self.calibration_factor:.2f}.")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)


def validate_mmm_vs_experiment(
    mmm_estimate: float,
    mmm_se: Optional[float],
    experiment_estimate: float,
    experiment_se: float,
    alpha: float = 0.05,
) -> MMMValidationResult:
    """
    Validate MMM estimate against experimental result.

    Parameters
    ----------
    mmm_estimate : float
        MMM-predicted incrementality
    mmm_se : float, optional
        Standard error of MMM estimate (if available)
    experiment_estimate : float
        Experimental estimate of incrementality
    experiment_se : float
        Standard error of experimental estimate
    alpha : float
        Significance level

    Returns
    -------
    MMMValidationResult
        Validation results

    Example
    -------
    >>> # MMM says campaign generated $100k incremental revenue
    >>> # Experiment shows $80k with SE of $15k
    >>> result = validate_mmm_vs_experiment(
    ...     mmm_estimate=100000,
    ...     mmm_se=None,
    ...     experiment_estimate=80000,
    ...     experiment_se=15000
    ... )
    >>> print(result)
    """
    # Calculate confidence interval for experiment
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = experiment_estimate - z_crit * experiment_se
    ci_upper = experiment_estimate + z_crit * experiment_se

    # Calculate errors
    absolute_error = abs(mmm_estimate - experiment_estimate)
    relative_error = absolute_error / abs(experiment_estimate) if experiment_estimate != 0 else float('inf')
    percentage_error = (mmm_estimate - experiment_estimate) / abs(experiment_estimate) if experiment_estimate != 0 else 0

    # Check if MMM is within experimental CI
    within_ci = ci_lower <= mmm_estimate <= ci_upper

    # Statistical test for difference
    if mmm_se is not None:
        combined_se = np.sqrt(mmm_se**2 + experiment_se**2)
    else:
        combined_se = experiment_se

    z_stat = (mmm_estimate - experiment_estimate) / combined_se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    significantly_different = p_value < alpha

    # Bias direction
    if abs(percentage_error) < 0.1:  # Within 10%
        bias_direction = 'aligned'
    elif mmm_estimate > experiment_estimate:
        bias_direction = 'overestimate'
    else:
        bias_direction = 'underestimate'

    # Calibration factor
    calibration_factor = experiment_estimate / mmm_estimate if mmm_estimate != 0 else 1.0

    # Confidence level based on experiment precision
    cv = experiment_se / abs(experiment_estimate) if experiment_estimate != 0 else float('inf')
    if cv < 0.1:
        confidence_level = 'high'
    elif cv < 0.25:
        confidence_level = 'medium'
    else:
        confidence_level = 'low'

    return MMMValidationResult(
        mmm_estimate=mmm_estimate,
        experiment_estimate=experiment_estimate,
        experiment_ci_lower=ci_lower,
        experiment_ci_upper=ci_upper,
        absolute_error=absolute_error,
        relative_error=relative_error,
        percentage_error=percentage_error,
        within_ci=within_ci,
        z_statistic=z_stat,
        p_value=p_value,
        significantly_different=significantly_different,
        bias_direction=bias_direction,
        calibration_factor=calibration_factor,
        confidence_level=confidence_level,
    )


def validate_multiple_channels(
    validations: Dict[str, Dict[str, float]],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Validate MMM across multiple channels.

    Parameters
    ----------
    validations : dict
        Dictionary mapping channel names to dicts with:
        - 'mmm_estimate': MMM prediction
        - 'mmm_se': MMM standard error (optional)
        - 'experiment_estimate': Experimental result
        - 'experiment_se': Experimental standard error
    alpha : float
        Significance level

    Returns
    -------
    pd.DataFrame
        Validation results for all channels

    Example
    -------
    >>> validations = {
    ...     'Facebook': {'mmm_estimate': 100000, 'experiment_estimate': 85000, 'experiment_se': 12000},
    ...     'Google': {'mmm_estimate': 150000, 'experiment_estimate': 160000, 'experiment_se': 20000},
    ... }
    >>> results = validate_multiple_channels(validations)
    """
    results = []

    for channel, data in validations.items():
        result = validate_mmm_vs_experiment(
            mmm_estimate=data['mmm_estimate'],
            mmm_se=data.get('mmm_se'),
            experiment_estimate=data['experiment_estimate'],
            experiment_se=data['experiment_se'],
            alpha=alpha,
        )

        results.append({
            'channel': channel,
            'mmm_estimate': result.mmm_estimate,
            'experiment_estimate': result.experiment_estimate,
            'ci_lower': result.experiment_ci_lower,
            'ci_upper': result.experiment_ci_upper,
            'pct_error': result.percentage_error,
            'within_ci': result.within_ci,
            'p_value': result.p_value,
            'significantly_different': result.significantly_different,
            'bias_direction': result.bias_direction,
            'calibration_factor': result.calibration_factor,
        })

    return pd.DataFrame(results)


def calculate_calibration_metrics(
    mmm_estimates: np.ndarray,
    experiment_estimates: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate overall calibration metrics across multiple validations.

    Parameters
    ----------
    mmm_estimates : array-like
        MMM estimates for each validation
    experiment_estimates : array-like
        Experimental estimates for each validation

    Returns
    -------
    dict
        Calibration metrics:
        - 'mape': Mean Absolute Percentage Error
        - 'bias': Average bias (positive = overestimate)
        - 'correlation': Correlation between MMM and experiments
        - 'slope': Regression slope (MMM = slope * experiment + intercept)
        - 'r_squared': R-squared from regression
    """
    mmm = np.asarray(mmm_estimates)
    exp = np.asarray(experiment_estimates)

    # Remove zeros/NaN
    mask = (exp != 0) & np.isfinite(mmm) & np.isfinite(exp)
    mmm = mmm[mask]
    exp = exp[mask]

    if len(mmm) < 2:
        return {
            'mape': float('nan'),
            'bias': float('nan'),
            'correlation': float('nan'),
            'slope': float('nan'),
            'r_squared': float('nan'),
        }

    # MAPE
    mape = np.mean(np.abs(mmm - exp) / np.abs(exp))

    # Bias
    bias = np.mean((mmm - exp) / np.abs(exp))

    # Correlation
    correlation = np.corrcoef(mmm, exp)[0, 1]

    # Regression: MMM = slope * Experiment + intercept
    slope, intercept, r_value, p_value, std_err = stats.linregress(exp, mmm)

    return {
        'mape': mape,
        'bias': bias,
        'correlation': correlation,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value ** 2,
    }


def plot_validation(
    result: MMMValidationResult,
    channel_name: str = "Channel",
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot single channel validation.

    Parameters
    ----------
    result : MMMValidationResult
        Validation result
    channel_name : str
        Channel name for title
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Experimental estimate with CI
    ax.errorbar(
        0, result.experiment_estimate,
        yerr=[[result.experiment_estimate - result.experiment_ci_lower],
              [result.experiment_ci_upper - result.experiment_estimate]],
        fmt='o', markersize=15, capsize=10, color='blue',
        label='Experiment'
    )

    # MMM estimate
    color = 'green' if result.within_ci else 'red'
    ax.plot(0.2, result.mmm_estimate, 's', markersize=15, color=color,
            label='MMM')

    # Reference lines
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Annotations
    ax.annotate(
        f'MMM: {result.mmm_estimate:,.0f}',
        xy=(0.2, result.mmm_estimate),
        xytext=(0.4, result.mmm_estimate),
        fontsize=10,
        arrowprops=dict(arrowstyle='->', color='gray')
    )

    ax.set_xlim(-0.5, 1)
    ax.set_xticks([])
    ax.set_ylabel('Incremental Impact')
    ax.set_title(f'{channel_name}: MMM vs Experiment Validation')
    ax.legend()

    plt.tight_layout()
    return fig


def plot_calibration(
    mmm_estimates: np.ndarray,
    experiment_estimates: np.ndarray,
    channel_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot calibration across multiple channels.

    Parameters
    ----------
    mmm_estimates : array-like
        MMM estimates
    experiment_estimates : array-like
        Experimental estimates
    channel_names : list, optional
        Names for each point
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    mmm = np.asarray(mmm_estimates)
    exp = np.asarray(experiment_estimates)

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    ax.scatter(exp, mmm, s=100, alpha=0.7)

    # Add labels if provided
    if channel_names is not None:
        for i, name in enumerate(channel_names):
            ax.annotate(name, (exp[i], mmm[i]), xytext=(5, 5),
                        textcoords='offset points', fontsize=9)

    # Perfect calibration line
    lims = [
        min(min(exp), min(mmm)),
        max(max(exp), max(mmm))
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect calibration')

    # Regression line
    mask = (exp != 0) & np.isfinite(mmm) & np.isfinite(exp)
    if np.sum(mask) >= 2:
        slope, intercept, _, _, _ = stats.linregress(exp[mask], mmm[mask])
        x_line = np.linspace(min(exp[mask]), max(exp[mask]), 100)
        ax.plot(x_line, slope * x_line + intercept, 'r-',
                label=f'Fit (slope={slope:.2f})')

    # Calculate and show metrics
    metrics = calculate_calibration_metrics(mmm, exp)
    ax.text(
        0.05, 0.95,
        f"MAPE: {metrics['mape']:.1%}\nBias: {metrics['bias']:+.1%}\nR²: {metrics['r_squared']:.2f}",
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    ax.set_xlabel('Experimental Estimate')
    ax.set_ylabel('MMM Estimate')
    ax.set_title('MMM Calibration: Model vs Reality')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def recommend_calibration(
    mmm_estimates: np.ndarray,
    experiment_estimates: np.ndarray,
    method: str = 'regression',
) -> Dict[str, float]:
    """
    Recommend calibration adjustments for MMM.

    Parameters
    ----------
    mmm_estimates : array-like
        MMM estimates
    experiment_estimates : array-like
        Experimental estimates
    method : str
        'regression' (linear fit) or 'ratio' (simple scaling)

    Returns
    -------
    dict
        Recommended calibration parameters:
        - For 'ratio': {'multiplier': float}
        - For 'regression': {'slope': float, 'intercept': float}
    """
    mmm = np.asarray(mmm_estimates)
    exp = np.asarray(experiment_estimates)

    mask = (exp != 0) & (mmm != 0) & np.isfinite(mmm) & np.isfinite(exp)
    mmm = mmm[mask]
    exp = exp[mask]

    if len(mmm) < 2:
        return {'multiplier': 1.0}

    if method == 'ratio':
        # Simple ratio-based calibration
        multiplier = np.mean(exp / mmm)
        return {'multiplier': multiplier}

    else:  # regression
        # Calibrated_MMM = slope * MMM + intercept ≈ Experiment
        slope, intercept, _, _, _ = stats.linregress(mmm, exp)
        return {
            'slope': slope,
            'intercept': intercept,
        }


def apply_calibration(
    mmm_estimate: float,
    calibration: Dict[str, float],
) -> float:
    """
    Apply calibration to MMM estimate.

    Parameters
    ----------
    mmm_estimate : float
        Original MMM estimate
    calibration : dict
        Calibration parameters from recommend_calibration

    Returns
    -------
    float
        Calibrated estimate
    """
    if 'multiplier' in calibration:
        return mmm_estimate * calibration['multiplier']
    else:
        return calibration['slope'] * mmm_estimate + calibration['intercept']


# =============================================================================
# Adstock Validation
# =============================================================================

def validate_adstock_decay(
    time_series: np.ndarray,
    spend_series: np.ndarray,
    mmm_decay_rate: float,
    experiment_window: int,
) -> Dict[str, float]:
    """
    Validate MMM adstock decay rate against experimental evidence.

    Parameters
    ----------
    time_series : array
        Response variable over time
    spend_series : array
        Spend/media over time
    mmm_decay_rate : float
        Decay rate from MMM (0-1, where 0=instant decay, 1=no decay)
    experiment_window : int
        Number of periods with observable carryover effect from experiment

    Returns
    -------
    dict
        Validation metrics
    """
    # Simple correlation-based decay estimation
    n = len(time_series)

    correlations = []
    for lag in range(min(experiment_window, n // 2)):
        if lag == 0:
            corr = np.corrcoef(spend_series, time_series)[0, 1]
        else:
            corr = np.corrcoef(spend_series[:-lag], time_series[lag:])[0, 1]
        correlations.append(corr)

    correlations = np.array(correlations)

    # Fit exponential decay to correlations
    if correlations[0] > 0:
        # Normalize correlations
        norm_corr = correlations / correlations[0]
        norm_corr = np.maximum(norm_corr, 0.01)  # Avoid log(0)

        # Linear regression on log scale
        x = np.arange(len(norm_corr))
        log_corr = np.log(norm_corr)
        slope, intercept, _, _, _ = stats.linregress(x, log_corr)

        estimated_decay = np.exp(slope)
    else:
        estimated_decay = 0.5  # Default if can't estimate

    # Compare to MMM decay
    decay_diff = abs(mmm_decay_rate - estimated_decay)

    return {
        'mmm_decay_rate': mmm_decay_rate,
        'estimated_decay_rate': estimated_decay,
        'decay_difference': decay_diff,
        'correlation_at_lag0': correlations[0] if len(correlations) > 0 else float('nan'),
        'half_life_mmm': np.log(0.5) / np.log(mmm_decay_rate) if mmm_decay_rate > 0 else float('inf'),
        'half_life_estimated': np.log(0.5) / np.log(estimated_decay) if estimated_decay > 0 else float('inf'),
    }
