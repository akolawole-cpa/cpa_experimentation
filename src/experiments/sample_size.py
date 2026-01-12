"""
Sample size and power analysis for experiment planning.

This module provides comprehensive tools for:
- Sample size calculations for different test types
- Power analysis and power curves
- Minimum detectable effect (MDE) calculations
- Effect size conversions
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional, List, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class PowerAnalysisResult:
    """Results from power analysis calculations."""

    sample_size_per_group: int
    total_sample_size: int
    power: float
    alpha: float
    effect_size: float
    effect_size_type: str  # 'absolute', 'relative', 'cohens_d', etc.
    test_type: str
    notes: str = ""

    def __repr__(self) -> str:
        return (
            f"PowerAnalysisResult\n"
            f"  Sample size per group: {self.sample_size_per_group:,}\n"
            f"  Total sample size: {self.total_sample_size:,}\n"
            f"  Power: {self.power:.2%}\n"
            f"  Alpha: {self.alpha}\n"
            f"  Effect size ({self.effect_size_type}): {self.effect_size}\n"
            f"  Test type: {self.test_type}"
        )


# =============================================================================
# Sample Size Calculations
# =============================================================================

def sample_size_proportion(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    two_tailed: bool = True,
    ratio: float = 1.0,
) -> PowerAnalysisResult:
    """
    Calculate sample size for comparing two proportions.

    Parameters
    ----------
    baseline_rate : float
        Expected conversion rate for control group (0-1)
    mde : float
        Minimum detectable effect (absolute difference in proportions)
    alpha : float, default=0.05
        Significance level
    power : float, default=0.8
        Statistical power (1 - Type II error rate)
    two_tailed : bool, default=True
        Whether to use two-tailed test
    ratio : float, default=1.0
        Ratio of treatment to control sample sizes (n_treatment / n_control)

    Returns
    -------
    PowerAnalysisResult
        Sample size and configuration details

    Example
    -------
    >>> result = sample_size_proportion(baseline_rate=0.10, mde=0.02)
    >>> print(f"Need {result.sample_size_per_group} users per group")
    """
    p1 = baseline_rate
    p2 = baseline_rate + mde
    pooled_p = (p1 + p2 * ratio) / (1 + ratio)

    if two_tailed:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)

    z_beta = stats.norm.ppf(power)

    # Sample size formula with unequal allocation
    var1 = p1 * (1 - p1)
    var2 = p2 * (1 - p2)

    numerator = (
        z_alpha * np.sqrt((1 + 1/ratio) * pooled_p * (1 - pooled_p))
        + z_beta * np.sqrt(var1 + var2/ratio)
    ) ** 2
    denominator = (p2 - p1) ** 2

    n_control = int(np.ceil(numerator / denominator))
    n_treatment = int(np.ceil(n_control * ratio))

    return PowerAnalysisResult(
        sample_size_per_group=n_control,
        total_sample_size=n_control + n_treatment,
        power=power,
        alpha=alpha,
        effect_size=mde,
        effect_size_type='absolute_proportion',
        test_type='two_proportion_z_test',
        notes=f"Control rate: {baseline_rate:.2%}, Expected treatment rate: {p2:.2%}"
    )


def sample_size_continuous(
    baseline_std: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    two_tailed: bool = True,
    ratio: float = 1.0,
) -> PowerAnalysisResult:
    """
    Calculate sample size for comparing two means (t-test).

    Parameters
    ----------
    baseline_std : float
        Expected standard deviation (assumed equal for both groups)
    mde : float
        Minimum detectable effect (absolute difference in means)
    alpha : float, default=0.05
        Significance level
    power : float, default=0.8
        Statistical power
    two_tailed : bool, default=True
        Whether to use two-tailed test
    ratio : float, default=1.0
        Ratio of treatment to control sample sizes

    Returns
    -------
    PowerAnalysisResult
        Sample size and configuration details

    Example
    -------
    >>> result = sample_size_continuous(baseline_std=10, mde=2)
    >>> print(f"Need {result.sample_size_per_group} per group")
    """
    if two_tailed:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)

    z_beta = stats.norm.ppf(power)

    # Cohen's d
    d = mde / baseline_std

    # Sample size with unequal allocation
    n_control = int(np.ceil((1 + 1/ratio) * ((z_alpha + z_beta) / d) ** 2))
    n_treatment = int(np.ceil(n_control * ratio))

    return PowerAnalysisResult(
        sample_size_per_group=n_control,
        total_sample_size=n_control + n_treatment,
        power=power,
        alpha=alpha,
        effect_size=mde,
        effect_size_type='absolute_continuous',
        test_type='two_sample_t_test',
        notes=f"Cohen's d: {d:.3f}"
    )


def sample_size_cohens_d(
    d: float,
    alpha: float = 0.05,
    power: float = 0.8,
    two_tailed: bool = True,
    ratio: float = 1.0,
) -> PowerAnalysisResult:
    """
    Calculate sample size given Cohen's d effect size.

    Parameters
    ----------
    d : float
        Cohen's d effect size (standardized mean difference)
    alpha : float, default=0.05
        Significance level
    power : float, default=0.8
        Statistical power
    two_tailed : bool, default=True
        Whether to use two-tailed test
    ratio : float, default=1.0
        Ratio of treatment to control sample sizes

    Returns
    -------
    PowerAnalysisResult
        Sample size and configuration details

    Notes
    -----
    Cohen's d interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    """
    if two_tailed:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)

    z_beta = stats.norm.ppf(power)

    n_control = int(np.ceil((1 + 1/ratio) * ((z_alpha + z_beta) / d) ** 2))
    n_treatment = int(np.ceil(n_control * ratio))

    return PowerAnalysisResult(
        sample_size_per_group=n_control,
        total_sample_size=n_control + n_treatment,
        power=power,
        alpha=alpha,
        effect_size=d,
        effect_size_type='cohens_d',
        test_type='two_sample_t_test'
    )


def sample_size_anova(
    n_groups: int,
    effect_size_f: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> PowerAnalysisResult:
    """
    Calculate sample size for one-way ANOVA.

    Parameters
    ----------
    n_groups : int
        Number of groups to compare
    effect_size_f : float
        Cohen's f effect size
    alpha : float, default=0.05
        Significance level
    power : float, default=0.8
        Statistical power

    Returns
    -------
    PowerAnalysisResult
        Sample size and configuration details

    Notes
    -----
    Cohen's f interpretation:
    - f = 0.10: small effect
    - f = 0.25: medium effect
    - f = 0.40: large effect

    Cohen's f = sqrt(eta_squared / (1 - eta_squared))
    """
    # Approximate using non-central F distribution
    df1 = n_groups - 1

    # Iterative search for sample size
    for n_per_group in range(2, 100000):
        total_n = n_per_group * n_groups
        df2 = total_n - n_groups
        ncp = effect_size_f ** 2 * total_n  # Non-centrality parameter

        # Critical F value
        f_crit = stats.f.ppf(1 - alpha, df1, df2)

        # Power calculation
        calc_power = 1 - stats.ncf.cdf(f_crit, df1, df2, ncp)

        if calc_power >= power:
            return PowerAnalysisResult(
                sample_size_per_group=n_per_group,
                total_sample_size=total_n,
                power=calc_power,
                alpha=alpha,
                effect_size=effect_size_f,
                effect_size_type='cohens_f',
                test_type='one_way_anova',
                notes=f"Number of groups: {n_groups}"
            )

    raise ValueError("Could not find sample size within search range")


def sample_size_paired(
    baseline_std: float,
    mde: float,
    correlation: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.8,
    two_tailed: bool = True,
) -> PowerAnalysisResult:
    """
    Calculate sample size for paired t-test.

    Parameters
    ----------
    baseline_std : float
        Expected standard deviation of individual measurements
    mde : float
        Minimum detectable effect (expected mean difference)
    correlation : float, default=0.5
        Expected correlation between paired measurements
    alpha : float, default=0.05
        Significance level
    power : float, default=0.8
        Statistical power
    two_tailed : bool, default=True
        Whether to use two-tailed test

    Returns
    -------
    PowerAnalysisResult
        Sample size (number of pairs)

    Notes
    -----
    Higher correlation between pairs leads to smaller required sample size.
    """
    if two_tailed:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)

    z_beta = stats.norm.ppf(power)

    # Standard deviation of differences
    std_diff = baseline_std * np.sqrt(2 * (1 - correlation))

    # Cohen's d for paired data
    d = mde / std_diff

    n_pairs = int(np.ceil(((z_alpha + z_beta) / d) ** 2))

    return PowerAnalysisResult(
        sample_size_per_group=n_pairs,
        total_sample_size=n_pairs,  # For paired, total = number of pairs
        power=power,
        alpha=alpha,
        effect_size=mde,
        effect_size_type='absolute_paired',
        test_type='paired_t_test',
        notes=f"Assumed correlation: {correlation}, SD of differences: {std_diff:.3f}"
    )


# =============================================================================
# Power Calculations
# =============================================================================

def power_proportion(
    n_per_group: int,
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    two_tailed: bool = True,
) -> float:
    """
    Calculate power for a two-proportion z-test.

    Parameters
    ----------
    n_per_group : int
        Sample size per group
    baseline_rate : float
        Expected control group proportion
    mde : float
        Minimum detectable effect
    alpha : float, default=0.05
        Significance level
    two_tailed : bool, default=True
        Whether to use two-tailed test

    Returns
    -------
    float
        Statistical power (0 to 1)
    """
    p1 = baseline_rate
    p2 = baseline_rate + mde
    pooled_p = (p1 + p2) / 2

    if two_tailed:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)

    se_pooled = np.sqrt(2 * pooled_p * (1 - pooled_p) / n_per_group)
    se_unpooled = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n_per_group)

    z_beta = (abs(p2 - p1) - z_alpha * se_pooled) / se_unpooled

    return float(stats.norm.cdf(z_beta))


def power_continuous(
    n_per_group: int,
    baseline_std: float,
    mde: float,
    alpha: float = 0.05,
    two_tailed: bool = True,
) -> float:
    """
    Calculate power for a two-sample t-test.

    Parameters
    ----------
    n_per_group : int
        Sample size per group
    baseline_std : float
        Expected standard deviation
    mde : float
        Minimum detectable effect
    alpha : float, default=0.05
        Significance level
    two_tailed : bool, default=True
        Whether to use two-tailed test

    Returns
    -------
    float
        Statistical power (0 to 1)
    """
    if two_tailed:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)

    d = mde / baseline_std
    ncp = d * np.sqrt(n_per_group / 2)
    z_beta = ncp - z_alpha

    return float(stats.norm.cdf(z_beta))


# =============================================================================
# MDE Calculations
# =============================================================================

def mde_proportion(
    n_per_group: int,
    baseline_rate: float,
    alpha: float = 0.05,
    power: float = 0.8,
    two_tailed: bool = True,
) -> float:
    """
    Calculate minimum detectable effect for proportions.

    Parameters
    ----------
    n_per_group : int
        Sample size per group
    baseline_rate : float
        Expected control group proportion
    alpha : float, default=0.05
        Significance level
    power : float, default=0.8
        Statistical power
    two_tailed : bool, default=True
        Whether to use two-tailed test

    Returns
    -------
    float
        Minimum detectable effect (absolute proportion difference)
    """
    if two_tailed:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)

    z_beta = stats.norm.ppf(power)

    # Approximate MDE using baseline variance
    se = np.sqrt(2 * baseline_rate * (1 - baseline_rate) / n_per_group)
    mde = (z_alpha + z_beta) * se

    return float(mde)


def mde_continuous(
    n_per_group: int,
    baseline_std: float,
    alpha: float = 0.05,
    power: float = 0.8,
    two_tailed: bool = True,
) -> float:
    """
    Calculate minimum detectable effect for continuous metrics.

    Parameters
    ----------
    n_per_group : int
        Sample size per group
    baseline_std : float
        Expected standard deviation
    alpha : float, default=0.05
        Significance level
    power : float, default=0.8
        Statistical power
    two_tailed : bool, default=True
        Whether to use two-tailed test

    Returns
    -------
    float
        Minimum detectable effect (absolute difference)
    """
    if two_tailed:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)

    z_beta = stats.norm.ppf(power)

    se = baseline_std * np.sqrt(2 / n_per_group)
    mde = (z_alpha + z_beta) * se

    return float(mde)


# =============================================================================
# Effect Size Conversions
# =============================================================================

def cohens_d_to_r(d: float) -> float:
    """
    Convert Cohen's d to correlation coefficient r.

    Parameters
    ----------
    d : float
        Cohen's d effect size

    Returns
    -------
    float
        Correlation coefficient r
    """
    return d / np.sqrt(d**2 + 4)


def r_to_cohens_d(r: float) -> float:
    """
    Convert correlation coefficient r to Cohen's d.

    Parameters
    ----------
    r : float
        Correlation coefficient

    Returns
    -------
    float
        Cohen's d effect size
    """
    return 2 * r / np.sqrt(1 - r**2)


def cohens_d_to_f(d: float) -> float:
    """
    Convert Cohen's d to Cohen's f (for ANOVA with 2 groups).

    Parameters
    ----------
    d : float
        Cohen's d effect size

    Returns
    -------
    float
        Cohen's f effect size
    """
    return abs(d) / 2


def eta_squared_to_f(eta_sq: float) -> float:
    """
    Convert eta-squared to Cohen's f.

    Parameters
    ----------
    eta_sq : float
        Eta-squared (proportion of variance explained)

    Returns
    -------
    float
        Cohen's f effect size
    """
    return np.sqrt(eta_sq / (1 - eta_sq))


def odds_ratio_to_d(or_: float) -> float:
    """
    Convert odds ratio to Cohen's d (approximate).

    Parameters
    ----------
    or_ : float
        Odds ratio

    Returns
    -------
    float
        Cohen's d (approximate)
    """
    return np.log(or_) * np.sqrt(3) / np.pi


def relative_lift_to_absolute(
    baseline_rate: float,
    relative_lift: float,
) -> float:
    """
    Convert relative lift to absolute effect.

    Parameters
    ----------
    baseline_rate : float
        Baseline conversion rate
    relative_lift : float
        Relative lift (e.g., 0.10 for 10% lift)

    Returns
    -------
    float
        Absolute effect (difference in proportions)

    Example
    -------
    >>> abs_effect = relative_lift_to_absolute(0.10, 0.20)  # 20% lift
    >>> print(f"Absolute MDE: {abs_effect}")  # 0.02
    """
    return baseline_rate * relative_lift


# =============================================================================
# Visualization
# =============================================================================

def plot_power_curve(
    baseline_rate: Optional[float] = None,
    baseline_std: Optional[float] = None,
    mde_range: Optional[Tuple[float, float]] = None,
    n_per_group: Optional[int] = None,
    alpha: float = 0.05,
    n_points: int = 50,
    metric_type: str = 'proportion',
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot power curve showing power vs effect size or sample size.

    Parameters
    ----------
    baseline_rate : float, optional
        Baseline rate for proportions
    baseline_std : float, optional
        Baseline std for continuous metrics
    mde_range : tuple, optional
        Range of MDE values to plot
    n_per_group : int, optional
        Fixed sample size (if plotting power vs MDE)
    alpha : float, default=0.05
        Significance level
    n_points : int, default=50
        Number of points to plot
    metric_type : str, default='proportion'
        Either 'proportion' or 'continuous'
    figsize : tuple, default=(10, 6)
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Power curve figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if n_per_group is not None:
        # Plot power vs MDE for fixed sample size
        if mde_range is None:
            if metric_type == 'proportion' and baseline_rate:
                mde_range = (0.001, min(baseline_rate, 1 - baseline_rate) * 0.5)
            else:
                mde_range = (0.01, 0.5)

        mde_values = np.linspace(mde_range[0], mde_range[1], n_points)
        powers = []

        for mde in mde_values:
            if metric_type == 'proportion':
                p = power_proportion(n_per_group, baseline_rate or 0.1, mde, alpha)
            else:
                p = power_continuous(n_per_group, baseline_std or 1.0, mde, alpha)
            powers.append(p)

        ax.plot(mde_values, powers, 'b-', linewidth=2)
        ax.set_xlabel('Minimum Detectable Effect (MDE)')
        ax.set_title(f'Power Curve (n={n_per_group:,} per group, α={alpha})')

    else:
        # Plot power vs sample size for fixed MDE
        if mde_range:
            mde = mde_range[0]  # Use first value as the fixed MDE
        else:
            mde = 0.02 if metric_type == 'proportion' else 0.2

        sample_sizes = np.logspace(2, 5, n_points).astype(int)
        powers = []

        for n in sample_sizes:
            if metric_type == 'proportion':
                p = power_proportion(n, baseline_rate or 0.1, mde, alpha)
            else:
                p = power_continuous(n, baseline_std or 1.0, mde, alpha)
            powers.append(p)

        ax.plot(sample_sizes, powers, 'b-', linewidth=2)
        ax.set_xlabel('Sample Size per Group')
        ax.set_xscale('log')
        ax.set_title(f'Power Curve (MDE={mde}, α={alpha})')

    ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='80% Power')
    ax.axhline(y=0.9, color='g', linestyle='--', alpha=0.7, label='90% Power')
    ax.set_ylabel('Statistical Power')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_sample_size_curve(
    baseline_rate: Optional[float] = None,
    baseline_std: Optional[float] = None,
    mde_range: Tuple[float, float] = None,
    alpha: float = 0.05,
    power: float = 0.8,
    n_points: int = 50,
    metric_type: str = 'proportion',
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot sample size requirements across different effect sizes.

    Parameters
    ----------
    baseline_rate : float, optional
        Baseline rate for proportions
    baseline_std : float, optional
        Baseline std for continuous metrics
    mde_range : tuple
        Range of MDE values to plot
    alpha : float, default=0.05
        Significance level
    power : float, default=0.8
        Statistical power
    n_points : int, default=50
        Number of points to plot
    metric_type : str, default='proportion'
        Either 'proportion' or 'continuous'
    figsize : tuple, default=(10, 6)
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Sample size curve figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if mde_range is None:
        if metric_type == 'proportion' and baseline_rate:
            mde_range = (0.005, min(baseline_rate, 1 - baseline_rate) * 0.3)
        else:
            mde_range = (0.05, 0.5)

    mde_values = np.linspace(mde_range[0], mde_range[1], n_points)
    sample_sizes = []

    for mde in mde_values:
        if metric_type == 'proportion':
            result = sample_size_proportion(baseline_rate or 0.1, mde, alpha, power)
        else:
            result = sample_size_continuous(baseline_std or 1.0, mde, alpha, power)
        sample_sizes.append(result.sample_size_per_group)

    ax.plot(mde_values, sample_sizes, 'b-', linewidth=2)
    ax.set_xlabel('Minimum Detectable Effect (MDE)')
    ax.set_ylabel('Sample Size per Group')
    ax.set_yscale('log')
    ax.set_title(f'Sample Size Requirements (Power={power:.0%}, α={alpha})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_sample_size_table(
    baseline_rate: Optional[float] = None,
    baseline_std: Optional[float] = None,
    mde_values: List[float] = None,
    power_values: List[float] = None,
    alpha: float = 0.05,
    metric_type: str = 'proportion',
) -> pd.DataFrame:
    """
    Create a sample size lookup table.

    Parameters
    ----------
    baseline_rate : float, optional
        Baseline rate for proportions
    baseline_std : float, optional
        Baseline std for continuous metrics
    mde_values : list of float
        Effect sizes to include
    power_values : list of float
        Power levels to include
    alpha : float, default=0.05
        Significance level
    metric_type : str, default='proportion'
        Either 'proportion' or 'continuous'

    Returns
    -------
    pd.DataFrame
        Sample size table with MDE as rows and power as columns
    """
    if mde_values is None:
        if metric_type == 'proportion':
            mde_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        else:
            mde_values = [0.1, 0.2, 0.3, 0.4, 0.5]

    if power_values is None:
        power_values = [0.7, 0.8, 0.9, 0.95]

    data = []
    for mde in mde_values:
        row = {'MDE': mde}
        for pwr in power_values:
            if metric_type == 'proportion':
                result = sample_size_proportion(baseline_rate or 0.1, mde, alpha, pwr)
            else:
                result = sample_size_continuous(baseline_std or 1.0, mde, alpha, pwr)
            row[f'Power {pwr:.0%}'] = result.sample_size_per_group
        data.append(row)

    return pd.DataFrame(data).set_index('MDE')
