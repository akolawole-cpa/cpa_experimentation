"""
Experiment diagnostics and validation tools.

This module provides methods for detecting common issues in A/B tests:
- Sample Ratio Mismatch (SRM) detection
- A/A test validation
- Novelty and primacy effect detection
- Covariate balance checking
- Metric validation and outlier handling
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Tuple, Optional, Union, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class SRMResult:
    """Results from Sample Ratio Mismatch detection."""

    observed_control: int
    observed_treatment: int
    expected_ratio: float
    observed_ratio: float
    chi2_statistic: float
    p_value: float
    is_mismatch: bool
    severity: str  # 'none', 'warning', 'severe'

    def __repr__(self) -> str:
        status = "SRM DETECTED" if self.is_mismatch else "No SRM detected"
        return (
            f"SRMResult({status})\n"
            f"  Expected ratio: {self.expected_ratio:.4f}\n"
            f"  Observed ratio: {self.observed_ratio:.4f}\n"
            f"  Control: {self.observed_control:,}, Treatment: {self.observed_treatment:,}\n"
            f"  Chi-square: {self.chi2_statistic:.2f}, p-value: {self.p_value:.6f}\n"
            f"  Severity: {self.severity}"
        )


@dataclass
class AATestResult:
    """Results from A/A test analysis."""

    n_tests: int
    n_significant: int
    expected_significant: float
    false_positive_rate: float
    p_values: np.ndarray
    chi2_uniformity: float
    uniformity_p_value: float
    is_valid: bool
    warnings: List[str]

    def __repr__(self) -> str:
        status = "VALID" if self.is_valid else "ISSUES DETECTED"
        return (
            f"AATestResult({status})\n"
            f"  Tests run: {self.n_tests}\n"
            f"  Significant results: {self.n_significant} (expected: {self.expected_significant:.1f})\n"
            f"  False positive rate: {self.false_positive_rate:.2%}\n"
            f"  P-value uniformity: chi2={self.chi2_uniformity:.2f}, p={self.uniformity_p_value:.4f}"
        )


# =============================================================================
# Sample Ratio Mismatch (SRM) Detection
# =============================================================================

def detect_srm(
    n_control: int,
    n_treatment: int,
    expected_ratio: float = 1.0,
    alpha: float = 0.001,
) -> SRMResult:
    """
    Detect Sample Ratio Mismatch using chi-square test.

    SRM occurs when the ratio of users in control vs treatment differs
    significantly from the expected ratio. This indicates a bug in
    randomization or data collection.

    Parameters
    ----------
    n_control : int
        Number of observations in control group
    n_treatment : int
        Number of observations in treatment group
    expected_ratio : float, default=1.0
        Expected ratio of treatment to control (n_treatment / n_control)
        Use 1.0 for 50/50 split
    alpha : float, default=0.001
        Significance level (default is very low because SRM is serious)

    Returns
    -------
    SRMResult
        Detection results

    Notes
    -----
    SRM is one of the most common and serious issues in A/B testing.
    Always check for SRM before analyzing experiment results.

    Common causes of SRM:
    - Browser/bot filtering applied unevenly
    - Bucketing bugs
    - Data pipeline issues
    - User logout/login affecting tracking

    Example
    -------
    >>> result = detect_srm(n_control=10000, n_treatment=9500)
    >>> if result.is_mismatch:
    ...     print("WARNING: Do not trust experiment results!")
    """
    total = n_control + n_treatment

    # Expected counts
    expected_control = total / (1 + expected_ratio)
    expected_treatment = total - expected_control

    # Chi-square test
    observed = np.array([n_control, n_treatment])
    expected = np.array([expected_control, expected_treatment])

    chi2 = np.sum((observed - expected) ** 2 / expected)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    # Observed ratio
    observed_ratio = n_treatment / n_control if n_control > 0 else float('inf')

    # Determine severity
    is_mismatch = p_value < alpha
    if not is_mismatch:
        severity = 'none'
    elif p_value < 0.0001:
        severity = 'severe'
    else:
        severity = 'warning'

    return SRMResult(
        observed_control=n_control,
        observed_treatment=n_treatment,
        expected_ratio=expected_ratio,
        observed_ratio=observed_ratio,
        chi2_statistic=chi2,
        p_value=p_value,
        is_mismatch=is_mismatch,
        severity=severity
    )


def detect_srm_multiple(
    groups: Dict[str, int],
    expected_proportions: Optional[Dict[str, float]] = None,
    alpha: float = 0.001,
) -> Tuple[float, float, bool]:
    """
    Detect SRM across multiple groups.

    Parameters
    ----------
    groups : dict
        Dictionary of group names to counts
    expected_proportions : dict, optional
        Expected proportion for each group (should sum to 1)
        If None, assumes equal proportions
    alpha : float, default=0.001
        Significance level

    Returns
    -------
    tuple
        (chi2_statistic, p_value, is_mismatch)

    Example
    -------
    >>> groups = {'A': 5000, 'B': 5100, 'C': 4900}
    >>> chi2, p, mismatch = detect_srm_multiple(groups)
    """
    names = list(groups.keys())
    observed = np.array([groups[n] for n in names])
    total = np.sum(observed)

    if expected_proportions is None:
        expected = np.full(len(names), total / len(names))
    else:
        expected = np.array([expected_proportions[n] * total for n in names])

    chi2 = np.sum((observed - expected) ** 2 / expected)
    df = len(names) - 1
    p_value = 1 - stats.chi2.cdf(chi2, df=df)

    return chi2, p_value, p_value < alpha


# =============================================================================
# A/A Test Validation
# =============================================================================

def run_aa_test(
    control_data: Union[np.ndarray, pd.Series],
    treatment_data: Union[np.ndarray, pd.Series],
    n_simulations: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> AATestResult:
    """
    Run A/A test to validate experiment infrastructure.

    An A/A test runs the same treatment to both groups. Any significant
    differences indicate problems with randomization, instrumentation,
    or the testing framework itself.

    Parameters
    ----------
    control_data : array-like
        Data from "control" group (both groups have same treatment)
    treatment_data : array-like
        Data from "treatment" group
    n_simulations : int, default=1000
        Number of bootstrap simulations for p-value distribution
    alpha : float, default=0.05
        Significance level
    seed : int, optional
        Random seed

    Returns
    -------
    AATestResult
        Validation results

    Notes
    -----
    In a valid A/A test:
    - The false positive rate should be close to alpha
    - P-values should be uniformly distributed
    - About alpha% of simulated tests should be significant
    """
    if seed is not None:
        np.random.seed(seed)

    control = np.asarray(control_data)
    treatment = np.asarray(treatment_data)

    all_data = np.concatenate([control, treatment])
    n_total = len(all_data)
    n_control = len(control)

    p_values = []
    significant_count = 0

    for _ in range(n_simulations):
        # Shuffle and split
        np.random.shuffle(all_data)
        sim_control = all_data[:n_control]
        sim_treatment = all_data[n_control:]

        # Run t-test
        _, p = stats.ttest_ind(sim_control, sim_treatment)
        p_values.append(p)

        if p < alpha:
            significant_count += 1

    p_values = np.array(p_values)
    false_positive_rate = significant_count / n_simulations
    expected_significant = n_simulations * alpha

    # Test for uniform distribution of p-values
    # Using chi-square goodness-of-fit test
    n_bins = 10
    observed_hist, _ = np.histogram(p_values, bins=n_bins, range=(0, 1))
    expected_hist = np.full(n_bins, n_simulations / n_bins)
    chi2_uniform = np.sum((observed_hist - expected_hist) ** 2 / expected_hist)
    uniformity_p = 1 - stats.chi2.cdf(chi2_uniform, df=n_bins - 1)

    # Check for issues
    warnings = []
    is_valid = True

    # Check false positive rate
    ci_low = alpha - 2 * np.sqrt(alpha * (1 - alpha) / n_simulations)
    ci_high = alpha + 2 * np.sqrt(alpha * (1 - alpha) / n_simulations)
    if false_positive_rate < ci_low or false_positive_rate > ci_high:
        warnings.append(f"False positive rate {false_positive_rate:.3f} outside expected range [{ci_low:.3f}, {ci_high:.3f}]")
        is_valid = False

    # Check p-value uniformity
    if uniformity_p < 0.01:
        warnings.append(f"P-values not uniformly distributed (p={uniformity_p:.4f})")
        is_valid = False

    return AATestResult(
        n_tests=n_simulations,
        n_significant=significant_count,
        expected_significant=expected_significant,
        false_positive_rate=false_positive_rate,
        p_values=p_values,
        chi2_uniformity=chi2_uniform,
        uniformity_p_value=uniformity_p,
        is_valid=is_valid,
        warnings=warnings
    )


def plot_aa_test_results(result: AATestResult, figsize: Tuple[int, int] = (12, 4)) -> plt.Figure:
    """
    Visualize A/A test results.

    Parameters
    ----------
    result : AATestResult
        Results from run_aa_test
    figsize : tuple, default=(12, 4)
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # P-value distribution
    ax1 = axes[0]
    ax1.hist(result.p_values, bins=20, density=True, alpha=0.7, edgecolor='black')
    ax1.axhline(y=1.0, color='r', linestyle='--', label='Uniform (expected)')
    ax1.set_xlabel('P-value')
    ax1.set_ylabel('Density')
    ax1.set_title('P-value Distribution')
    ax1.legend()

    # Q-Q plot against uniform
    ax2 = axes[1]
    sorted_p = np.sort(result.p_values)
    expected_quantiles = np.linspace(0, 1, len(sorted_p))
    ax2.scatter(expected_quantiles, sorted_p, alpha=0.5, s=10)
    ax2.plot([0, 1], [0, 1], 'r--', label='Perfect uniform')
    ax2.set_xlabel('Expected (Uniform)')
    ax2.set_ylabel('Observed P-values')
    ax2.set_title('Q-Q Plot')
    ax2.legend()

    plt.tight_layout()
    return fig


# =============================================================================
# Novelty and Primacy Effects
# =============================================================================

def detect_novelty_effect(
    df: pd.DataFrame,
    date_col: str,
    metric_col: str,
    group_col: str,
    control_label: str = 'control',
    treatment_label: str = 'treatment',
    window_days: int = 7,
) -> Tuple[bool, pd.DataFrame, float]:
    """
    Detect novelty or primacy effects in experiment data.

    Novelty effect: Treatment appears better early but effect fades
    Primacy effect: Treatment appears worse early but improves over time

    Parameters
    ----------
    df : pd.DataFrame
        Experiment data with dates
    date_col : str
        Column containing dates
    metric_col : str
        Column containing the metric to analyze
    group_col : str
        Column containing group assignment
    control_label : str
        Label for control group
    treatment_label : str
        Label for treatment group
    window_days : int, default=7
        Size of rolling window for trend analysis

    Returns
    -------
    tuple
        (effect_detected, daily_lifts_df, correlation_with_time)
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Calculate daily means by group
    daily = df.groupby([date_col, group_col])[metric_col].mean().unstack()

    # Calculate daily lift
    daily['lift'] = (daily[treatment_label] - daily[control_label]) / daily[control_label]

    # Calculate day number from start
    daily['day_num'] = (daily.index - daily.index.min()).days

    # Test for time trend in lift
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        daily['day_num'], daily['lift']
    )

    # Effect detected if significant trend
    effect_detected = p_value < 0.05

    result_df = pd.DataFrame({
        'date': daily.index,
        'control_mean': daily[control_label].values,
        'treatment_mean': daily[treatment_label].values,
        'daily_lift': daily['lift'].values,
        'rolling_lift': daily['lift'].rolling(window_days).mean().values
    })

    return effect_detected, result_df, r_value


def plot_novelty_effect(
    df: pd.DataFrame,
    date_col: str = 'date',
    lift_col: str = 'daily_lift',
    rolling_col: str = 'rolling_lift',
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plot lift over time to visualize novelty/primacy effects.

    Parameters
    ----------
    df : pd.DataFrame
        Output from detect_novelty_effect
    date_col, lift_col, rolling_col : str
        Column names
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(df[date_col], df[lift_col], alpha=0.5, label='Daily lift')
    ax.plot(df[date_col], df[rolling_col], 'r-', linewidth=2, label='Rolling average')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

    ax.set_xlabel('Date')
    ax.set_ylabel('Relative Lift')
    ax.set_title('Treatment Effect Over Time')
    ax.legend()

    plt.tight_layout()
    return fig


# =============================================================================
# Covariate Balance
# =============================================================================

def check_covariate_balance(
    df: pd.DataFrame,
    group_col: str,
    covariates: List[str],
    control_label: str = 'control',
    treatment_label: str = 'treatment',
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Check balance of covariates between groups.

    Imbalanced covariates may indicate randomization issues or
    confounding factors.

    Parameters
    ----------
    df : pd.DataFrame
        Experiment data
    group_col : str
        Column containing group assignment
    covariates : list of str
        Columns to check for balance
    control_label, treatment_label : str
        Group labels
    alpha : float
        Significance level

    Returns
    -------
    pd.DataFrame
        Balance statistics for each covariate
    """
    control = df[df[group_col] == control_label]
    treatment = df[df[group_col] == treatment_label]

    results = []
    for cov in covariates:
        ctrl_vals = control[cov].dropna()
        trt_vals = treatment[cov].dropna()

        # Check if numeric or categorical
        if np.issubdtype(ctrl_vals.dtype, np.number):
            # Numeric: use t-test
            t_stat, p_value = stats.ttest_ind(ctrl_vals, trt_vals)
            ctrl_mean = ctrl_vals.mean()
            trt_mean = trt_vals.mean()

            # Standardized mean difference
            pooled_std = np.sqrt(
                ((len(ctrl_vals) - 1) * ctrl_vals.std()**2 +
                 (len(trt_vals) - 1) * trt_vals.std()**2) /
                (len(ctrl_vals) + len(trt_vals) - 2)
            )
            smd = (trt_mean - ctrl_mean) / pooled_std if pooled_std > 0 else 0

            results.append({
                'covariate': cov,
                'type': 'numeric',
                'control_mean': ctrl_mean,
                'treatment_mean': trt_mean,
                'std_mean_diff': smd,
                'test_stat': t_stat,
                'p_value': p_value,
                'balanced': p_value >= alpha
            })
        else:
            # Categorical: use chi-square
            contingency = pd.crosstab(df[group_col], df[cov])
            chi2, p_value, dof, _ = stats.chi2_contingency(contingency)

            results.append({
                'covariate': cov,
                'type': 'categorical',
                'control_mean': None,
                'treatment_mean': None,
                'std_mean_diff': None,
                'test_stat': chi2,
                'p_value': p_value,
                'balanced': p_value >= alpha
            })

    return pd.DataFrame(results)


# =============================================================================
# Metric Validation and Outlier Handling
# =============================================================================

def winsorize_metric(
    data: Union[np.ndarray, pd.Series],
    lower_percentile: float = 0.01,
    upper_percentile: float = 0.99,
) -> np.ndarray:
    """
    Winsorize metric to reduce impact of outliers.

    Values below lower_percentile are set to that percentile's value.
    Values above upper_percentile are set to that percentile's value.

    Parameters
    ----------
    data : array-like
        Metric values
    lower_percentile : float, default=0.01
        Lower percentile (0-1)
    upper_percentile : float, default=0.99
        Upper percentile (0-1)

    Returns
    -------
    np.ndarray
        Winsorized values

    Example
    -------
    >>> revenue = [100, 150, 120, 5000, 110]  # 5000 is outlier
    >>> winsorized = winsorize_metric(revenue, upper_percentile=0.95)
    """
    data = np.asarray(data)
    lower = np.percentile(data, lower_percentile * 100)
    upper = np.percentile(data, upper_percentile * 100)

    return np.clip(data, lower, upper)


def cap_metric(
    data: Union[np.ndarray, pd.Series],
    cap_value: float,
    cap_type: str = 'upper',
) -> np.ndarray:
    """
    Cap metric at a fixed value.

    Parameters
    ----------
    data : array-like
        Metric values
    cap_value : float
        Value to cap at
    cap_type : str
        'upper', 'lower', or 'both'

    Returns
    -------
    np.ndarray
        Capped values
    """
    data = np.asarray(data)

    if cap_type == 'upper':
        return np.minimum(data, cap_value)
    elif cap_type == 'lower':
        return np.maximum(data, cap_value)
    else:  # both
        return np.clip(data, -cap_value, cap_value)


def detect_outliers(
    data: Union[np.ndarray, pd.Series],
    method: str = 'iqr',
    threshold: float = 1.5,
) -> np.ndarray:
    """
    Detect outliers in metric data.

    Parameters
    ----------
    data : array-like
        Metric values
    method : str, default='iqr'
        'iqr' (Interquartile Range) or 'zscore'
    threshold : float, default=1.5
        For IQR: multiplier (1.5 = mild outliers, 3.0 = extreme)
        For zscore: number of standard deviations

    Returns
    -------
    np.ndarray
        Boolean array where True indicates outlier
    """
    data = np.asarray(data)

    if method == 'iqr':
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return (data < lower_bound) | (data > upper_bound)

    else:  # zscore
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        return z_scores > threshold


def calculate_variance_reduction(
    metric_original: np.ndarray,
    metric_adjusted: np.ndarray,
) -> float:
    """
    Calculate variance reduction from metric adjustment.

    Parameters
    ----------
    metric_original : np.ndarray
        Original metric values
    metric_adjusted : np.ndarray
        Adjusted metric values (e.g., after winsorization or CUPED)

    Returns
    -------
    float
        Percentage variance reduction

    Example
    -------
    >>> original = np.array([100, 150, 5000, 120, 110])
    >>> winsorized = winsorize_metric(original)
    >>> reduction = calculate_variance_reduction(original, winsorized)
    >>> print(f"Variance reduced by {reduction:.1%}")
    """
    var_original = np.var(metric_original)
    var_adjusted = np.var(metric_adjusted)

    return 1 - (var_adjusted / var_original)


# =============================================================================
# Pre-Experiment Checks
# =============================================================================

def pre_experiment_checklist(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    date_col: Optional[str] = None,
    user_id_col: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    expected_ratio: float = 1.0,
) -> Dict[str, bool]:
    """
    Run pre-experiment validation checks.

    Parameters
    ----------
    df : pd.DataFrame
        Experiment data
    group_col : str
        Group assignment column
    metric_col : str
        Primary metric column
    date_col : str, optional
        Date column for time-based checks
    user_id_col : str, optional
        User ID column for duplicate checks
    covariates : list, optional
        Covariate columns to check for balance
    expected_ratio : float
        Expected treatment/control ratio

    Returns
    -------
    dict
        Check results (True = passed)
    """
    results = {}

    groups = df[group_col].value_counts()
    control_n = groups.get('control', groups.iloc[1] if len(groups) > 1 else 0)
    treatment_n = groups.get('treatment', groups.iloc[0])

    # Check 1: Sample Ratio Mismatch
    srm = detect_srm(control_n, treatment_n, expected_ratio)
    results['no_srm'] = not srm.is_mismatch

    # Check 2: No missing metric values
    missing_rate = df[metric_col].isna().mean()
    results['low_missing_rate'] = missing_rate < 0.05

    # Check 3: No duplicate users (if user_id provided)
    if user_id_col:
        n_duplicates = df.duplicated(subset=[user_id_col, group_col]).sum()
        results['no_duplicate_users'] = n_duplicates == 0

    # Check 4: Covariate balance (if covariates provided)
    if covariates:
        balance = check_covariate_balance(df, group_col, covariates)
        results['covariates_balanced'] = balance['balanced'].all()

    # Check 5: Reasonable sample size
    results['adequate_sample_size'] = min(control_n, treatment_n) >= 100

    # Check 6: Outlier proportion
    outliers = detect_outliers(df[metric_col].dropna())
    outlier_rate = outliers.mean()
    results['low_outlier_rate'] = outlier_rate < 0.05

    return results


def summarize_experiment_health(checks: Dict[str, bool]) -> str:
    """
    Generate human-readable summary of experiment health checks.

    Parameters
    ----------
    checks : dict
        Output from pre_experiment_checklist

    Returns
    -------
    str
        Formatted summary
    """
    lines = ["Experiment Health Check Summary", "=" * 40]

    check_names = {
        'no_srm': 'No Sample Ratio Mismatch',
        'low_missing_rate': 'Low Missing Data Rate',
        'no_duplicate_users': 'No Duplicate Users',
        'covariates_balanced': 'Covariates Balanced',
        'adequate_sample_size': 'Adequate Sample Size',
        'low_outlier_rate': 'Low Outlier Rate',
    }

    passed = 0
    for key, value in checks.items():
        name = check_names.get(key, key)
        status = "PASS" if value else "FAIL"
        symbol = "+" if value else "X"
        lines.append(f"  [{symbol}] {name}: {status}")
        if value:
            passed += 1

    lines.append("=" * 40)
    lines.append(f"Overall: {passed}/{len(checks)} checks passed")

    if passed == len(checks):
        lines.append("Experiment is ready for analysis!")
    else:
        lines.append("Please investigate failed checks before analysis.")

    return "\n".join(lines)
