"""
Interval-based hypothesis testing for experimentation.

This module provides methods for:
- Superiority testing: Prove treatment is better by margin δ
- Non-inferiority testing: Prove treatment is not worse by margin δ
- Equivalence testing (TOST): Prove treatments are equivalent within margin δ

These tests go beyond "is there a difference?" to answer practical questions
about the magnitude and direction of effects.
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List
from enum import Enum
import matplotlib.pyplot as plt


class HypothesisType(Enum):
    """Type of interval-based hypothesis test."""
    SUPERIORITY = "superiority"
    NON_INFERIORITY = "non_inferiority"
    EQUIVALENCE = "equivalence"


@dataclass
class IntervalTestResult:
    """Results from interval-based hypothesis testing."""

    # Effect estimates
    effect: float  # Point estimate (treatment - control)
    se: float  # Standard error of effect
    ci_lower: float  # Confidence interval lower bound
    ci_upper: float  # Confidence interval upper bound

    # Test configuration
    hypothesis_type: HypothesisType
    margin: float  # Non-inferiority/equivalence margin (delta)
    alpha: float

    # Test results
    test_statistic: float
    p_value: float
    conclusion: str  # Human-readable conclusion
    is_significant: bool

    def __repr__(self) -> str:
        status = "PASSED" if self.is_significant else "FAILED"
        return (
            f"IntervalTestResult({self.hypothesis_type.value}, {status})\n"
            f"  Effect: {self.effect:.4f} (SE: {self.se:.4f})\n"
            f"  95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]\n"
            f"  Margin (δ): {self.margin:.4f}\n"
            f"  p-value: {self.p_value:.4f}\n"
            f"  {self.conclusion}"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'effect': self.effect,
            'se': self.se,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'hypothesis_type': self.hypothesis_type.value,
            'margin': self.margin,
            'alpha': self.alpha,
            'test_statistic': self.test_statistic,
            'p_value': self.p_value,
            'conclusion': self.conclusion,
            'is_significant': self.is_significant
        }


def _calculate_effect_and_se(
    control_data: Union[np.ndarray, pd.Series, List[float]],
    treatment_data: Union[np.ndarray, pd.Series, List[float]],
) -> Tuple[float, float, float, float, int, int]:
    """Calculate effect size and standard error for two groups."""
    control = np.asarray(control_data)
    treatment = np.asarray(treatment_data)

    mean_control = np.mean(control)
    mean_treatment = np.mean(treatment)
    n_control = len(control)
    n_treatment = len(treatment)

    var_control = np.var(control, ddof=1)
    var_treatment = np.var(treatment, ddof=1)

    effect = mean_treatment - mean_control
    se = np.sqrt(var_control / n_control + var_treatment / n_treatment)

    return effect, se, mean_control, mean_treatment, n_control, n_treatment


def superiority_test(
    control_data: Union[np.ndarray, pd.Series, List[float]],
    treatment_data: Union[np.ndarray, pd.Series, List[float]],
    margin: float,
    alpha: float = 0.05,
) -> IntervalTestResult:
    """
    Perform a superiority test.

    Tests whether the treatment is better than control by at least a margin δ.

    H0: μ_treatment - μ_control <= δ (treatment is not superior)
    H1: μ_treatment - μ_control > δ (treatment is superior)

    The treatment is declared superior if the lower bound of the CI
    for the effect is greater than the margin δ.

    Parameters
    ----------
    control_data : array-like
        Observations from control group
    treatment_data : array-like
        Observations from treatment group
    margin : float
        Superiority margin δ. Treatment must beat control by at least this much.
        Use a positive value (e.g., 0.02 for 2 percentage points)
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    IntervalTestResult
        Test results including conclusion

    Example
    -------
    >>> control = [10, 12, 11, 13, 12]
    >>> treatment = [15, 17, 16, 18, 17]
    >>> result = superiority_test(control, treatment, margin=2.0)
    >>> print(result)

    Notes
    -----
    Superiority testing is used when you want to prove the new treatment
    is meaningfully better, not just statistically different. This is common
    when introducing a new, more expensive treatment that should show
    substantial improvement to justify the cost.
    """
    effect, se, mean_c, mean_t, n_c, n_t = _calculate_effect_and_se(
        control_data, treatment_data
    )

    # Degrees of freedom (Welch-Satterthwaite)
    var_c = np.var(np.asarray(control_data), ddof=1)
    var_t = np.var(np.asarray(treatment_data), ddof=1)
    df = (var_c/n_c + var_t/n_t)**2 / (
        (var_c/n_c)**2 / (n_c - 1) + (var_t/n_t)**2 / (n_t - 1)
    )

    # One-sided test: H0: effect <= margin, H1: effect > margin
    t_stat = (effect - margin) / se
    p_value = 1 - stats.t.cdf(t_stat, df)

    # Confidence interval (one-sided, lower bound only for superiority)
    t_crit = stats.t.ppf(1 - alpha, df)
    ci_lower = effect - t_crit * se
    ci_upper = float('inf')  # One-sided

    # For reporting, also compute two-sided CI
    t_crit_2 = stats.t.ppf(1 - alpha/2, df)
    ci_lower_2sided = effect - t_crit_2 * se
    ci_upper_2sided = effect + t_crit_2 * se

    is_significant = p_value < alpha

    if is_significant:
        conclusion = f"Treatment is SUPERIOR to control by at least {margin:.4f}"
    else:
        conclusion = f"Cannot conclude treatment is superior by {margin:.4f}"

    return IntervalTestResult(
        effect=effect,
        se=se,
        ci_lower=ci_lower_2sided,  # Report 2-sided for interpretability
        ci_upper=ci_upper_2sided,
        hypothesis_type=HypothesisType.SUPERIORITY,
        margin=margin,
        alpha=alpha,
        test_statistic=t_stat,
        p_value=p_value,
        conclusion=conclusion,
        is_significant=is_significant
    )


def non_inferiority_test(
    control_data: Union[np.ndarray, pd.Series, List[float]],
    treatment_data: Union[np.ndarray, pd.Series, List[float]],
    margin: float,
    alpha: float = 0.05,
) -> IntervalTestResult:
    """
    Perform a non-inferiority test.

    Tests whether the treatment is not worse than control by more than margin δ.

    H0: μ_treatment - μ_control <= -δ (treatment is inferior)
    H1: μ_treatment - μ_control > -δ (treatment is non-inferior)

    The treatment is declared non-inferior if the lower bound of the CI
    for the effect is greater than -δ (the negative of the margin).

    Parameters
    ----------
    control_data : array-like
        Observations from control group
    treatment_data : array-like
        Observations from treatment group
    margin : float
        Non-inferiority margin δ. Use a positive value.
        Treatment can be at most δ worse than control and still be acceptable.
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    IntervalTestResult
        Test results including conclusion

    Example
    -------
    >>> # Testing if new cheaper treatment is not much worse than standard
    >>> control = [100, 105, 98, 102, 101]  # Standard treatment
    >>> treatment = [97, 100, 96, 99, 98]   # New cheaper treatment
    >>> result = non_inferiority_test(control, treatment, margin=5.0)
    >>> print(result)

    Notes
    -----
    Non-inferiority testing is used when you want to show a new treatment
    is "not unacceptably worse" than the standard. This is common when:
    - The new treatment is cheaper or has fewer side effects
    - The new treatment is more convenient
    - You want to show treatments are interchangeable

    The margin δ represents the largest clinically acceptable difference.
    """
    effect, se, mean_c, mean_t, n_c, n_t = _calculate_effect_and_se(
        control_data, treatment_data
    )

    # Degrees of freedom
    var_c = np.var(np.asarray(control_data), ddof=1)
    var_t = np.var(np.asarray(treatment_data), ddof=1)
    df = (var_c/n_c + var_t/n_t)**2 / (
        (var_c/n_c)**2 / (n_c - 1) + (var_t/n_t)**2 / (n_t - 1)
    )

    # One-sided test: H0: effect <= -margin, H1: effect > -margin
    t_stat = (effect - (-margin)) / se  # = (effect + margin) / se
    p_value = 1 - stats.t.cdf(t_stat, df)

    # Confidence interval
    t_crit_2 = stats.t.ppf(1 - alpha/2, df)
    ci_lower = effect - t_crit_2 * se
    ci_upper = effect + t_crit_2 * se

    is_significant = p_value < alpha

    if is_significant:
        conclusion = f"Treatment is NON-INFERIOR to control (not worse by more than {margin:.4f})"
    else:
        conclusion = f"Cannot conclude treatment is non-inferior (margin = {margin:.4f})"

    return IntervalTestResult(
        effect=effect,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        hypothesis_type=HypothesisType.NON_INFERIORITY,
        margin=margin,
        alpha=alpha,
        test_statistic=t_stat,
        p_value=p_value,
        conclusion=conclusion,
        is_significant=is_significant
    )


def equivalence_test(
    control_data: Union[np.ndarray, pd.Series, List[float]],
    treatment_data: Union[np.ndarray, pd.Series, List[float]],
    margin: float,
    alpha: float = 0.05,
) -> IntervalTestResult:
    """
    Perform a Two One-Sided Tests (TOST) equivalence test.

    Tests whether the treatment effect is within ±margin of zero.

    H0: |μ_treatment - μ_control| >= δ (treatments differ)
    H1: |μ_treatment - μ_control| < δ (treatments are equivalent)

    This uses the TOST procedure:
    1. Test H01: effect <= -δ vs H11: effect > -δ (lower bound)
    2. Test H02: effect >= +δ vs H12: effect < +δ (upper bound)

    Equivalence is concluded if BOTH null hypotheses are rejected.

    Parameters
    ----------
    control_data : array-like
        Observations from control group
    treatment_data : array-like
        Observations from treatment group
    margin : float
        Equivalence margin δ. Use a positive value.
        Treatments are equivalent if effect is within ±δ.
    alpha : float, default=0.05
        Significance level (applied to EACH one-sided test)

    Returns
    -------
    IntervalTestResult
        Test results including conclusion

    Example
    -------
    >>> # Testing if generic drug is equivalent to brand name
    >>> brand = [100, 102, 98, 101, 99]
    >>> generic = [99, 101, 97, 100, 100]
    >>> result = equivalence_test(brand, generic, margin=5.0)
    >>> print(result)

    Notes
    -----
    Equivalence testing is used when you want to prove two treatments
    are practically the same. Unlike non-inferiority, this requires
    showing the difference is bounded on BOTH sides.

    Common applications:
    - Generic drug approval
    - Demonstrating consistency across batches/sites
    - Showing algorithm changes don't meaningfully affect outcomes

    The p-value returned is the MAXIMUM of the two one-sided p-values.
    """
    effect, se, mean_c, mean_t, n_c, n_t = _calculate_effect_and_se(
        control_data, treatment_data
    )

    # Degrees of freedom
    var_c = np.var(np.asarray(control_data), ddof=1)
    var_t = np.var(np.asarray(treatment_data), ddof=1)
    df = (var_c/n_c + var_t/n_t)**2 / (
        (var_c/n_c)**2 / (n_c - 1) + (var_t/n_t)**2 / (n_t - 1)
    )

    # TOST procedure: Two one-sided tests
    # Test 1: H0: effect <= -margin, H1: effect > -margin (lower bound)
    t_lower = (effect - (-margin)) / se
    p_lower = 1 - stats.t.cdf(t_lower, df)

    # Test 2: H0: effect >= +margin, H1: effect < +margin (upper bound)
    t_upper = (effect - margin) / se
    p_upper = stats.t.cdf(t_upper, df)

    # TOST p-value is the maximum of the two
    p_value = max(p_lower, p_upper)

    # Use the test statistic from the "harder" test
    if p_lower >= p_upper:
        t_stat = t_lower
    else:
        t_stat = t_upper

    # Confidence interval (1-2*alpha CI is used for equivalence)
    t_crit = stats.t.ppf(1 - alpha, df)
    ci_lower = effect - t_crit * se
    ci_upper = effect + t_crit * se

    is_significant = p_value < alpha

    if is_significant:
        conclusion = f"Treatments are EQUIVALENT within ±{margin:.4f}"
    else:
        conclusion = f"Cannot conclude equivalence within ±{margin:.4f}"

    return IntervalTestResult(
        effect=effect,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        hypothesis_type=HypothesisType.EQUIVALENCE,
        margin=margin,
        alpha=alpha,
        test_statistic=t_stat,
        p_value=p_value,
        conclusion=conclusion,
        is_significant=is_significant
    )


def superiority_test_proportions(
    control_successes: int,
    control_total: int,
    treatment_successes: int,
    treatment_total: int,
    margin: float,
    alpha: float = 0.05,
) -> IntervalTestResult:
    """
    Superiority test for proportions (e.g., conversion rates).

    Parameters
    ----------
    control_successes : int
        Number of successes in control group
    control_total : int
        Total observations in control group
    treatment_successes : int
        Number of successes in treatment group
    treatment_total : int
        Total observations in treatment group
    margin : float
        Superiority margin δ (as proportion, e.g., 0.02 for 2%)
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    IntervalTestResult
        Test results
    """
    p_c = control_successes / control_total
    p_t = treatment_successes / treatment_total

    effect = p_t - p_c
    se = np.sqrt(p_c * (1 - p_c) / control_total + p_t * (1 - p_t) / treatment_total)

    # Z-test for proportions
    z_stat = (effect - margin) / se
    p_value = 1 - stats.norm.cdf(z_stat)

    # Confidence interval
    z_crit = stats.norm.ppf(1 - alpha/2)
    ci_lower = effect - z_crit * se
    ci_upper = effect + z_crit * se

    is_significant = p_value < alpha

    if is_significant:
        conclusion = f"Treatment rate is SUPERIOR by at least {margin:.4f}"
    else:
        conclusion = f"Cannot conclude treatment rate is superior by {margin:.4f}"

    return IntervalTestResult(
        effect=effect,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        hypothesis_type=HypothesisType.SUPERIORITY,
        margin=margin,
        alpha=alpha,
        test_statistic=z_stat,
        p_value=p_value,
        conclusion=conclusion,
        is_significant=is_significant
    )


def non_inferiority_test_proportions(
    control_successes: int,
    control_total: int,
    treatment_successes: int,
    treatment_total: int,
    margin: float,
    alpha: float = 0.05,
) -> IntervalTestResult:
    """
    Non-inferiority test for proportions.

    Parameters
    ----------
    control_successes : int
        Number of successes in control group
    control_total : int
        Total observations in control group
    treatment_successes : int
        Number of successes in treatment group
    treatment_total : int
        Total observations in treatment group
    margin : float
        Non-inferiority margin δ (as proportion)
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    IntervalTestResult
        Test results
    """
    p_c = control_successes / control_total
    p_t = treatment_successes / treatment_total

    effect = p_t - p_c
    se = np.sqrt(p_c * (1 - p_c) / control_total + p_t * (1 - p_t) / treatment_total)

    z_stat = (effect + margin) / se
    p_value = 1 - stats.norm.cdf(z_stat)

    z_crit = stats.norm.ppf(1 - alpha/2)
    ci_lower = effect - z_crit * se
    ci_upper = effect + z_crit * se

    is_significant = p_value < alpha

    if is_significant:
        conclusion = f"Treatment rate is NON-INFERIOR (not worse by more than {margin:.4f})"
    else:
        conclusion = f"Cannot conclude non-inferiority (margin = {margin:.4f})"

    return IntervalTestResult(
        effect=effect,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        hypothesis_type=HypothesisType.NON_INFERIORITY,
        margin=margin,
        alpha=alpha,
        test_statistic=z_stat,
        p_value=p_value,
        conclusion=conclusion,
        is_significant=is_significant
    )


def equivalence_test_proportions(
    control_successes: int,
    control_total: int,
    treatment_successes: int,
    treatment_total: int,
    margin: float,
    alpha: float = 0.05,
) -> IntervalTestResult:
    """
    TOST equivalence test for proportions.

    Parameters
    ----------
    control_successes : int
        Number of successes in control group
    control_total : int
        Total observations in control group
    treatment_successes : int
        Number of successes in treatment group
    treatment_total : int
        Total observations in treatment group
    margin : float
        Equivalence margin δ (as proportion)
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    IntervalTestResult
        Test results
    """
    p_c = control_successes / control_total
    p_t = treatment_successes / treatment_total

    effect = p_t - p_c
    se = np.sqrt(p_c * (1 - p_c) / control_total + p_t * (1 - p_t) / treatment_total)

    # TOST
    z_lower = (effect + margin) / se
    p_lower = 1 - stats.norm.cdf(z_lower)

    z_upper = (effect - margin) / se
    p_upper = stats.norm.cdf(z_upper)

    p_value = max(p_lower, p_upper)
    z_stat = z_lower if p_lower >= p_upper else z_upper

    z_crit = stats.norm.ppf(1 - alpha)
    ci_lower = effect - z_crit * se
    ci_upper = effect + z_crit * se

    is_significant = p_value < alpha

    if is_significant:
        conclusion = f"Rates are EQUIVALENT within ±{margin:.4f}"
    else:
        conclusion = f"Cannot conclude equivalence within ±{margin:.4f}"

    return IntervalTestResult(
        effect=effect,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        hypothesis_type=HypothesisType.EQUIVALENCE,
        margin=margin,
        alpha=alpha,
        test_statistic=z_stat,
        p_value=p_value,
        conclusion=conclusion,
        is_significant=is_significant
    )


def plot_interval_test(
    result: IntervalTestResult,
    figsize: Tuple[int, int] = (10, 4),
) -> plt.Figure:
    """
    Visualize interval-based hypothesis test results.

    Shows the effect estimate, confidence interval, and margin(s)
    to help interpret the test result.

    Parameters
    ----------
    result : IntervalTestResult
        Results from an interval test
    figsize : tuple, default=(10, 4)
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the effect with CI
    ax.errorbar(
        result.effect, 0,
        xerr=[[result.effect - result.ci_lower], [result.ci_upper - result.effect]],
        fmt='o', markersize=10, capsize=10,
        color='#2ecc71' if result.is_significant else '#e74c3c',
        label=f'Effect: {result.effect:.4f}'
    )

    # Plot margins based on test type
    if result.hypothesis_type == HypothesisType.SUPERIORITY:
        ax.axvline(x=result.margin, color='gray', linestyle='--', alpha=0.7,
                   label=f'Superiority margin (δ = {result.margin:.4f})')
        ax.axvspan(result.margin, ax.get_xlim()[1] if ax.get_xlim()[1] > result.margin else result.margin + abs(result.effect),
                   alpha=0.2, color='green', label='Superiority zone')

    elif result.hypothesis_type == HypothesisType.NON_INFERIORITY:
        ax.axvline(x=-result.margin, color='gray', linestyle='--', alpha=0.7,
                   label=f'Non-inferiority margin (-δ = {-result.margin:.4f})')
        ax.axvspan(-result.margin, ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else abs(result.effect),
                   alpha=0.2, color='green', label='Non-inferiority zone')

    elif result.hypothesis_type == HypothesisType.EQUIVALENCE:
        ax.axvline(x=-result.margin, color='gray', linestyle='--', alpha=0.7,
                   label=f'Equivalence bounds (±{result.margin:.4f})')
        ax.axvline(x=result.margin, color='gray', linestyle='--', alpha=0.7)
        ax.axvspan(-result.margin, result.margin, alpha=0.2, color='green',
                   label='Equivalence zone')

    # Reference line at zero
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)

    ax.set_yticks([])
    ax.set_xlabel('Effect (Treatment - Control)')
    ax.set_title(f'{result.hypothesis_type.value.replace("_", " ").title()} Test\n'
                 f'{"PASSED" if result.is_significant else "FAILED"} (p = {result.p_value:.4f})')
    ax.legend(loc='upper right')

    plt.tight_layout()
    return fig


def sample_size_superiority(
    baseline_mean: float,
    baseline_std: float,
    margin: float,
    expected_effect: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """
    Calculate sample size for a superiority test.

    Parameters
    ----------
    baseline_mean : float
        Expected mean for control group
    baseline_std : float
        Expected standard deviation
    margin : float
        Superiority margin δ
    expected_effect : float
        Expected true effect (must be > margin for test to succeed)
    alpha : float, default=0.05
        Significance level
    power : float, default=0.8
        Statistical power

    Returns
    -------
    int
        Required sample size per group
    """
    z_alpha = stats.norm.ppf(1 - alpha)  # One-sided
    z_beta = stats.norm.ppf(power)

    # Effect size relative to margin
    delta = expected_effect - margin

    if delta <= 0:
        raise ValueError("expected_effect must be greater than margin")

    n = 2 * ((z_alpha + z_beta) * baseline_std / delta) ** 2

    return int(np.ceil(n))


def sample_size_non_inferiority(
    baseline_mean: float,
    baseline_std: float,
    margin: float,
    expected_effect: float = 0,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """
    Calculate sample size for a non-inferiority test.

    Parameters
    ----------
    baseline_mean : float
        Expected mean for control group
    baseline_std : float
        Expected standard deviation
    margin : float
        Non-inferiority margin δ
    expected_effect : float, default=0
        Expected true effect (typically 0 for non-inferiority)
    alpha : float, default=0.05
        Significance level
    power : float, default=0.8
        Statistical power

    Returns
    -------
    int
        Required sample size per group
    """
    z_alpha = stats.norm.ppf(1 - alpha)  # One-sided
    z_beta = stats.norm.ppf(power)

    # The "effect" for non-inferiority is expected_effect - (-margin) = expected_effect + margin
    delta = expected_effect + margin

    n = 2 * ((z_alpha + z_beta) * baseline_std / delta) ** 2

    return int(np.ceil(n))


def sample_size_equivalence(
    baseline_mean: float,
    baseline_std: float,
    margin: float,
    expected_effect: float = 0,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """
    Calculate sample size for an equivalence (TOST) test.

    Parameters
    ----------
    baseline_mean : float
        Expected mean for control group
    baseline_std : float
        Expected standard deviation
    margin : float
        Equivalence margin δ
    expected_effect : float, default=0
        Expected true effect (typically 0 for equivalence)
    alpha : float, default=0.05
        Significance level (for each one-sided test)
    power : float, default=0.8
        Statistical power

    Returns
    -------
    int
        Required sample size per group

    Notes
    -----
    Equivalence tests typically require larger sample sizes than
    superiority or non-inferiority tests because both bounds must
    be rejected.
    """
    z_alpha = stats.norm.ppf(1 - alpha)  # One-sided for each test
    z_beta = stats.norm.ppf(power)

    # For equivalence, the "worst case" effect is when expected_effect
    # is at the edge of the margin
    delta = margin - abs(expected_effect)

    if delta <= 0:
        raise ValueError("expected_effect must be within the equivalence margin")

    n = 2 * ((z_alpha + z_beta) * baseline_std / delta) ** 2

    return int(np.ceil(n))
