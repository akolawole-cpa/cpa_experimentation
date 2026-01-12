"""
Multiple testing corrections for controlling error rates.

This module provides methods for:
- Family-wise error rate (FWER) control: Bonferroni, Holm-Bonferroni
- False discovery rate (FDR) control: Benjamini-Hochberg, Benjamini-Yekutieli
- Adjusted p-value calculations

When running multiple hypothesis tests, the probability of at least one
false positive increases. These methods adjust for this inflation.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Union, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class MultipleTestResult:
    """Results from multiple testing correction."""

    original_p_values: np.ndarray
    adjusted_p_values: np.ndarray
    rejected: np.ndarray  # Boolean array of rejected hypotheses
    method: str
    alpha: float
    n_rejected: int
    n_tests: int

    def __repr__(self) -> str:
        return (
            f"MultipleTestResult({self.method})\n"
            f"  Tests: {self.n_tests}\n"
            f"  Rejected: {self.n_rejected}\n"
            f"  Alpha: {self.alpha}\n"
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for easy viewing."""
        return pd.DataFrame({
            'original_p': self.original_p_values,
            'adjusted_p': self.adjusted_p_values,
            'rejected': self.rejected
        })


# =============================================================================
# Family-Wise Error Rate (FWER) Control
# =============================================================================

def bonferroni(
    p_values: Union[List[float], np.ndarray],
    alpha: float = 0.05,
) -> MultipleTestResult:
    """
    Apply Bonferroni correction.

    The simplest and most conservative method for FWER control.
    Divides alpha by the number of tests.

    Parameters
    ----------
    p_values : array-like
        Original p-values from multiple tests
    alpha : float, default=0.05
        Family-wise error rate to control

    Returns
    -------
    MultipleTestResult
        Corrected results

    Notes
    -----
    - Controls FWER at level alpha
    - Very conservative (low power), especially with many tests
    - Adjusted p-value = min(1, p * n)

    Example
    -------
    >>> p_values = [0.01, 0.04, 0.03, 0.20, 0.001]
    >>> result = bonferroni(p_values, alpha=0.05)
    >>> print(result.rejected)  # [False, False, False, False, True]
    """
    p_values = np.asarray(p_values)
    n = len(p_values)

    # Bonferroni adjusted p-values
    adjusted = np.minimum(p_values * n, 1.0)

    # Rejection decisions
    rejected = adjusted < alpha

    return MultipleTestResult(
        original_p_values=p_values,
        adjusted_p_values=adjusted,
        rejected=rejected,
        method='bonferroni',
        alpha=alpha,
        n_rejected=int(np.sum(rejected)),
        n_tests=n
    )


def holm_bonferroni(
    p_values: Union[List[float], np.ndarray],
    alpha: float = 0.05,
) -> MultipleTestResult:
    """
    Apply Holm-Bonferroni step-down correction.

    A uniformly more powerful alternative to Bonferroni that still
    controls FWER. Starts with the smallest p-value and stops when
    a hypothesis is not rejected.

    Parameters
    ----------
    p_values : array-like
        Original p-values from multiple tests
    alpha : float, default=0.05
        Family-wise error rate to control

    Returns
    -------
    MultipleTestResult
        Corrected results

    Notes
    -----
    - Controls FWER at level alpha
    - More powerful than Bonferroni
    - Step-down procedure: reject H(1), H(2), ... until one is not rejected

    Algorithm:
    1. Order p-values: p(1) <= p(2) <= ... <= p(n)
    2. Compare p(i) to alpha / (n - i + 1)
    3. Reject H(1) through H(k-1) where k is first index where p(k) > alpha/(n-k+1)
    """
    p_values = np.asarray(p_values)
    n = len(p_values)

    # Sort p-values and keep track of original order
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # Calculate adjusted p-values
    adjusted_sorted = np.zeros(n)
    for i in range(n):
        adjusted_sorted[i] = sorted_p[i] * (n - i)

    # Enforce monotonicity (cumulative maximum)
    for i in range(1, n):
        adjusted_sorted[i] = max(adjusted_sorted[i], adjusted_sorted[i-1])

    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    # Unsort to original order
    adjusted = np.zeros(n)
    adjusted[sorted_indices] = adjusted_sorted

    rejected = adjusted < alpha

    return MultipleTestResult(
        original_p_values=p_values,
        adjusted_p_values=adjusted,
        rejected=rejected,
        method='holm_bonferroni',
        alpha=alpha,
        n_rejected=int(np.sum(rejected)),
        n_tests=n
    )


def sidak(
    p_values: Union[List[float], np.ndarray],
    alpha: float = 0.05,
) -> MultipleTestResult:
    """
    Apply Sidak correction.

    Similar to Bonferroni but slightly less conservative.
    Uses 1 - (1 - alpha)^(1/n) as the per-test alpha.

    Parameters
    ----------
    p_values : array-like
        Original p-values from multiple tests
    alpha : float, default=0.05
        Family-wise error rate to control

    Returns
    -------
    MultipleTestResult
        Corrected results

    Notes
    -----
    - Controls FWER at level alpha
    - Slightly more powerful than Bonferroni
    - Assumes independence of tests (Bonferroni doesn't)
    """
    p_values = np.asarray(p_values)
    n = len(p_values)

    # Sidak adjusted p-values
    adjusted = 1 - (1 - p_values) ** n
    adjusted = np.minimum(adjusted, 1.0)

    rejected = adjusted < alpha

    return MultipleTestResult(
        original_p_values=p_values,
        adjusted_p_values=adjusted,
        rejected=rejected,
        method='sidak',
        alpha=alpha,
        n_rejected=int(np.sum(rejected)),
        n_tests=n
    )


# =============================================================================
# False Discovery Rate (FDR) Control
# =============================================================================

def benjamini_hochberg(
    p_values: Union[List[float], np.ndarray],
    alpha: float = 0.05,
) -> MultipleTestResult:
    """
    Apply Benjamini-Hochberg procedure for FDR control.

    Controls the expected proportion of false discoveries among
    rejected hypotheses, rather than the probability of any false positive.

    Parameters
    ----------
    p_values : array-like
        Original p-values from multiple tests
    alpha : float, default=0.05
        False discovery rate to control

    Returns
    -------
    MultipleTestResult
        Corrected results

    Notes
    -----
    - Controls FDR at level alpha (under independence or positive dependence)
    - More powerful than FWER methods when many tests are run
    - Step-up procedure

    Algorithm:
    1. Order p-values: p(1) <= p(2) <= ... <= p(n)
    2. Find largest k where p(k) <= k * alpha / n
    3. Reject H(1) through H(k)

    Example
    -------
    >>> p_values = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.07, 0.08]
    >>> result = benjamini_hochberg(p_values, alpha=0.05)
    >>> print(result.rejected)
    """
    p_values = np.asarray(p_values)
    n = len(p_values)

    # Sort p-values and keep track of original order
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # Calculate adjusted p-values (BH step-up)
    adjusted_sorted = np.zeros(n)
    for i in range(n):
        adjusted_sorted[i] = sorted_p[i] * n / (i + 1)

    # Enforce monotonicity (cumulative minimum from right)
    for i in range(n - 2, -1, -1):
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])

    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    # Unsort to original order
    adjusted = np.zeros(n)
    adjusted[sorted_indices] = adjusted_sorted

    rejected = adjusted < alpha

    return MultipleTestResult(
        original_p_values=p_values,
        adjusted_p_values=adjusted,
        rejected=rejected,
        method='benjamini_hochberg',
        alpha=alpha,
        n_rejected=int(np.sum(rejected)),
        n_tests=n
    )


def benjamini_yekutieli(
    p_values: Union[List[float], np.ndarray],
    alpha: float = 0.05,
) -> MultipleTestResult:
    """
    Apply Benjamini-Yekutieli procedure for FDR control.

    Controls FDR under arbitrary dependence between tests.
    More conservative than Benjamini-Hochberg.

    Parameters
    ----------
    p_values : array-like
        Original p-values from multiple tests
    alpha : float, default=0.05
        False discovery rate to control

    Returns
    -------
    MultipleTestResult
        Corrected results

    Notes
    -----
    - Controls FDR under any dependence structure
    - Uses a correction factor: sum(1/i) for i = 1 to n
    """
    p_values = np.asarray(p_values)
    n = len(p_values)

    # BY correction factor
    c_n = np.sum(1.0 / np.arange(1, n + 1))

    # Sort p-values
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # Calculate adjusted p-values
    adjusted_sorted = np.zeros(n)
    for i in range(n):
        adjusted_sorted[i] = sorted_p[i] * n * c_n / (i + 1)

    # Enforce monotonicity
    for i in range(n - 2, -1, -1):
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])

    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    # Unsort
    adjusted = np.zeros(n)
    adjusted[sorted_indices] = adjusted_sorted

    rejected = adjusted < alpha

    return MultipleTestResult(
        original_p_values=p_values,
        adjusted_p_values=adjusted,
        rejected=rejected,
        method='benjamini_yekutieli',
        alpha=alpha,
        n_rejected=int(np.sum(rejected)),
        n_tests=n
    )


# =============================================================================
# Helper Functions
# =============================================================================

def fwer_at_k(
    n_tests: int,
    alpha_per_test: float,
) -> float:
    """
    Calculate family-wise error rate given per-test alpha.

    Parameters
    ----------
    n_tests : int
        Number of independent tests
    alpha_per_test : float
        Significance level for each test

    Returns
    -------
    float
        Family-wise error rate

    Example
    -------
    >>> fwer = fwer_at_k(20, 0.05)
    >>> print(f"FWER: {fwer:.2%}")  # ~64%
    """
    return 1 - (1 - alpha_per_test) ** n_tests


def alpha_for_fwer(
    n_tests: int,
    fwer: float = 0.05,
    method: str = 'sidak',
) -> float:
    """
    Calculate per-test alpha to achieve desired FWER.

    Parameters
    ----------
    n_tests : int
        Number of tests
    fwer : float, default=0.05
        Desired family-wise error rate
    method : str, default='sidak'
        'sidak' or 'bonferroni'

    Returns
    -------
    float
        Per-test alpha level
    """
    if method == 'sidak':
        return 1 - (1 - fwer) ** (1 / n_tests)
    else:  # bonferroni
        return fwer / n_tests


def expected_false_discoveries(
    p_values: Union[List[float], np.ndarray],
    alpha: float = 0.05,
) -> float:
    """
    Estimate expected number of false discoveries under null.

    Under the global null (all null hypotheses true),
    the expected number of false discoveries is n * alpha.

    Parameters
    ----------
    p_values : array-like
        P-values from tests
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    float
        Expected false discoveries if all nulls are true
    """
    return len(p_values) * alpha


def compare_methods(
    p_values: Union[List[float], np.ndarray],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compare different multiple testing correction methods.

    Parameters
    ----------
    p_values : array-like
        Original p-values
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    pd.DataFrame
        Comparison of methods showing adjusted p-values and rejections
    """
    methods = {
        'Original': lambda p, a: MultipleTestResult(
            p, p, p < a, 'original', a, int(np.sum(p < a)), len(p)
        ),
        'Bonferroni': bonferroni,
        'Holm': holm_bonferroni,
        'Sidak': sidak,
        'BH (FDR)': benjamini_hochberg,
        'BY (FDR)': benjamini_yekutieli,
    }

    p_values = np.asarray(p_values)
    results = {}

    for name, func in methods.items():
        result = func(p_values, alpha)
        results[f'{name} adj_p'] = result.adjusted_p_values
        results[f'{name} rejected'] = result.rejected

    df = pd.DataFrame(results)
    df.index.name = 'Test'

    return df


def plot_adjusted_pvalues(
    p_values: Union[List[float], np.ndarray],
    alpha: float = 0.05,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Visualize original and adjusted p-values across methods.

    Parameters
    ----------
    p_values : array-like
        Original p-values
    alpha : float, default=0.05
        Significance level
    figsize : tuple, default=(12, 6)
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Comparison plot
    """
    p_values = np.asarray(p_values)
    n = len(p_values)

    # Get adjusted p-values for each method
    methods = {
        'Original': p_values,
        'Bonferroni': bonferroni(p_values, alpha).adjusted_p_values,
        'Holm': holm_bonferroni(p_values, alpha).adjusted_p_values,
        'BH (FDR)': benjamini_hochberg(p_values, alpha).adjusted_p_values,
    }

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n)
    width = 0.2
    offsets = np.linspace(-1.5*width, 1.5*width, len(methods))

    for i, (name, adj_p) in enumerate(methods.items()):
        # Sort by original p-value for cleaner visualization
        sort_idx = np.argsort(p_values)
        bars = ax.bar(x + offsets[i], adj_p[sort_idx], width, label=name, alpha=0.7)

    ax.axhline(y=alpha, color='r', linestyle='--', label=f'α = {alpha}')
    ax.set_xlabel('Test (sorted by original p-value)')
    ax.set_ylabel('P-value')
    ax.set_title('Comparison of Multiple Testing Corrections')
    ax.legend()
    ax.set_yscale('log')
    ax.set_ylim(0.0001, 1.5)

    plt.tight_layout()
    return fig


# =============================================================================
# Simulation and Validation
# =============================================================================

def simulate_fwer(
    n_tests: int,
    alpha: float = 0.05,
    method: str = 'bonferroni',
    n_simulations: int = 10000,
    seed: Optional[int] = None,
) -> float:
    """
    Simulate family-wise error rate under the global null.

    Parameters
    ----------
    n_tests : int
        Number of tests per simulation
    alpha : float, default=0.05
        Significance level
    method : str, default='bonferroni'
        Correction method to use
    n_simulations : int, default=10000
        Number of simulations
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    float
        Simulated FWER
    """
    if seed is not None:
        np.random.seed(seed)

    methods = {
        'bonferroni': bonferroni,
        'holm': holm_bonferroni,
        'sidak': sidak,
        'bh': benjamini_hochberg,
        'by': benjamini_yekutieli,
        'none': lambda p, a: MultipleTestResult(
            p, p, p < a, 'none', a, int(np.sum(p < a)), len(p)
        ),
    }

    correction = methods[method]
    false_positives = 0

    for _ in range(n_simulations):
        # Generate p-values under null (uniform distribution)
        p_values = np.random.uniform(0, 1, n_tests)
        result = correction(p_values, alpha)

        # FWER: at least one false positive
        if result.n_rejected > 0:
            false_positives += 1

    return false_positives / n_simulations


def simulate_fdr(
    n_true_null: int,
    n_true_alt: int,
    effect_size: float = 2.0,
    alpha: float = 0.05,
    method: str = 'bh',
    n_simulations: int = 1000,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Simulate false discovery rate and power.

    Parameters
    ----------
    n_true_null : int
        Number of true null hypotheses
    n_true_alt : int
        Number of true alternative hypotheses
    effect_size : float, default=2.0
        Effect size for alternatives (in z-score units)
    alpha : float, default=0.05
        Significance level
    method : str, default='bh'
        Correction method
    n_simulations : int, default=1000
        Number of simulations
    seed : int, optional
        Random seed

    Returns
    -------
    tuple
        (average FDR, average power)
    """
    from scipy import stats

    if seed is not None:
        np.random.seed(seed)

    methods = {
        'bonferroni': bonferroni,
        'holm': holm_bonferroni,
        'bh': benjamini_hochberg,
        'by': benjamini_yekutieli,
        'none': lambda p, a: MultipleTestResult(
            p, p, p < a, 'none', a, int(np.sum(p < a)), len(p)
        ),
    }

    correction = methods[method]
    n_total = n_true_null + n_true_alt
    fdrs = []
    powers = []

    for _ in range(n_simulations):
        # Generate test statistics
        z_null = np.random.normal(0, 1, n_true_null)
        z_alt = np.random.normal(effect_size, 1, n_true_alt)
        z_all = np.concatenate([z_null, z_alt])

        # Convert to p-values (two-tailed)
        p_values = 2 * (1 - stats.norm.cdf(np.abs(z_all)))

        # Apply correction
        result = correction(p_values, alpha)

        # Calculate FDR and power
        if result.n_rejected > 0:
            # False discoveries: rejections among true nulls
            false_discoveries = np.sum(result.rejected[:n_true_null])
            fdr = false_discoveries / result.n_rejected
        else:
            fdr = 0

        # Power: proportion of true alternatives detected
        true_discoveries = np.sum(result.rejected[n_true_null:])
        power = true_discoveries / n_true_alt if n_true_alt > 0 else 0

        fdrs.append(fdr)
        powers.append(power)

    return np.mean(fdrs), np.mean(powers)
