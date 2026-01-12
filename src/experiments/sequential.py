"""
Sequential Testing and Early Stopping for A/B Tests.

Sequential testing allows you to peek at results during an experiment
and make statistically valid decisions about early stopping. Without
proper sequential methods, repeated peeking inflates the false positive rate.

Key methods:
1. Alpha spending: Control overall alpha across multiple looks
2. O'Brien-Fleming: Conservative early stopping boundaries
3. Pocock: Less conservative, equal boundaries at each look
4. Group Sequential: Discrete look times with formal stopping rules

Use these methods when:
- You want to stop early if a clear winner emerges
- You have ethical/cost reasons to minimize exposure to inferior treatments
- You need to peek at results before the experiment ends
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class SequentialTestResult:
    """Results from sequential test monitoring."""

    current_look: int
    total_looks: int
    current_z_stat: float
    current_p_value: float

    # Boundaries
    upper_boundary: float  # Reject H0 for benefit
    lower_boundary: float  # Reject H0 for harm (futility)

    # Decisions
    reject_h0: bool
    stop_for_benefit: bool
    stop_for_futility: bool
    continue_testing: bool

    # Spending
    alpha_spent: float
    alpha_remaining: float

    # Effect estimates
    current_effect: float
    current_se: float

    def __repr__(self) -> str:
        status = "STOP" if not self.continue_testing else "CONTINUE"
        if self.stop_for_benefit:
            status = "STOP (Benefit)"
        elif self.stop_for_futility:
            status = "STOP (Futility)"

        return (
            f"SequentialTestResult (Look {self.current_look}/{self.total_looks}): {status}\n"
            f"  Z-statistic: {self.current_z_stat:.3f}\n"
            f"  Upper boundary: {self.upper_boundary:.3f}\n"
            f"  Lower boundary: {self.lower_boundary:.3f}\n"
            f"  Alpha spent: {self.alpha_spent:.4f}, remaining: {self.alpha_remaining:.4f}"
        )


# =============================================================================
# Alpha Spending Functions
# =============================================================================

def obrien_fleming_spending(
    alpha: float,
    t: float,
) -> float:
    """
    O'Brien-Fleming alpha spending function.

    Very conservative early: spends little alpha in early looks,
    most of the alpha at the final look.

    Parameters
    ----------
    alpha : float
        Total alpha to spend
    t : float
        Information fraction (0 to 1)

    Returns
    -------
    float
        Cumulative alpha spent at time t

    Notes
    -----
    α*(t) = 2 - 2Φ(z_{α/2} / √t)
    """
    if t <= 0:
        return 0.0
    if t >= 1:
        return alpha

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    return 2 * (1 - stats.norm.cdf(z_alpha / np.sqrt(t)))


def pocock_spending(
    alpha: float,
    t: float,
) -> float:
    """
    Pocock alpha spending function.

    Spends alpha more evenly across looks.

    Parameters
    ----------
    alpha : float
        Total alpha to spend
    t : float
        Information fraction (0 to 1)

    Returns
    -------
    float
        Cumulative alpha spent at time t

    Notes
    -----
    α*(t) = α * log(1 + (e-1)*t)
    """
    if t <= 0:
        return 0.0
    if t >= 1:
        return alpha

    return alpha * np.log(1 + (np.e - 1) * t)


def hwang_shih_decani_spending(
    alpha: float,
    t: float,
    gamma: float = -4,
) -> float:
    """
    Hwang-Shih-DeCani (power family) alpha spending function.

    Parameters
    ----------
    alpha : float
        Total alpha to spend
    t : float
        Information fraction (0 to 1)
    gamma : float
        Shape parameter:
        - gamma = -4: Similar to O'Brien-Fleming (very conservative)
        - gamma = 0: Linear spending
        - gamma = 1: Similar to Pocock (less conservative)

    Returns
    -------
    float
        Cumulative alpha spent at time t
    """
    if t <= 0:
        return 0.0
    if t >= 1:
        return alpha

    if gamma == 0:
        return alpha * t
    else:
        return alpha * (1 - np.exp(-gamma * t)) / (1 - np.exp(-gamma))


def lan_demets_spending(
    alpha: float,
    t: float,
    spending_type: str = 'obf',
) -> float:
    """
    Lan-DeMets alpha spending function (flexible spending).

    Parameters
    ----------
    alpha : float
        Total alpha to spend
    t : float
        Information fraction
    spending_type : str
        'obf' for O'Brien-Fleming like
        'pocock' for Pocock like

    Returns
    -------
    float
        Cumulative alpha spent
    """
    if spending_type == 'obf':
        return obrien_fleming_spending(alpha, t)
    elif spending_type == 'pocock':
        return pocock_spending(alpha, t)
    else:
        raise ValueError(f"Unknown spending type: {spending_type}")


# =============================================================================
# Boundary Calculations
# =============================================================================

def calculate_boundary(
    alpha_increment: float,
    two_sided: bool = True,
) -> float:
    """
    Calculate z-score boundary for given alpha increment.

    Parameters
    ----------
    alpha_increment : float
        Alpha to spend at this look
    two_sided : bool
        Whether test is two-sided

    Returns
    -------
    float
        Z-score boundary
    """
    if two_sided:
        return stats.norm.ppf(1 - alpha_increment / 2)
    else:
        return stats.norm.ppf(1 - alpha_increment)


def calculate_sequential_boundaries(
    alpha: float,
    n_looks: int,
    spending_function: str = 'obf',
    two_sided: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate stopping boundaries for all planned looks.

    Parameters
    ----------
    alpha : float
        Total alpha
    n_looks : int
        Number of planned looks
    spending_function : str
        'obf', 'pocock', or 'linear'
    two_sided : bool
        Whether test is two-sided

    Returns
    -------
    tuple
        (information_fractions, boundaries)
    """
    spending_funcs = {
        'obf': obrien_fleming_spending,
        'pocock': pocock_spending,
        'linear': lambda a, t: a * t,
    }

    spend_func = spending_funcs.get(spending_function, obrien_fleming_spending)

    info_fractions = np.linspace(1/n_looks, 1, n_looks)
    boundaries = []
    prev_alpha = 0

    for t in info_fractions:
        cum_alpha = spend_func(alpha, t)
        alpha_increment = cum_alpha - prev_alpha

        # Approximate boundary
        boundary = calculate_boundary(alpha_increment, two_sided)
        boundaries.append(boundary)
        prev_alpha = cum_alpha

    return info_fractions, np.array(boundaries)


# =============================================================================
# Sequential Test Classes
# =============================================================================

class GroupSequentialTest:
    """
    Group Sequential Test for A/B experiments.

    Allows multiple interim analyses with proper alpha spending
    to control overall Type I error rate.

    Parameters
    ----------
    alpha : float
        Overall significance level
    n_looks : int
        Number of planned interim looks + final look
    spending_function : str
        'obf' (O'Brien-Fleming), 'pocock', or 'linear'
    two_sided : bool
        Whether to use two-sided test

    Example
    -------
    >>> gst = GroupSequentialTest(alpha=0.05, n_looks=4)
    >>> # At each interim analysis:
    >>> result = gst.analyze(z_stat=2.5, current_look=1)
    >>> if result.stop_for_benefit:
    ...     print("Stop early - significant benefit detected!")
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_looks: int = 4,
        spending_function: str = 'obf',
        two_sided: bool = True,
    ):
        self.alpha = alpha
        self.n_looks = n_looks
        self.spending_function = spending_function
        self.two_sided = two_sided

        # Pre-compute boundaries
        self.info_fractions, self.boundaries = calculate_sequential_boundaries(
            alpha, n_looks, spending_function, two_sided
        )

        self._current_look = 0
        self._alpha_spent = 0

    def get_boundary(self, look: int) -> float:
        """Get the stopping boundary for a specific look."""
        if look < 1 or look > self.n_looks:
            raise ValueError(f"Look must be between 1 and {self.n_looks}")
        return self.boundaries[look - 1]

    def analyze(
        self,
        z_stat: float,
        current_look: int,
        effect: Optional[float] = None,
        se: Optional[float] = None,
    ) -> SequentialTestResult:
        """
        Analyze results at an interim look.

        Parameters
        ----------
        z_stat : float
            Current z-statistic
        current_look : int
            Current look number (1-indexed)
        effect : float, optional
            Current effect estimate
        se : float, optional
            Standard error of effect

        Returns
        -------
        SequentialTestResult
            Analysis result with stopping decision
        """
        if current_look < 1 or current_look > self.n_looks:
            raise ValueError(f"Look must be between 1 and {self.n_looks}")

        t = self.info_fractions[current_look - 1]
        boundary = self.boundaries[current_look - 1]

        # Calculate alpha spent
        spending_funcs = {
            'obf': obrien_fleming_spending,
            'pocock': pocock_spending,
            'linear': lambda a, t: a * t,
        }
        spend_func = spending_funcs[self.spending_function]
        alpha_spent = spend_func(self.alpha, t)

        # P-value at this look
        if self.two_sided:
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            p_value = 1 - stats.norm.cdf(z_stat)

        # Decision
        if self.two_sided:
            stop_for_benefit = abs(z_stat) >= boundary
        else:
            stop_for_benefit = z_stat >= boundary

        # Futility (optional - simplified version)
        stop_for_futility = False
        if current_look < self.n_looks:
            # Basic futility: if effect is opposite direction
            if effect is not None and effect < 0:
                # Could add more sophisticated futility rules here
                pass

        continue_testing = not stop_for_benefit and not stop_for_futility
        reject_h0 = stop_for_benefit

        return SequentialTestResult(
            current_look=current_look,
            total_looks=self.n_looks,
            current_z_stat=z_stat,
            current_p_value=p_value,
            upper_boundary=boundary,
            lower_boundary=-boundary if self.two_sided else float('-inf'),
            reject_h0=reject_h0,
            stop_for_benefit=stop_for_benefit,
            stop_for_futility=stop_for_futility,
            continue_testing=continue_testing,
            alpha_spent=alpha_spent,
            alpha_remaining=self.alpha - alpha_spent,
            current_effect=effect if effect is not None else z_stat,
            current_se=se if se is not None else 1.0,
        )

    def analyze_proportions(
        self,
        successes_a: int,
        total_a: int,
        successes_b: int,
        total_b: int,
        current_look: int,
    ) -> SequentialTestResult:
        """
        Analyze sequential test for proportions.

        Parameters
        ----------
        successes_a : int
            Successes in control
        total_a : int
            Total in control
        successes_b : int
            Successes in treatment
        total_b : int
            Total in treatment
        current_look : int
            Current look number

        Returns
        -------
        SequentialTestResult
        """
        p_a = successes_a / total_a
        p_b = successes_b / total_b

        pooled_p = (successes_a + successes_b) / (total_a + total_b)
        se = np.sqrt(pooled_p * (1 - pooled_p) * (1/total_a + 1/total_b))

        z_stat = (p_b - p_a) / se

        return self.analyze(z_stat, current_look, effect=p_b - p_a, se=se)

    def analyze_continuous(
        self,
        mean_a: float,
        std_a: float,
        n_a: int,
        mean_b: float,
        std_b: float,
        n_b: int,
        current_look: int,
    ) -> SequentialTestResult:
        """
        Analyze sequential test for continuous metrics.

        Parameters
        ----------
        mean_a, std_a, n_a : float, float, int
            Control group statistics
        mean_b, std_b, n_b : float, float, int
            Treatment group statistics
        current_look : int
            Current look number

        Returns
        -------
        SequentialTestResult
        """
        se = np.sqrt(std_a**2/n_a + std_b**2/n_b)
        z_stat = (mean_b - mean_a) / se

        return self.analyze(z_stat, current_look, effect=mean_b - mean_a, se=se)

    def plot_boundaries(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot stopping boundaries across looks.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        looks = np.arange(1, self.n_looks + 1)

        ax.plot(looks, self.boundaries, 'b-o', linewidth=2, markersize=8,
                label='Upper boundary')
        if self.two_sided:
            ax.plot(looks, -self.boundaries, 'b-o', linewidth=2, markersize=8,
                    label='Lower boundary')

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # Reference lines
        z_alpha = stats.norm.ppf(1 - self.alpha/2) if self.two_sided else stats.norm.ppf(1 - self.alpha)
        ax.axhline(y=z_alpha, color='red', linestyle=':', alpha=0.5,
                   label=f'Fixed test boundary (z={z_alpha:.2f})')

        ax.fill_between(looks, self.boundaries,
                        [max(self.boundaries) + 0.5] * len(looks),
                        alpha=0.2, color='green', label='Rejection region')

        ax.set_xlabel('Look Number')
        ax.set_ylabel('Z-statistic Boundary')
        ax.set_title(f'Group Sequential Boundaries ({self.spending_function.upper()})')
        ax.set_xticks(looks)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


# =============================================================================
# Convenience Functions
# =============================================================================

def run_sequential_analysis(
    z_stats: List[float],
    alpha: float = 0.05,
    spending_function: str = 'obf',
    two_sided: bool = True,
) -> List[SequentialTestResult]:
    """
    Run sequential analysis across multiple looks.

    Parameters
    ----------
    z_stats : list of float
        Z-statistics at each look
    alpha : float
        Overall significance level
    spending_function : str
        Spending function type
    two_sided : bool
        Whether two-sided test

    Returns
    -------
    list of SequentialTestResult
        Results for each look
    """
    n_looks = len(z_stats)
    gst = GroupSequentialTest(alpha, n_looks, spending_function, two_sided)

    results = []
    for i, z in enumerate(z_stats):
        result = gst.analyze(z, i + 1)
        results.append(result)

        if not result.continue_testing:
            break

    return results


def calculate_adjusted_ci(
    effect: float,
    se: float,
    alpha: float,
    n_looks: int,
    current_look: int,
    spending_function: str = 'obf',
) -> Tuple[float, float]:
    """
    Calculate adjusted confidence interval for sequential test.

    Parameters
    ----------
    effect : float
        Point estimate
    se : float
        Standard error
    alpha : float
        Significance level
    n_looks : int
        Total planned looks
    current_look : int
        Current look
    spending_function : str
        Spending function

    Returns
    -------
    tuple
        (ci_lower, ci_upper)
    """
    _, boundaries = calculate_sequential_boundaries(
        alpha, n_looks, spending_function, True
    )

    # Use the boundary at current look for CI
    z_crit = boundaries[current_look - 1]

    return (effect - z_crit * se, effect + z_crit * se)


def simulate_sequential_test(
    true_effect: float,
    se: float,
    alpha: float = 0.05,
    n_looks: int = 4,
    spending_function: str = 'obf',
    n_simulations: int = 10000,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Simulate sequential test to estimate operating characteristics.

    Parameters
    ----------
    true_effect : float
        True effect size (0 for null)
    se : float
        Standard error (assumed constant across looks)
    alpha : float
        Significance level
    n_looks : int
        Number of looks
    spending_function : str
        Spending function
    n_simulations : int
        Number of simulations
    seed : int, optional
        Random seed

    Returns
    -------
    dict
        Operating characteristics:
        - 'rejection_rate': Proportion of rejections (power if effect>0)
        - 'average_look': Average look at which test stops
        - 'early_stop_rate': Proportion stopping before final look
    """
    if seed is not None:
        np.random.seed(seed)

    gst = GroupSequentialTest(alpha, n_looks, spending_function)

    rejections = 0
    total_looks = 0
    early_stops = 0

    for _ in range(n_simulations):
        stopped = False
        for look in range(1, n_looks + 1):
            # Simulate z-statistic at this look
            # (simplified: assume constant SE)
            z = np.random.normal(true_effect / se, 1)

            result = gst.analyze(z, look)

            if not result.continue_testing:
                if result.stop_for_benefit:
                    rejections += 1
                total_looks += look
                if look < n_looks:
                    early_stops += 1
                stopped = True
                break

        if not stopped:
            total_looks += n_looks

    return {
        'rejection_rate': rejections / n_simulations,
        'average_look': total_looks / n_simulations,
        'early_stop_rate': early_stops / n_simulations,
    }
