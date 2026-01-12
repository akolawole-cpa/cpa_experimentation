"""
Adaptive Testing and Multi-Armed Bandits.

Multi-armed bandit (MAB) algorithms balance exploration (learning which
option is best) with exploitation (choosing the current best option).

Key algorithms:
1. Epsilon-Greedy: Simple random exploration
2. Thompson Sampling: Bayesian probability matching
3. Upper Confidence Bound (UCB): Optimism in face of uncertainty

Use MAB when:
- You want to minimize regret during the experiment
- The cost of showing inferior treatments is high
- You can tolerate some statistical complexity

Trade-offs vs A/B testing:
- MAB: Faster convergence to best arm, less precise effect estimates
- A/B: Better statistical inference, longer to find winner
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class BanditArm:
    """Represents one arm (variant) in a bandit."""

    name: str
    successes: int = 0
    failures: int = 0
    total_reward: float = 0.0
    n_pulls: int = 0

    @property
    def conversion_rate(self) -> float:
        """Observed conversion rate."""
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0.0

    @property
    def mean_reward(self) -> float:
        """Mean reward per pull."""
        return self.total_reward / self.n_pulls if self.n_pulls > 0 else 0.0

    def update_binary(self, success: bool) -> None:
        """Update with binary outcome (conversion)."""
        if success:
            self.successes += 1
        else:
            self.failures += 1
        self.n_pulls += 1

    def update_continuous(self, reward: float) -> None:
        """Update with continuous reward."""
        self.total_reward += reward
        self.n_pulls += 1


@dataclass
class BanditResult:
    """Results from bandit experiment."""

    arms: List[BanditArm]
    best_arm: str
    best_arm_probability: float
    total_pulls: int
    total_reward: float
    regret: float
    regret_history: np.ndarray
    arm_selection_history: np.ndarray
    algorithm: str

    def __repr__(self) -> str:
        return (
            f"BanditResult({self.algorithm})\n"
            f"  Best arm: {self.best_arm} (prob: {self.best_arm_probability:.2%})\n"
            f"  Total pulls: {self.total_pulls}\n"
            f"  Total reward: {self.total_reward:.2f}\n"
            f"  Cumulative regret: {self.regret:.2f}"
        )

    def summary(self) -> pd.DataFrame:
        """Get summary DataFrame of all arms."""
        data = []
        for arm in self.arms:
            data.append({
                'arm': arm.name,
                'pulls': arm.n_pulls,
                'pull_rate': arm.n_pulls / self.total_pulls if self.total_pulls > 0 else 0,
                'successes': arm.successes,
                'conversion_rate': arm.conversion_rate,
                'total_reward': arm.total_reward,
                'mean_reward': arm.mean_reward,
            })
        return pd.DataFrame(data)


class MultiArmedBandit:
    """
    Base class for multi-armed bandit algorithms.

    Parameters
    ----------
    arm_names : list of str
        Names for each arm
    """

    def __init__(self, arm_names: List[str]):
        self.arms = [BanditArm(name=name) for name in arm_names]
        self.arm_names = arm_names
        self.n_arms = len(arm_names)
        self._history = []
        self._regret_history = []

    def select_arm(self) -> int:
        """Select which arm to pull. Override in subclasses."""
        raise NotImplementedError

    def update(self, arm_idx: int, reward: float) -> None:
        """Update arm with observed reward."""
        self.arms[arm_idx].update_continuous(reward)
        self._history.append(arm_idx)

    def update_binary(self, arm_idx: int, success: bool) -> None:
        """Update arm with binary outcome."""
        self.arms[arm_idx].update_binary(success)
        self._history.append(arm_idx)

    def get_best_arm(self) -> Tuple[int, str]:
        """Get current best arm based on observed performance."""
        best_idx = np.argmax([arm.mean_reward for arm in self.arms])
        return best_idx, self.arms[best_idx].name


class EpsilonGreedy(MultiArmedBandit):
    """
    Epsilon-Greedy bandit algorithm.

    With probability epsilon, explore randomly.
    Otherwise, exploit current best arm.

    Parameters
    ----------
    arm_names : list of str
        Names for each arm
    epsilon : float
        Exploration probability (0-1)
    decay : float
        Epsilon decay rate per pull (1.0 = no decay)

    Example
    -------
    >>> bandit = EpsilonGreedy(['A', 'B', 'C'], epsilon=0.1)
    >>> for i in range(1000):
    ...     arm = bandit.select_arm()
    ...     reward = simulate_reward(arm)
    ...     bandit.update(arm, reward)
    """

    def __init__(
        self,
        arm_names: List[str],
        epsilon: float = 0.1,
        decay: float = 1.0,
    ):
        super().__init__(arm_names)
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.decay = decay
        self._pulls = 0

    def select_arm(self) -> int:
        """Select arm using epsilon-greedy strategy."""
        self._pulls += 1

        # Decay epsilon
        current_epsilon = self.initial_epsilon * (self.decay ** self._pulls)

        if np.random.random() < current_epsilon:
            # Explore: random arm
            return np.random.randint(self.n_arms)
        else:
            # Exploit: best arm
            return self.get_best_arm()[0]


class ThompsonSampling(MultiArmedBandit):
    """
    Thompson Sampling bandit algorithm.

    Maintains Beta prior for each arm and samples from posterior
    to select arm. Naturally balances exploration and exploitation.

    Parameters
    ----------
    arm_names : list of str
        Names for each arm
    prior_alpha : float
        Prior alpha for Beta distribution
    prior_beta : float
        Prior beta for Beta distribution

    Example
    -------
    >>> bandit = ThompsonSampling(['control', 'variant_a', 'variant_b'])
    >>> for i in range(1000):
    ...     arm = bandit.select_arm()
    ...     success = simulate_conversion(arm)
    ...     bandit.update_binary(arm, success)
    """

    def __init__(
        self,
        arm_names: List[str],
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ):
        super().__init__(arm_names)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def select_arm(self) -> int:
        """Select arm by sampling from Beta posteriors."""
        samples = []
        for arm in self.arms:
            # Posterior: Beta(alpha + successes, beta + failures)
            alpha = self.prior_alpha + arm.successes
            beta = self.prior_beta + arm.failures
            sample = np.random.beta(alpha, beta)
            samples.append(sample)

        return np.argmax(samples)

    def get_arm_probabilities(self, n_samples: int = 10000) -> np.ndarray:
        """
        Estimate probability each arm is best.

        Parameters
        ----------
        n_samples : int
            Number of Monte Carlo samples

        Returns
        -------
        np.ndarray
            Probability each arm is best
        """
        samples = np.zeros((n_samples, self.n_arms))

        for i, arm in enumerate(self.arms):
            alpha = self.prior_alpha + arm.successes
            beta = self.prior_beta + arm.failures
            samples[:, i] = np.random.beta(alpha, beta, n_samples)

        best_arm = np.argmax(samples, axis=1)
        probabilities = np.array([np.mean(best_arm == i) for i in range(self.n_arms)])

        return probabilities


class UCB(MultiArmedBandit):
    """
    Upper Confidence Bound (UCB1) bandit algorithm.

    Selects arm with highest upper confidence bound:
    UCB = mean_reward + c * sqrt(log(t) / n_pulls)

    Parameters
    ----------
    arm_names : list of str
        Names for each arm
    c : float
        Exploration parameter (higher = more exploration)

    Example
    -------
    >>> bandit = UCB(['A', 'B', 'C'], c=2.0)
    >>> for i in range(1000):
    ...     arm = bandit.select_arm()
    ...     reward = simulate_reward(arm)
    ...     bandit.update(arm, reward)
    """

    def __init__(
        self,
        arm_names: List[str],
        c: float = 2.0,
    ):
        super().__init__(arm_names)
        self.c = c
        self._total_pulls = 0

    def select_arm(self) -> int:
        """Select arm with highest UCB."""
        self._total_pulls += 1

        ucb_values = []
        for arm in self.arms:
            if arm.n_pulls == 0:
                # Pull unpulled arms first
                ucb_values.append(float('inf'))
            else:
                # UCB formula
                mean = arm.mean_reward
                exploration = self.c * np.sqrt(np.log(self._total_pulls) / arm.n_pulls)
                ucb_values.append(mean + exploration)

        return np.argmax(ucb_values)


class UCB_Tuned(MultiArmedBandit):
    """
    UCB-Tuned: Improved UCB with variance estimate.

    Uses estimated variance in the confidence bound for
    potentially better performance.

    Parameters
    ----------
    arm_names : list of str
        Names for each arm
    """

    def __init__(self, arm_names: List[str]):
        super().__init__(arm_names)
        self._total_pulls = 0
        self._squared_rewards = [0.0] * len(arm_names)

    def update(self, arm_idx: int, reward: float) -> None:
        """Update arm with observed reward."""
        super().update(arm_idx, reward)
        self._squared_rewards[arm_idx] += reward ** 2

    def select_arm(self) -> int:
        """Select arm with highest UCB-Tuned value."""
        self._total_pulls += 1

        ucb_values = []
        for i, arm in enumerate(self.arms):
            if arm.n_pulls == 0:
                ucb_values.append(float('inf'))
            else:
                mean = arm.mean_reward
                n = arm.n_pulls
                t = self._total_pulls

                # Variance estimate
                variance = self._squared_rewards[i] / n - mean ** 2
                variance = max(0, variance)  # Ensure non-negative

                # UCB-Tuned formula
                V = variance + np.sqrt(2 * np.log(t) / n)
                exploration = np.sqrt(np.log(t) / n * min(0.25, V))
                ucb_values.append(mean + exploration)

        return np.argmax(ucb_values)


# =============================================================================
# Simulation and Analysis
# =============================================================================

def simulate_bandit(
    true_rates: List[float],
    algorithm: str = 'thompson',
    n_pulls: int = 1000,
    **kwargs,
) -> BanditResult:
    """
    Simulate a bandit experiment.

    Parameters
    ----------
    true_rates : list of float
        True conversion rates for each arm
    algorithm : str
        'epsilon_greedy', 'thompson', 'ucb', or 'ucb_tuned'
    n_pulls : int
        Number of total pulls
    **kwargs
        Additional arguments for the algorithm

    Returns
    -------
    BanditResult
        Simulation results
    """
    arm_names = [f'Arm_{i}' for i in range(len(true_rates))]

    algorithms = {
        'epsilon_greedy': EpsilonGreedy,
        'thompson': ThompsonSampling,
        'ucb': UCB,
        'ucb_tuned': UCB_Tuned,
    }

    bandit_class = algorithms.get(algorithm)
    if bandit_class is None:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    bandit = bandit_class(arm_names, **kwargs)

    # Track regret
    best_rate = max(true_rates)
    cumulative_regret = 0
    regret_history = []
    arm_history = []
    total_reward = 0

    for _ in range(n_pulls):
        arm = bandit.select_arm()
        arm_history.append(arm)

        # Simulate binary outcome
        success = np.random.random() < true_rates[arm]
        bandit.update_binary(arm, success)

        # Track regret
        regret = best_rate - true_rates[arm]
        cumulative_regret += regret
        regret_history.append(cumulative_regret)

        total_reward += int(success)

    # Get best arm probabilities (for Thompson)
    if algorithm == 'thompson':
        best_probs = bandit.get_arm_probabilities()
    else:
        # Estimate from observed data
        rates = [arm.conversion_rate for arm in bandit.arms]
        best_probs = np.zeros(len(rates))
        best_probs[np.argmax(rates)] = 1.0

    best_idx = np.argmax(best_probs)

    return BanditResult(
        arms=bandit.arms,
        best_arm=arm_names[best_idx],
        best_arm_probability=best_probs[best_idx],
        total_pulls=n_pulls,
        total_reward=total_reward,
        regret=cumulative_regret,
        regret_history=np.array(regret_history),
        arm_selection_history=np.array(arm_history),
        algorithm=algorithm,
    )


def compare_algorithms(
    true_rates: List[float],
    n_pulls: int = 1000,
    n_simulations: int = 100,
    algorithms: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compare bandit algorithms across multiple simulations.

    Parameters
    ----------
    true_rates : list of float
        True conversion rates for each arm
    n_pulls : int
        Pulls per simulation
    n_simulations : int
        Number of simulations
    algorithms : list of str, optional
        Algorithms to compare (default: all)
    seed : int, optional
        Random seed

    Returns
    -------
    pd.DataFrame
        Comparison results
    """
    if seed is not None:
        np.random.seed(seed)

    if algorithms is None:
        algorithms = ['epsilon_greedy', 'thompson', 'ucb']

    results = []

    for alg in algorithms:
        regrets = []
        correct_best = 0
        total_rewards = []

        for _ in range(n_simulations):
            result = simulate_bandit(true_rates, alg, n_pulls)
            regrets.append(result.regret)
            total_rewards.append(result.total_reward)

            # Check if correctly identified best arm
            true_best = np.argmax(true_rates)
            estimated_best = int(result.best_arm.split('_')[1])
            if true_best == estimated_best:
                correct_best += 1

        results.append({
            'algorithm': alg,
            'mean_regret': np.mean(regrets),
            'std_regret': np.std(regrets),
            'mean_reward': np.mean(total_rewards),
            'correct_rate': correct_best / n_simulations,
        })

    return pd.DataFrame(results)


def plot_regret(
    results: List[BanditResult],
    labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot cumulative regret over time.

    Parameters
    ----------
    results : list of BanditResult
        Results to compare
    labels : list of str, optional
        Labels for each result
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if labels is None:
        labels = [r.algorithm for r in results]

    for result, label in zip(results, labels):
        ax.plot(result.regret_history, label=label)

    ax.set_xlabel('Pull Number')
    ax.set_ylabel('Cumulative Regret')
    ax.set_title('Bandit Algorithm Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_arm_selection(
    result: BanditResult,
    window: int = 100,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot arm selection frequency over time.

    Parameters
    ----------
    result : BanditResult
        Bandit results
    window : int
        Rolling window size
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    n_arms = len(result.arms)
    arm_history = result.arm_selection_history

    for i in range(n_arms):
        # Calculate rolling proportion
        is_arm = (arm_history == i).astype(float)
        rolling_prop = pd.Series(is_arm).rolling(window).mean()
        ax.plot(rolling_prop, label=result.arms[i].name)

    ax.set_xlabel('Pull Number')
    ax.set_ylabel(f'Selection Frequency (rolling {window})')
    ax.set_title('Arm Selection Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    return fig


# =============================================================================
# Bayesian Stopping Rules
# =============================================================================

def bayesian_stopping_check(
    bandit: ThompsonSampling,
    threshold: float = 0.95,
    n_samples: int = 10000,
) -> Tuple[bool, int, float]:
    """
    Check if we should stop the bandit experiment.

    Stops when probability that best arm is best exceeds threshold.

    Parameters
    ----------
    bandit : ThompsonSampling
        Current bandit state
    threshold : float
        Probability threshold for stopping
    n_samples : int
        Monte Carlo samples

    Returns
    -------
    tuple
        (should_stop, best_arm_idx, best_arm_probability)
    """
    probs = bandit.get_arm_probabilities(n_samples)
    best_idx = np.argmax(probs)
    best_prob = probs[best_idx]

    return best_prob >= threshold, best_idx, best_prob


def expected_loss(
    bandit: ThompsonSampling,
    n_samples: int = 10000,
) -> np.ndarray:
    """
    Calculate expected loss for each arm.

    Expected loss = E[max(θ) - θ_i] = opportunity cost of choosing arm i

    Parameters
    ----------
    bandit : ThompsonSampling
        Current bandit state
    n_samples : int
        Monte Carlo samples

    Returns
    -------
    np.ndarray
        Expected loss for each arm
    """
    samples = np.zeros((n_samples, bandit.n_arms))

    for i, arm in enumerate(bandit.arms):
        alpha = bandit.prior_alpha + arm.successes
        beta = bandit.prior_beta + arm.failures
        samples[:, i] = np.random.beta(alpha, beta, n_samples)

    best_sample = np.max(samples, axis=1, keepdims=True)
    losses = best_sample - samples

    return losses.mean(axis=0)
