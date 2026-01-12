"""A/B Test class for managing experiment lifecycle and analysis."""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, Union, List
from enum import Enum
import matplotlib.pyplot as plt


class MetricType(Enum):
    """Type of metric being tested."""
    PROPORTION = "proportion"  # Binary outcomes (conversion rate)
    CONTINUOUS = "continuous"  # Continuous outcomes (revenue, time)


class TestType(Enum):
    """Type of statistical test."""
    TWO_SIDED = "two_sided"
    ONE_SIDED_GREATER = "one_sided_greater"  # Treatment > Control
    ONE_SIDED_LESS = "one_sided_less"  # Treatment < Control


@dataclass
class ABTestResult:
    """Container for A/B test results."""

    # Group statistics
    control_mean: float
    treatment_mean: float
    control_std: float
    treatment_std: float
    control_n: int
    treatment_n: int

    # Effect estimates
    absolute_lift: float
    relative_lift: float

    # Statistical inference
    test_statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]

    # Test configuration
    alpha: float
    power: float
    metric_type: MetricType
    test_type: TestType

    # Decision
    is_significant: bool

    def __repr__(self) -> str:
        status = "SIGNIFICANT" if self.is_significant else "NOT SIGNIFICANT"
        return (
            f"ABTestResult({status})\n"
            f"  Control: {self.control_mean:.4f} (n={self.control_n})\n"
            f"  Treatment: {self.treatment_mean:.4f} (n={self.treatment_n})\n"
            f"  Lift: {self.relative_lift:.2%} ({self.absolute_lift:+.4f})\n"
            f"  95% CI: [{self.confidence_interval[0]:.4f}, {self.confidence_interval[1]:.4f}]\n"
            f"  p-value: {self.p_value:.4f}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "control_mean": self.control_mean,
            "treatment_mean": self.treatment_mean,
            "control_std": self.control_std,
            "treatment_std": self.treatment_std,
            "control_n": self.control_n,
            "treatment_n": self.treatment_n,
            "absolute_lift": self.absolute_lift,
            "relative_lift": self.relative_lift,
            "test_statistic": self.test_statistic,
            "p_value": self.p_value,
            "ci_lower": self.confidence_interval[0],
            "ci_upper": self.confidence_interval[1],
            "alpha": self.alpha,
            "is_significant": self.is_significant,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame."""
        return pd.DataFrame([self.to_dict()])


class ABTest:
    """
    A/B Test manager for experiment configuration, analysis, and reporting.

    This class provides a complete workflow for A/B testing:
    1. Configure the experiment (alpha, power, MDE)
    2. Calculate required sample size
    3. Analyze results when data is collected
    4. Generate reports and visualizations

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level (Type I error rate)
    power : float, default=0.8
        Statistical power (1 - Type II error rate)
    metric_type : MetricType, default=MetricType.PROPORTION
        Type of metric being measured
    test_type : TestType, default=TestType.TWO_SIDED
        Type of hypothesis test

    Examples
    --------
    >>> # Testing conversion rates
    >>> test = ABTest(alpha=0.05, power=0.8, metric_type=MetricType.PROPORTION)
    >>>
    >>> # Calculate required sample size
    >>> n = test.get_sample_size(baseline_rate=0.10, mde=0.02)
    >>> print(f"Need {n} users per group")
    >>>
    >>> # Analyze results
    >>> result = test.analyze_proportions(
    ...     control_conversions=100, control_total=1000,
    ...     treatment_conversions=120, treatment_total=1000
    ... )
    >>> print(result)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        power: float = 0.8,
        metric_type: MetricType = MetricType.PROPORTION,
        test_type: TestType = TestType.TWO_SIDED,
    ):
        self._validate_params(alpha, power)
        self.alpha = alpha
        self.power = power
        self.metric_type = metric_type
        self.test_type = test_type
        self._results: Optional[ABTestResult] = None

    @staticmethod
    def _validate_params(alpha: float, power: float) -> None:
        """Validate input parameters."""
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        if not 0 < power < 1:
            raise ValueError(f"power must be between 0 and 1, got {power}")

    @property
    def results(self) -> Optional[ABTestResult]:
        """Get the most recent test results."""
        return self._results

    def _get_z_alpha(self) -> float:
        """Get the critical z-value for the significance level."""
        if self.test_type == TestType.TWO_SIDED:
            return stats.norm.ppf(1 - self.alpha / 2)
        else:
            return stats.norm.ppf(1 - self.alpha)

    def _get_z_beta(self) -> float:
        """Get the critical z-value for the power."""
        return stats.norm.ppf(self.power)

    # =========================================================================
    # Sample Size Calculations
    # =========================================================================

    def get_sample_size(
        self,
        baseline_rate: Optional[float] = None,
        mde: Optional[float] = None,
        baseline_mean: Optional[float] = None,
        baseline_std: Optional[float] = None,
        effect_size: Optional[float] = None,
    ) -> int:
        """
        Calculate required sample size per group.

        For proportions, provide baseline_rate and mde.
        For continuous metrics, provide baseline_mean, baseline_std, and effect_size.

        Parameters
        ----------
        baseline_rate : float, optional
            Expected conversion rate for control (for proportions)
        mde : float, optional
            Minimum detectable effect (absolute difference for proportions)
        baseline_mean : float, optional
            Expected mean for control (for continuous metrics)
        baseline_std : float, optional
            Expected standard deviation (for continuous metrics)
        effect_size : float, optional
            Expected absolute effect size (for continuous metrics)

        Returns
        -------
        int
            Required sample size per group
        """
        if self.metric_type == MetricType.PROPORTION:
            if baseline_rate is None or mde is None:
                raise ValueError("For proportions, provide baseline_rate and mde")
            return self._sample_size_proportion(baseline_rate, mde)
        else:
            if baseline_std is None or effect_size is None:
                raise ValueError("For continuous metrics, provide baseline_std and effect_size")
            return self._sample_size_continuous(baseline_std, effect_size)

    def _sample_size_proportion(self, baseline_rate: float, mde: float) -> int:
        """Calculate sample size for proportion test."""
        p1 = baseline_rate
        p2 = baseline_rate + mde
        pooled_p = (p1 + p2) / 2

        z_alpha = self._get_z_alpha()
        z_beta = self._get_z_beta()

        numerator = (
            z_alpha * np.sqrt(2 * pooled_p * (1 - pooled_p))
            + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        ) ** 2
        denominator = (p2 - p1) ** 2

        return int(np.ceil(numerator / denominator))

    def _sample_size_continuous(self, std: float, effect_size: float) -> int:
        """Calculate sample size for continuous metric test."""
        z_alpha = self._get_z_alpha()
        z_beta = self._get_z_beta()

        # Cohen's d
        d = effect_size / std

        # Sample size formula for two-sample t-test
        n = 2 * ((z_alpha + z_beta) / d) ** 2

        return int(np.ceil(n))

    def get_power(
        self,
        sample_size: int,
        baseline_rate: Optional[float] = None,
        mde: Optional[float] = None,
        baseline_std: Optional[float] = None,
        effect_size: Optional[float] = None,
    ) -> float:
        """
        Calculate statistical power for a given sample size.

        Parameters
        ----------
        sample_size : int
            Sample size per group
        baseline_rate : float, optional
            Expected conversion rate for control (for proportions)
        mde : float, optional
            Minimum detectable effect (for proportions)
        baseline_std : float, optional
            Expected standard deviation (for continuous metrics)
        effect_size : float, optional
            Expected absolute effect size (for continuous metrics)

        Returns
        -------
        float
            Statistical power (0 to 1)
        """
        if self.metric_type == MetricType.PROPORTION:
            if baseline_rate is None or mde is None:
                raise ValueError("For proportions, provide baseline_rate and mde")
            return self._power_proportion(sample_size, baseline_rate, mde)
        else:
            if baseline_std is None or effect_size is None:
                raise ValueError("For continuous metrics, provide baseline_std and effect_size")
            return self._power_continuous(sample_size, baseline_std, effect_size)

    def _power_proportion(self, n: int, baseline_rate: float, mde: float) -> float:
        """Calculate power for proportion test."""
        p1 = baseline_rate
        p2 = baseline_rate + mde
        pooled_p = (p1 + p2) / 2

        z_alpha = self._get_z_alpha()

        se_pooled = np.sqrt(2 * pooled_p * (1 - pooled_p) / n)
        se_unpooled = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)

        z_beta = (abs(p2 - p1) - z_alpha * se_pooled) / se_unpooled

        return float(stats.norm.cdf(z_beta))

    def _power_continuous(self, n: int, std: float, effect_size: float) -> float:
        """Calculate power for continuous metric test."""
        z_alpha = self._get_z_alpha()
        d = effect_size / std

        # Non-centrality parameter
        ncp = d * np.sqrt(n / 2)

        # Power calculation
        z_beta = ncp - z_alpha

        return float(stats.norm.cdf(z_beta))

    def get_mde(
        self,
        sample_size: int,
        baseline_rate: Optional[float] = None,
        baseline_std: Optional[float] = None,
    ) -> float:
        """
        Calculate minimum detectable effect for a given sample size.

        Parameters
        ----------
        sample_size : int
            Sample size per group
        baseline_rate : float, optional
            Expected conversion rate (for proportions)
        baseline_std : float, optional
            Expected standard deviation (for continuous metrics)

        Returns
        -------
        float
            Minimum detectable effect (absolute)
        """
        z_alpha = self._get_z_alpha()
        z_beta = self._get_z_beta()

        if self.metric_type == MetricType.PROPORTION:
            if baseline_rate is None:
                raise ValueError("For proportions, provide baseline_rate")
            # Approximate MDE for proportions
            se = np.sqrt(2 * baseline_rate * (1 - baseline_rate) / sample_size)
            mde = (z_alpha + z_beta) * se
        else:
            if baseline_std is None:
                raise ValueError("For continuous metrics, provide baseline_std")
            se = baseline_std * np.sqrt(2 / sample_size)
            mde = (z_alpha + z_beta) * se

        return float(mde)

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    def analyze_proportions(
        self,
        control_conversions: int,
        control_total: int,
        treatment_conversions: int,
        treatment_total: int,
    ) -> ABTestResult:
        """
        Analyze A/B test results for proportion metrics (e.g., conversion rate).

        Parameters
        ----------
        control_conversions : int
            Number of conversions in control group
        control_total : int
            Total observations in control group
        treatment_conversions : int
            Number of conversions in treatment group
        treatment_total : int
            Total observations in treatment group

        Returns
        -------
        ABTestResult
            Complete test results
        """
        # Calculate proportions
        p_control = control_conversions / control_total
        p_treatment = treatment_conversions / treatment_total

        # Standard deviations (for proportions)
        std_control = np.sqrt(p_control * (1 - p_control))
        std_treatment = np.sqrt(p_treatment * (1 - p_treatment))

        # Pooled proportion for z-test
        pooled_p = (control_conversions + treatment_conversions) / (control_total + treatment_total)
        se = np.sqrt(pooled_p * (1 - pooled_p) * (1/control_total + 1/treatment_total))

        # Test statistic
        z_stat = (p_treatment - p_control) / se

        # P-value
        if self.test_type == TestType.TWO_SIDED:
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        elif self.test_type == TestType.ONE_SIDED_GREATER:
            p_value = 1 - stats.norm.cdf(z_stat)
        else:
            p_value = stats.norm.cdf(z_stat)

        # Confidence interval for the difference
        se_diff = np.sqrt(
            p_control * (1 - p_control) / control_total
            + p_treatment * (1 - p_treatment) / treatment_total
        )
        z_crit = stats.norm.ppf(1 - self.alpha / 2)
        diff = p_treatment - p_control
        ci = (diff - z_crit * se_diff, diff + z_crit * se_diff)

        # Lift calculations
        absolute_lift = p_treatment - p_control
        relative_lift = absolute_lift / p_control if p_control > 0 else float('inf')

        self._results = ABTestResult(
            control_mean=p_control,
            treatment_mean=p_treatment,
            control_std=std_control,
            treatment_std=std_treatment,
            control_n=control_total,
            treatment_n=treatment_total,
            absolute_lift=absolute_lift,
            relative_lift=relative_lift,
            test_statistic=z_stat,
            p_value=p_value,
            confidence_interval=ci,
            alpha=self.alpha,
            power=self.power,
            metric_type=self.metric_type,
            test_type=self.test_type,
            is_significant=p_value < self.alpha,
        )

        return self._results

    def analyze_continuous(
        self,
        control_data: Union[np.ndarray, pd.Series, List[float]],
        treatment_data: Union[np.ndarray, pd.Series, List[float]],
        equal_var: bool = False,
    ) -> ABTestResult:
        """
        Analyze A/B test results for continuous metrics (e.g., revenue).

        Parameters
        ----------
        control_data : array-like
            Observations from control group
        treatment_data : array-like
            Observations from treatment group
        equal_var : bool, default=False
            If False, uses Welch's t-test (recommended)

        Returns
        -------
        ABTestResult
            Complete test results
        """
        # Convert to numpy arrays
        control = np.asarray(control_data)
        treatment = np.asarray(treatment_data)

        # Calculate statistics
        mean_control = np.mean(control)
        mean_treatment = np.mean(treatment)
        std_control = np.std(control, ddof=1)
        std_treatment = np.std(treatment, ddof=1)
        n_control = len(control)
        n_treatment = len(treatment)

        # Perform t-test
        if self.test_type == TestType.TWO_SIDED:
            alternative = "two-sided"
        elif self.test_type == TestType.ONE_SIDED_GREATER:
            alternative = "greater"
        else:
            alternative = "less"

        t_stat, p_value = stats.ttest_ind(
            treatment, control, equal_var=equal_var, alternative=alternative
        )

        # Confidence interval for the difference
        diff = mean_treatment - mean_control
        se_diff = np.sqrt(std_control**2 / n_control + std_treatment**2 / n_treatment)

        # Degrees of freedom (Welch-Satterthwaite)
        if not equal_var:
            df = (std_control**2/n_control + std_treatment**2/n_treatment)**2 / (
                (std_control**2/n_control)**2 / (n_control - 1)
                + (std_treatment**2/n_treatment)**2 / (n_treatment - 1)
            )
        else:
            df = n_control + n_treatment - 2

        t_crit = stats.t.ppf(1 - self.alpha / 2, df)
        ci = (diff - t_crit * se_diff, diff + t_crit * se_diff)

        # Lift calculations
        absolute_lift = diff
        relative_lift = absolute_lift / mean_control if mean_control != 0 else float('inf')

        self._results = ABTestResult(
            control_mean=mean_control,
            treatment_mean=mean_treatment,
            control_std=std_control,
            treatment_std=std_treatment,
            control_n=n_control,
            treatment_n=n_treatment,
            absolute_lift=absolute_lift,
            relative_lift=relative_lift,
            test_statistic=t_stat,
            p_value=p_value,
            confidence_interval=ci,
            alpha=self.alpha,
            power=self.power,
            metric_type=MetricType.CONTINUOUS,
            test_type=self.test_type,
            is_significant=p_value < self.alpha,
        )

        return self._results

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        group_col: str,
        metric_col: str,
        control_label: str = "control",
        treatment_label: str = "treatment",
    ) -> ABTestResult:
        """
        Analyze A/B test from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing experiment data
        group_col : str
            Column name for group assignment
        metric_col : str
            Column name for the metric to analyze
        control_label : str, default="control"
            Label for control group in group_col
        treatment_label : str, default="treatment"
            Label for treatment group in group_col

        Returns
        -------
        ABTestResult
            Complete test results
        """
        control_data = df[df[group_col] == control_label][metric_col]
        treatment_data = df[df[group_col] == treatment_label][metric_col]

        if self.metric_type == MetricType.PROPORTION:
            # Assume binary metric for proportions
            return self.analyze_proportions(
                control_conversions=int(control_data.sum()),
                control_total=len(control_data),
                treatment_conversions=int(treatment_data.sum()),
                treatment_total=len(treatment_data),
            )
        else:
            return self.analyze_continuous(control_data, treatment_data)

    # =========================================================================
    # Visualization Methods
    # =========================================================================

    def plot_results(
        self,
        figsize: Tuple[int, int] = (10, 6),
        show_ci: bool = True,
    ) -> plt.Figure:
        """
        Plot A/B test results showing means and confidence intervals.

        Parameters
        ----------
        figsize : tuple, default=(10, 6)
            Figure size
        show_ci : bool, default=True
            Whether to show confidence interval for the difference

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        if self._results is None:
            raise ValueError("No results to plot. Run analyze_* first.")

        r = self._results
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Left plot: Group means with error bars
        ax1 = axes[0]
        groups = ["Control", "Treatment"]
        means = [r.control_mean, r.treatment_mean]
        stds = [r.control_std, r.treatment_std]
        ns = [r.control_n, r.treatment_n]
        errors = [s / np.sqrt(n) * 1.96 for s, n in zip(stds, ns)]

        colors = ["#3498db", "#e74c3c"]
        bars = ax1.bar(groups, means, yerr=errors, capsize=5, color=colors, alpha=0.7)
        ax1.set_ylabel("Mean")
        ax1.set_title("Group Comparison")

        # Add value labels
        for bar, mean, n in zip(bars, means, ns):
            ax1.text(
                bar.get_x() + bar.get_width()/2, bar.get_height() + errors[groups.index(bar.get_x())],
                f"{mean:.4f}\n(n={n})", ha="center", va="bottom", fontsize=9
            )

        # Right plot: Lift with CI
        ax2 = axes[1]
        if show_ci:
            ax2.errorbar(
                0, r.absolute_lift,
                yerr=[[r.absolute_lift - r.confidence_interval[0]],
                      [r.confidence_interval[1] - r.absolute_lift]],
                fmt="o", markersize=10, capsize=10, color="#2ecc71" if r.is_significant else "#95a5a6"
            )
            ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)
            ax2.set_xlim(-0.5, 0.5)
            ax2.set_xticks([])
            ax2.set_ylabel("Absolute Lift (Treatment - Control)")
            ax2.set_title(f"Effect Size\n(p={r.p_value:.4f}, {'Significant' if r.is_significant else 'Not Significant'})")

            # Add CI text
            ax2.text(
                0.02, r.confidence_interval[1],
                f"95% CI: [{r.confidence_interval[0]:.4f}, {r.confidence_interval[1]:.4f}]",
                va="bottom", fontsize=9
            )

        plt.tight_layout()
        return fig

    def plot_power_curve(
        self,
        baseline_rate: Optional[float] = None,
        baseline_std: Optional[float] = None,
        mde_range: Optional[Tuple[float, float]] = None,
        n_points: int = 50,
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """
        Plot power curve showing relationship between effect size and power.

        Parameters
        ----------
        baseline_rate : float, optional
            Baseline rate (for proportions)
        baseline_std : float, optional
            Baseline standard deviation (for continuous)
        mde_range : tuple, optional
            Range of MDE values to plot
        n_points : int, default=50
            Number of points to plot
        figsize : tuple, default=(10, 6)
            Figure size

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        if self._results is None:
            raise ValueError("Run analyze_* first to determine sample sizes.")

        sample_size = self._results.control_n

        # Set up MDE range
        if mde_range is None:
            if self.metric_type == MetricType.PROPORTION and baseline_rate:
                mde_range = (0.001, min(baseline_rate, 1 - baseline_rate) * 0.5)
            else:
                mde_range = (0.01, 0.5)

        mde_values = np.linspace(mde_range[0], mde_range[1], n_points)

        # Calculate power for each MDE
        powers = []
        for mde in mde_values:
            if self.metric_type == MetricType.PROPORTION:
                p = self._power_proportion(sample_size, baseline_rate or 0.1, mde)
            else:
                p = self._power_continuous(sample_size, baseline_std or 1.0, mde)
            powers.append(p)

        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(mde_values, powers, "b-", linewidth=2)
        ax.axhline(y=0.8, color="r", linestyle="--", alpha=0.7, label="80% Power")
        ax.axhline(y=0.9, color="g", linestyle="--", alpha=0.7, label="90% Power")

        ax.set_xlabel("Minimum Detectable Effect (MDE)")
        ax.set_ylabel("Statistical Power")
        ax.set_title(f"Power Curve (n={sample_size} per group, α={self.alpha})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        return fig

    # =========================================================================
    # Reporting
    # =========================================================================

    def summary(self) -> str:
        """
        Generate a text summary of the test results.

        Returns
        -------
        str
            Formatted summary string
        """
        if self._results is None:
            return "No results available. Run analyze_* first."

        r = self._results

        summary = f"""
================================================================================
                            A/B TEST RESULTS SUMMARY
================================================================================

TEST CONFIGURATION
------------------
Metric Type:        {r.metric_type.value}
Test Type:          {r.test_type.value}
Significance Level: α = {r.alpha}
Target Power:       1 - β = {r.power}

GROUP STATISTICS
----------------
                    Control         Treatment
Mean:               {r.control_mean:<15.4f} {r.treatment_mean:.4f}
Std Dev:            {r.control_std:<15.4f} {r.treatment_std:.4f}
Sample Size:        {r.control_n:<15d} {r.treatment_n}

RESULTS
-------
Absolute Lift:      {r.absolute_lift:+.4f}
Relative Lift:      {r.relative_lift:+.2%}
Test Statistic:     {r.test_statistic:.4f}
P-Value:            {r.p_value:.4f}
95% CI:             [{r.confidence_interval[0]:.4f}, {r.confidence_interval[1]:.4f}]

CONCLUSION
----------
{'✓ STATISTICALLY SIGNIFICANT' if r.is_significant else '✗ NOT STATISTICALLY SIGNIFICANT'}
{'The treatment effect is unlikely due to chance.' if r.is_significant else 'We cannot reject the null hypothesis of no difference.'}

================================================================================
"""
        return summary
