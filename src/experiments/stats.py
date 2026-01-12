"""Statistical functions for experimentation."""

import numpy as np
from scipy import stats
from typing import Tuple


def calculate_sample_size(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    two_tailed: bool = True,
) -> int:
    """
    Calculate required sample size per group for a proportion test.

    Args:
        baseline_rate: Expected conversion rate for control group (0-1)
        mde: Minimum detectable effect (absolute difference)
        alpha: Significance level (default 0.05)
        power: Statistical power (default 0.8)
        two_tailed: Whether to use two-tailed test (default True)

    Returns:
        Required sample size per group
    """
    p1 = baseline_rate
    p2 = baseline_rate + mde
    pooled_p = (p1 + p2) / 2

    if two_tailed:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)

    z_beta = stats.norm.ppf(power)

    numerator = (
        z_alpha * np.sqrt(2 * pooled_p * (1 - pooled_p))
        + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    denominator = (p2 - p1) ** 2

    return int(np.ceil(numerator / denominator))


def calculate_power(
    baseline_rate: float,
    mde: float,
    sample_size: int,
    alpha: float = 0.05,
    two_tailed: bool = True,
) -> float:
    """
    Calculate statistical power for a given sample size.

    Args:
        baseline_rate: Expected conversion rate for control group (0-1)
        mde: Minimum detectable effect (absolute difference)
        sample_size: Sample size per group
        alpha: Significance level (default 0.05)
        two_tailed: Whether to use two-tailed test (default True)

    Returns:
        Statistical power (0-1)
    """
    p1 = baseline_rate
    p2 = baseline_rate + mde
    pooled_p = (p1 + p2) / 2

    if two_tailed:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)

    se_pooled = np.sqrt(2 * pooled_p * (1 - pooled_p) / sample_size)
    se_unpooled = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / sample_size)

    z_beta = (abs(p2 - p1) - z_alpha * se_pooled) / se_unpooled
    power = stats.norm.cdf(z_beta)

    return power


def z_test_proportions(
    successes_a: int,
    total_a: int,
    successes_b: int,
    total_b: int,
    two_tailed: bool = True,
) -> Tuple[float, float, float, float]:
    """
    Perform a z-test for comparing two proportions.

    Args:
        successes_a: Number of successes in group A (control)
        total_a: Total observations in group A
        successes_b: Number of successes in group B (treatment)
        total_b: Total observations in group B
        two_tailed: Whether to use two-tailed test (default True)

    Returns:
        Tuple of (p_a, p_b, z_statistic, p_value)
    """
    p_a = successes_a / total_a
    p_b = successes_b / total_b

    pooled_p = (successes_a + successes_b) / (total_a + total_b)
    se = np.sqrt(pooled_p * (1 - pooled_p) * (1 / total_a + 1 / total_b))

    z_stat = (p_b - p_a) / se

    if two_tailed:
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    else:
        p_value = 1 - stats.norm.cdf(z_stat)

    return p_a, p_b, z_stat, p_value


def t_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    equal_var: bool = False,
) -> Tuple[float, float, float, float]:
    """
    Perform an independent samples t-test.

    Args:
        group_a: Array of values for group A (control)
        group_b: Array of values for group B (treatment)
        equal_var: Assume equal variances (default False, uses Welch's t-test)

    Returns:
        Tuple of (mean_a, mean_b, t_statistic, p_value)
    """
    mean_a = np.mean(group_a)
    mean_b = np.mean(group_b)

    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=equal_var)

    return mean_a, mean_b, t_stat, p_value


def chi_square_test(
    observed: np.ndarray,
) -> Tuple[float, float, int]:
    """
    Perform a chi-square test for independence.

    Args:
        observed: 2D contingency table of observed frequencies

    Returns:
        Tuple of (chi2_statistic, p_value, degrees_of_freedom)
    """
    chi2, p_value, dof, _ = stats.chi2_contingency(observed)
    return chi2, p_value, dof


def one_sample_t_test(
    data: np.ndarray,
    population_mean: float,
    alternative: str = "two-sided",
) -> Tuple[float, float, float]:
    """
    Perform a one-sample t-test.

    Tests whether the sample mean differs from a known population mean.

    Args:
        data: Array of sample values
        population_mean: Hypothesized population mean (mu_0)
        alternative: Type of test ("two-sided", "greater", "less")

    Returns:
        Tuple of (sample_mean, t_statistic, p_value)

    Example:
        >>> data = np.array([105, 110, 103, 108, 112])
        >>> mean, t_stat, p_val = one_sample_t_test(data, population_mean=100)
        >>> print(f"Sample mean: {mean:.2f}, p-value: {p_val:.4f}")
    """
    sample_mean = np.mean(data)
    t_stat, p_value = stats.ttest_1samp(data, population_mean)

    # Adjust p-value for one-sided tests
    if alternative == "greater":
        p_value = p_value / 2 if t_stat > 0 else 1 - p_value / 2
    elif alternative == "less":
        p_value = p_value / 2 if t_stat < 0 else 1 - p_value / 2

    return sample_mean, t_stat, p_value


def paired_t_test(
    before: np.ndarray,
    after: np.ndarray,
    alternative: str = "two-sided",
) -> Tuple[float, float, float, float]:
    """
    Perform a paired samples t-test.

    Tests whether the mean difference between paired observations is zero.
    Use this when the same subjects are measured twice (e.g., before/after).

    Args:
        before: Array of values before treatment/intervention
        after: Array of values after treatment/intervention
        alternative: Type of test ("two-sided", "greater", "less")

    Returns:
        Tuple of (mean_difference, std_difference, t_statistic, p_value)

    Example:
        >>> before = np.array([100, 110, 95, 105, 120])
        >>> after = np.array([108, 115, 102, 110, 125])
        >>> mean_diff, std_diff, t_stat, p_val = paired_t_test(before, after)
    """
    if len(before) != len(after):
        raise ValueError("Arrays must have the same length for paired t-test")

    differences = after - before
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)

    t_stat, p_value = stats.ttest_rel(after, before)

    # Adjust p-value for one-sided tests
    if alternative == "greater":
        p_value = p_value / 2 if t_stat > 0 else 1 - p_value / 2
    elif alternative == "less":
        p_value = p_value / 2 if t_stat < 0 else 1 - p_value / 2

    return mean_diff, std_diff, t_stat, p_value


def mann_whitney_u_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """
    Perform Mann-Whitney U test (non-parametric alternative to t-test).

    Tests whether two independent samples come from the same distribution.
    Use this when data is not normally distributed or has outliers.

    Args:
        group_a: Array of values for group A (control)
        group_b: Array of values for group B (treatment)
        alternative: Type of test ("two-sided", "greater", "less")

    Returns:
        Tuple of (u_statistic, p_value)

    Example:
        >>> control = np.array([10, 15, 12, 18, 14])
        >>> treatment = np.array([22, 25, 19, 28, 24])
        >>> u_stat, p_val = mann_whitney_u_test(control, treatment)
    """
    u_stat, p_value = stats.mannwhitneyu(
        group_a, group_b, alternative=alternative
    )
    return u_stat, p_value


def one_way_anova(
    *groups: np.ndarray,
) -> Tuple[float, float, int, int]:
    """
    Perform one-way ANOVA (Analysis of Variance).

    Tests whether the means of multiple groups are equal.
    Use this when comparing more than two groups.

    Args:
        *groups: Variable number of arrays, one per group

    Returns:
        Tuple of (f_statistic, p_value, df_between, df_within)

    Example:
        >>> group1 = np.array([10, 12, 14, 16, 18])
        >>> group2 = np.array([15, 17, 19, 21, 23])
        >>> group3 = np.array([20, 22, 24, 26, 28])
        >>> f_stat, p_val, df_b, df_w = one_way_anova(group1, group2, group3)
    """
    if len(groups) < 2:
        raise ValueError("ANOVA requires at least 2 groups")

    f_stat, p_value = stats.f_oneway(*groups)

    # Calculate degrees of freedom
    k = len(groups)  # number of groups
    n_total = sum(len(g) for g in groups)
    df_between = k - 1
    df_within = n_total - k

    return f_stat, p_value, df_between, df_within


def two_way_anova(
    data: np.ndarray,
    factor_a: np.ndarray,
    factor_b: np.ndarray,
) -> dict:
    """
    Perform two-way ANOVA.

    Tests main effects of two factors and their interaction.

    Args:
        data: Array of dependent variable values
        factor_a: Array of factor A levels for each observation
        factor_b: Array of factor B levels for each observation

    Returns:
        Dictionary with ANOVA table results:
        - 'factor_a': (ss, df, f_stat, p_value)
        - 'factor_b': (ss, df, f_stat, p_value)
        - 'interaction': (ss, df, f_stat, p_value)
        - 'residual': (ss, df)

    Note:
        For full two-way ANOVA with interactions, consider using
        statsmodels.formula.api.ols with statsmodels.stats.anova.anova_lm
    """
    import pandas as pd
    from scipy.stats import f as f_dist

    # Create DataFrame for easier manipulation
    df = pd.DataFrame({
        'y': data,
        'A': factor_a,
        'B': factor_b
    })

    # Grand mean
    grand_mean = df['y'].mean()
    n_total = len(df)

    # Factor A means
    a_means = df.groupby('A')['y'].mean()
    a_levels = a_means.index.tolist()
    n_a = len(a_levels)

    # Factor B means
    b_means = df.groupby('B')['y'].mean()
    b_levels = b_means.index.tolist()
    n_b = len(b_levels)

    # Cell means
    cell_means = df.groupby(['A', 'B'])['y'].mean()
    cell_counts = df.groupby(['A', 'B'])['y'].count()

    # Sum of squares
    ss_total = ((df['y'] - grand_mean) ** 2).sum()

    # SS for factor A
    ss_a = sum(cell_counts.groupby('A').sum() * (a_means - grand_mean) ** 2)

    # SS for factor B
    ss_b = sum(cell_counts.groupby('B').sum() * (b_means - grand_mean) ** 2)

    # SS for interaction (using additive model)
    ss_interaction = 0
    for a in a_levels:
        for b in b_levels:
            if (a, b) in cell_means.index:
                expected = a_means[a] + b_means[b] - grand_mean
                n_cell = cell_counts[(a, b)]
                ss_interaction += n_cell * (cell_means[(a, b)] - expected) ** 2

    # SS residual
    ss_residual = ss_total - ss_a - ss_b - ss_interaction

    # Degrees of freedom
    df_a = n_a - 1
    df_b = n_b - 1
    df_interaction = df_a * df_b
    df_residual = n_total - n_a * n_b

    # Mean squares
    ms_a = ss_a / df_a if df_a > 0 else 0
    ms_b = ss_b / df_b if df_b > 0 else 0
    ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0
    ms_residual = ss_residual / df_residual if df_residual > 0 else 0

    # F-statistics and p-values
    if ms_residual > 0:
        f_a = ms_a / ms_residual
        f_b = ms_b / ms_residual
        f_interaction = ms_interaction / ms_residual

        p_a = 1 - f_dist.cdf(f_a, df_a, df_residual)
        p_b = 1 - f_dist.cdf(f_b, df_b, df_residual)
        p_interaction = 1 - f_dist.cdf(f_interaction, df_interaction, df_residual)
    else:
        f_a = f_b = f_interaction = float('inf')
        p_a = p_b = p_interaction = 0.0

    return {
        'factor_a': {'ss': ss_a, 'df': df_a, 'f_stat': f_a, 'p_value': p_a},
        'factor_b': {'ss': ss_b, 'df': df_b, 'f_stat': f_b, 'p_value': p_b},
        'interaction': {'ss': ss_interaction, 'df': df_interaction, 'f_stat': f_interaction, 'p_value': p_interaction},
        'residual': {'ss': ss_residual, 'df': df_residual}
    }


def tukey_hsd(
    *groups: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    Perform Tukey's HSD (Honestly Significant Difference) post-hoc test.

    Use this after ANOVA to determine which specific group pairs differ.

    Args:
        *groups: Variable number of arrays, one per group
        alpha: Significance level (default 0.05)

    Returns:
        Dictionary with pairwise comparisons:
        - Each key is a tuple (i, j) representing group indices
        - Each value is a dict with 'diff', 'se', 'q_stat', 'p_value', 'significant'

    Example:
        >>> g1 = np.array([10, 12, 14])
        >>> g2 = np.array([15, 17, 19])
        >>> g3 = np.array([20, 22, 24])
        >>> results = tukey_hsd(g1, g2, g3)
        >>> for pair, result in results.items():
        ...     print(f"Groups {pair}: diff={result['diff']:.2f}, p={result['p_value']:.4f}")
    """
    from scipy.stats import studentized_range

    k = len(groups)  # number of groups
    n_groups = [len(g) for g in groups]
    n_total = sum(n_groups)
    means = [np.mean(g) for g in groups]

    # Pool variance estimate (MSE from ANOVA)
    ss_within = sum(np.sum((g - np.mean(g))**2) for g in groups)
    df_within = n_total - k
    mse = ss_within / df_within

    results = {}
    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(means[i] - means[j])

            # Standard error for unequal sample sizes
            se = np.sqrt(mse * 0.5 * (1/n_groups[i] + 1/n_groups[j]))

            # q statistic
            q_stat = diff / se

            # p-value from studentized range distribution
            p_value = 1 - studentized_range.cdf(q_stat, k, df_within)

            results[(i, j)] = {
                'diff': means[j] - means[i],  # signed difference
                'se': se,
                'q_stat': q_stat,
                'p_value': p_value,
                'significant': p_value < alpha
            }

    return results


def effect_size_cohens_d(
    group_a: np.ndarray,
    group_b: np.ndarray,
    pooled: bool = True,
) -> float:
    """
    Calculate Cohen's d effect size for two groups.

    Cohen's d measures the standardized difference between two means.
    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large

    Args:
        group_a: Array of values for group A
        group_b: Array of values for group B
        pooled: Use pooled standard deviation (default True)

    Returns:
        Cohen's d effect size
    """
    mean_a = np.mean(group_a)
    mean_b = np.mean(group_b)

    if pooled:
        n_a = len(group_a)
        n_b = len(group_b)
        var_a = np.var(group_a, ddof=1)
        var_b = np.var(group_b, ddof=1)
        pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        d = (mean_b - mean_a) / pooled_std
    else:
        # Use control group std
        d = (mean_b - mean_a) / np.std(group_a, ddof=1)

    return d


def effect_size_cohens_h(
    p1: float,
    p2: float,
) -> float:
    """
    Calculate Cohen's h effect size for two proportions.

    Cohen's h measures the difference between two proportions using
    arcsine transformation.

    Interpretation:
    - |h| < 0.2: negligible
    - 0.2 <= |h| < 0.5: small
    - 0.5 <= |h| < 0.8: medium
    - |h| >= 0.8: large

    Args:
        p1: First proportion (0 to 1)
        p2: Second proportion (0 to 1)

    Returns:
        Cohen's h effect size
    """
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    return phi2 - phi1
