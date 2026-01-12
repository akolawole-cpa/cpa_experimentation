"""
CPA Experimentation Framework.

A comprehensive toolkit for experiment design, analysis, and validation
in marketing and product analytics.

Modules
-------
ab_test : Core A/B testing functionality
stats : Statistical tests and effect sizes
hypothesis_testing : Superiority, non-inferiority, equivalence tests
sample_size : Power analysis and sample size calculations
multiple_testing : Multiple comparison corrections (Bonferroni, BH, etc.)
diagnostics : Experiment validation (SRM, A/A tests, etc.)
sequential : Sequential testing and early stopping
adaptive : Multi-armed bandits (Thompson Sampling, UCB)
mmm_validation : MMM calibration against experiments
time_series : ITSA and Causal Impact analysis
geo_testing : Market matching and geo lift

Quick Start
-----------
>>> from experiments import ABTest, MetricType
>>>
>>> # Create and configure test
>>> test = ABTest(alpha=0.05, power=0.8, metric_type=MetricType.PROPORTION)
>>>
>>> # Calculate sample size
>>> n = test.get_sample_size(baseline_rate=0.10, mde=0.02)
>>> print(f"Need {n} users per group")
>>>
>>> # Analyze results
>>> result = test.analyze_proportions(
...     control_conversions=500, control_total=5000,
...     treatment_conversions=550, treatment_total=5000
... )
>>> print(result)
"""

# Core A/B Testing
from .ab_test import (
    ABTest,
    ABTestResult,
    MetricType,
    TestType,
)

# Statistical Tests
from .stats import (
    calculate_sample_size,
    calculate_power,
    z_test_proportions,
    t_test,
    chi_square_test,
    one_sample_t_test,
    paired_t_test,
    mann_whitney_u_test,
    one_way_anova,
    two_way_anova,
    tukey_hsd,
    effect_size_cohens_d,
    effect_size_cohens_h,
)

# Hypothesis Testing (Superiority, Non-inferiority, Equivalence)
from .hypothesis_testing import (
    HypothesisType,
    IntervalTestResult,
    superiority_test,
    non_inferiority_test,
    equivalence_test,
    superiority_test_proportions,
    non_inferiority_test_proportions,
    equivalence_test_proportions,
    plot_interval_test,
    sample_size_superiority,
    sample_size_non_inferiority,
    sample_size_equivalence,
)

# Sample Size and Power Analysis
from .sample_size import (
    PowerAnalysisResult,
    sample_size_proportion,
    sample_size_continuous,
    sample_size_cohens_d,
    sample_size_anova,
    sample_size_paired,
    power_proportion,
    power_continuous,
    mde_proportion,
    mde_continuous,
    cohens_d_to_r,
    r_to_cohens_d,
    cohens_d_to_f,
    eta_squared_to_f,
    odds_ratio_to_d,
    relative_lift_to_absolute,
    plot_power_curve,
    plot_sample_size_curve,
    create_sample_size_table,
)

# Multiple Testing Corrections
from .multiple_testing import (
    MultipleTestResult,
    bonferroni,
    holm_bonferroni,
    sidak,
    benjamini_hochberg,
    benjamini_yekutieli,
    fwer_at_k,
    alpha_for_fwer,
    expected_false_discoveries,
    compare_methods,
    plot_adjusted_pvalues,
    simulate_fwer,
    simulate_fdr,
)

# Diagnostics and Validation
from .diagnostics import (
    SRMResult,
    AATestResult,
    detect_srm,
    detect_srm_multiple,
    run_aa_test,
    plot_aa_test_results,
    detect_novelty_effect,
    plot_novelty_effect,
    check_covariate_balance,
    winsorize_metric,
    cap_metric,
    detect_outliers,
    calculate_variance_reduction,
    pre_experiment_checklist,
    summarize_experiment_health,
)

# Sequential Testing
from .sequential import (
    SequentialTestResult,
    GroupSequentialTest,
    obrien_fleming_spending,
    pocock_spending,
    hwang_shih_decani_spending,
    lan_demets_spending,
    calculate_boundary,
    calculate_sequential_boundaries,
    run_sequential_analysis,
    calculate_adjusted_ci,
    simulate_sequential_test,
)

# Adaptive Testing (Multi-Armed Bandits)
from .adaptive import (
    BanditArm,
    BanditResult,
    MultiArmedBandit,
    EpsilonGreedy,
    ThompsonSampling,
    UCB,
    UCB_Tuned,
    simulate_bandit,
    compare_algorithms,
    plot_regret,
    plot_arm_selection,
    bayesian_stopping_check,
    expected_loss,
)

# MMM Validation
from .mmm_validation import (
    MMMValidationResult,
    validate_mmm_vs_experiment,
    validate_multiple_channels,
    calculate_calibration_metrics,
    plot_validation,
    plot_calibration,
    recommend_calibration,
    apply_calibration,
    validate_adstock_decay,
)

# Time Series (submodule)
from . import time_series
from .time_series import (
    ITSAnalysis,
    ITSAResult,
    run_itsa,
    plot_itsa,
    CausalImpactResult,
    run_causal_impact,
    plot_causal_impact,
)

# Geo Testing (submodule)
from . import geo_testing
from .geo_testing import (
    MarketMatchResult,
    MarketMatcher,
    find_best_control_markets,
    dtw_distance,
    euclidean_distance,
    correlation_distance,
    GeoLiftResult,
    run_geo_lift,
    difference_in_differences,
    synthetic_control,
    plot_geo_lift,
)


__all__ = [
    # A/B Testing
    "ABTest",
    "ABTestResult",
    "MetricType",
    "TestType",

    # Stats
    "calculate_sample_size",
    "calculate_power",
    "z_test_proportions",
    "t_test",
    "chi_square_test",
    "one_sample_t_test",
    "paired_t_test",
    "mann_whitney_u_test",
    "one_way_anova",
    "two_way_anova",
    "tukey_hsd",
    "effect_size_cohens_d",
    "effect_size_cohens_h",

    # Hypothesis Testing
    "HypothesisType",
    "IntervalTestResult",
    "superiority_test",
    "non_inferiority_test",
    "equivalence_test",
    "superiority_test_proportions",
    "non_inferiority_test_proportions",
    "equivalence_test_proportions",
    "plot_interval_test",
    "sample_size_superiority",
    "sample_size_non_inferiority",
    "sample_size_equivalence",

    # Sample Size
    "PowerAnalysisResult",
    "sample_size_proportion",
    "sample_size_continuous",
    "sample_size_cohens_d",
    "sample_size_anova",
    "sample_size_paired",
    "power_proportion",
    "power_continuous",
    "mde_proportion",
    "mde_continuous",
    "cohens_d_to_r",
    "r_to_cohens_d",
    "cohens_d_to_f",
    "eta_squared_to_f",
    "odds_ratio_to_d",
    "relative_lift_to_absolute",
    "plot_power_curve",
    "plot_sample_size_curve",
    "create_sample_size_table",

    # Multiple Testing
    "MultipleTestResult",
    "bonferroni",
    "holm_bonferroni",
    "sidak",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "fwer_at_k",
    "alpha_for_fwer",
    "expected_false_discoveries",
    "compare_methods",
    "plot_adjusted_pvalues",
    "simulate_fwer",
    "simulate_fdr",

    # Diagnostics
    "SRMResult",
    "AATestResult",
    "detect_srm",
    "detect_srm_multiple",
    "run_aa_test",
    "plot_aa_test_results",
    "detect_novelty_effect",
    "plot_novelty_effect",
    "check_covariate_balance",
    "winsorize_metric",
    "cap_metric",
    "detect_outliers",
    "calculate_variance_reduction",
    "pre_experiment_checklist",
    "summarize_experiment_health",

    # Sequential
    "SequentialTestResult",
    "GroupSequentialTest",
    "obrien_fleming_spending",
    "pocock_spending",
    "hwang_shih_decani_spending",
    "lan_demets_spending",
    "calculate_boundary",
    "calculate_sequential_boundaries",
    "run_sequential_analysis",
    "calculate_adjusted_ci",
    "simulate_sequential_test",

    # Adaptive
    "BanditArm",
    "BanditResult",
    "MultiArmedBandit",
    "EpsilonGreedy",
    "ThompsonSampling",
    "UCB",
    "UCB_Tuned",
    "simulate_bandit",
    "compare_algorithms",
    "plot_regret",
    "plot_arm_selection",
    "bayesian_stopping_check",
    "expected_loss",

    # MMM Validation
    "MMMValidationResult",
    "validate_mmm_vs_experiment",
    "validate_multiple_channels",
    "calculate_calibration_metrics",
    "plot_validation",
    "plot_calibration",
    "recommend_calibration",
    "apply_calibration",
    "validate_adstock_decay",

    # Time Series
    "time_series",
    "ITSAnalysis",
    "ITSAResult",
    "run_itsa",
    "plot_itsa",
    "CausalImpactResult",
    "run_causal_impact",
    "plot_causal_impact",

    # Geo Testing
    "geo_testing",
    "MarketMatchResult",
    "MarketMatcher",
    "find_best_control_markets",
    "dtw_distance",
    "euclidean_distance",
    "correlation_distance",
    "GeoLiftResult",
    "run_geo_lift",
    "difference_in_differences",
    "synthetic_control",
    "plot_geo_lift",
]
