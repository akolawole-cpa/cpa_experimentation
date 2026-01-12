"""Geo-based experimentation methods."""

from .market_matcher import (
    MarketMatchResult,
    MarketMatcher,
    find_best_control_markets,
    dtw_distance,
    euclidean_distance,
    correlation_distance,
)
from .geo_lift import (
    GeoLiftResult,
    run_geo_lift,
    difference_in_differences,
    synthetic_control,
    plot_geo_lift,
)

__all__ = [
    # Market matching
    "MarketMatchResult",
    "MarketMatcher",
    "find_best_control_markets",
    "dtw_distance",
    "euclidean_distance",
    "correlation_distance",
    # Geo lift
    "GeoLiftResult",
    "run_geo_lift",
    "difference_in_differences",
    "synthetic_control",
    "plot_geo_lift",
]
