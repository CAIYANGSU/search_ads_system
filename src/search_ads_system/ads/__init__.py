"""Explicitly synthetic offline advertising-value and auction simulations."""

from .auction import group_candidates, run_auction
from .bidding import synthetic_bid
from .value_scoring import attribution_scores, search_value_scores

__all__ = ["attribution_scores", "search_value_scores", "synthetic_bid", "group_candidates", "run_auction"]
