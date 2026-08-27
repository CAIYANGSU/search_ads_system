from __future__ import annotations

import numpy as np
import pandas as pd
import sys
import yaml

from search_ads_system.ads.auction import group_candidates, run_auction
from search_ads_system.ads.bidding import synthetic_bid, target_cpa_bid, value_based_bid
from search_ads_system.ads.pacing import feedback_pacing_multiplier, pacing_multipliers
from search_ads_system.ads.simulator import SimulationConfig, _pacing_metrics, future_b_isolation_contract, simulate_attribution, simulate_search_conversion
from pipeline.run_ads_business_simulation import _reject_future_b_path, main as business_main


def test_first_and_second_price_clearing_and_winner_ranking() -> None:
    candidates = pd.DataFrame({"synthetic_auction_id": [0, 0], "quality": [1.0, 1.0], "bid": [5.0, 3.0]})
    first, _ = run_auction(candidates, quality_column="quality", bid_column="bid", mechanism="first_price")
    second, _ = run_auction(candidates, quality_column="quality", bid_column="bid", mechanism="second_price")
    assert first.winner_row_index.iat[0] == second.winner_row_index.iat[0] == 0
    assert first.payment.iat[0] == 5.0 and second.payment.iat[0] == 3.0


def test_grouping_seed_budget_and_pacing_are_deterministic() -> None:
    frame = pd.DataFrame({"x": range(10)})
    assert group_candidates(frame, candidates_per_auction=2, seed=7).synthetic_auction_id.equals(group_candidates(frame, candidates_per_auction=2, seed=7).synthetic_auction_id)
    winners, _ = run_auction(pd.DataFrame({"synthetic_auction_id": [0, 0], "quality": [1., 1.], "bid": [5., 3.]}), quality_column="quality", bid_column="bid", mechanism="first_price", budget=4.0)
    assert not winners.won.iat[0] and winners.payment.iat[0] == 0.0
    assert np.all((pacing_multipliers(np.array([1., 1.]), budget=2, minimum=.5, maximum=1.5) >= .5) & (pacing_multipliers(np.array([1., 1.]), budget=2, minimum=.5, maximum=1.5) <= 1.5))


def test_target_cpa_and_value_bid_monotonicity() -> None:
    probability = np.array([.1, .2])
    assert np.all(target_cpa_bid(calibrated_pctcvr=probability, target_cpa=2., bid_scale=1.) >= target_cpa_bid(calibrated_pctcvr=probability, target_cpa=1., bid_scale=1.))
    assert target_cpa_bid(calibrated_pctcvr=np.array([.2]), target_cpa=1., bid_scale=1.).item() >= target_cpa_bid(calibrated_pctcvr=np.array([.1]), target_cpa=1., bid_scale=1.).item()
    assert value_based_bid(calibrated_pctcvr=np.array([.2]), predicted_conditional_conversion_value=np.array([10.]), target_roas=2., bid_scale=1.).item() >= value_based_bid(calibrated_pctcvr=np.array([.2]), predicted_conditional_conversion_value=np.array([5.]), target_roas=2., bid_scale=1.).item()
    assert value_based_bid(calibrated_pctcvr=np.array([.2]), predicted_conditional_conversion_value=np.array([10.]), target_roas=4., bid_scale=1.).item() <= value_based_bid(calibrated_pctcvr=np.array([.2]), predicted_conditional_conversion_value=np.array([10.]), target_roas=2., bid_scale=1.).item()


def test_feedback_pacing_moves_in_the_correct_direction_and_respects_bounds() -> None:
    ahead = feedback_pacing_multiplier(current_multiplier=1., elapsed_horizon_fraction=.2, cumulative_spend=.8, total_budget=1., minimum=.5, maximum=2., alpha=.2)
    behind = feedback_pacing_multiplier(current_multiplier=1., elapsed_horizon_fraction=.2, cumulative_spend=.1, total_budget=1., minimum=.5, maximum=2., alpha=.2)
    assert ahead < 1. < behind
    assert feedback_pacing_multiplier(current_multiplier=2., elapsed_horizon_fraction=.01, cumulative_spend=1., total_budget=1., minimum=.5, maximum=2., alpha=1.) == .5
    assert feedback_pacing_multiplier(current_multiplier=.5, elapsed_horizon_fraction=1., cumulative_spend=0., total_budget=1., minimum=.5, maximum=2., alpha=1.) == 2.


def test_budget_exhaustion_uses_cumulative_spend_over_configured_budget() -> None:
    winners = pd.DataFrame({"won": [True] * 20, "winner_row_index": list(range(20))})
    early_curve = pd.DataFrame({"cumulative_spend": [.5, 1.] + [1.] * 18})
    late_curve = pd.DataFrame({"cumulative_spend": list(np.linspace(.01, .98, 18)) + [.99, 1.]})
    early = _pacing_metrics(winners, early_curve, budget=1., buckets=10)
    late = _pacing_metrics(winners, late_curve, budget=1., buckets=10)
    assert early["budget_exhaustion_horizon_fraction"] == .1 and early["early_budget_exhaustion"] is True
    assert late["budget_exhaustion_horizon_fraction"] == .95 and late["early_budget_exhaustion"] is False


def test_feedback_pacing_requires_no_labels() -> None:
    candidates = pd.DataFrame({"synthetic_auction_id": [0, 0, 1, 1], "quality": [.2, .1, .3, .2], "bid": [1., .5, 1., .5]})
    winners, curve = run_auction(candidates, quality_column="quality", bid_column="bid", mechanism="second_price", budget=1., feedback_pacing={"minimum": .5, "maximum": 2., "alpha": .2})
    assert len(winners) == len(curve) == 2
    assert curve.pacing_multiplier.between(.5, 2.).all()


def test_raw_calibrated_and_search_value_paths_stay_independent() -> None:
    frame = pd.DataFrame({"raw_pctr": [.1,.2,.3,.4], "raw_pctcvr": [.01,.04,.09,.16], "calibrated_pctr": [.15,.2,.25,.3], "calibrated_pctcvr": [.02,.03,.08,.1], "click": [0,1,0,1], "conversion": [0,0,0,1], "cost": [1.,1.,1.,1.]})
    report, _, _ = simulate_attribution(frame, config=SimulationConfig(candidates_per_auction=2, total_budget=10), calibrated_available=True)
    assert report["bid_semantics"] == "synthetic_offline_simulation"
    assert report["policies"]["calibrated_ctcvr_scaled"]["available"]
    search = pd.DataFrame({"pCVR_clicked": [.1,.3], "predicted_conditional_value": [5., 10.], "conversion_label": [0, 1], "conversion_value_eur": [0., 15.]})
    search_report, comparison, deciles = simulate_search_conversion(search, config=SimulationConfig(candidates_per_auction=2, total_budget=10))
    assert "impression" in search_report["definition"] and report["candidate_grouping"].startswith("synthetic")
    assert set(comparison.policy) == {"random_baseline", "pCVR_clicked", "predicted_conditional_value", "expected_value_per_click"}
    assert len(deciles) == 10
    assert synthetic_bid("fixed_bid", base_bid=2.0, pctr=np.array([.1,.2])).tolist() == [2.0, 2.0]
    assert future_b_isolation_contract()["future_b_read_for_policy_selection"] is False
    assert report["policies"]["target_cpa_bidding"]["available"] is True
    assert report["value_based_bidding"]["available"] is False
    assert report["cross_dataset_join"] is False
    for policy in report["policies"].values():
        if policy.get("available"):
            assert policy["simulated_spend"] <= 10 + 1e-12


def test_pacing_is_deterministic_and_shares_auction_order() -> None:
    frame = pd.DataFrame({"raw_pctr": [.1,.2,.3,.4,.5,.6], "raw_pctcvr": [.01,.02,.03,.04,.05,.06], "calibrated_pctr": [.1,.2,.3,.4,.5,.6], "calibrated_pctcvr": [.01,.02,.03,.04,.05,.06], "click": [0,1,0,1,0,1], "conversion": [0,0,0,1,0,1], "cost": [1.] * 6})
    config = SimulationConfig(candidates_per_auction=2, total_budget=.05, budget_levels=(.25,.5,.75,1.), seed=11)
    first, _, first_curve = simulate_attribution(frame, config=config, calibrated_available=True)
    second, _, second_curve = simulate_attribution(frame, config=config, calibrated_available=True)
    assert first["pacing_comparison"] == second["pacing_comparison"]
    selected = first_curve.loc[(first_curve.policy_name == "calibrated_ctcvr_scaled") & (first_curve.budget_level == 1.0)]
    orders = [tuple(group.auction_order) for _, group in selected.groupby("pacing_mode")]
    assert len(set(orders)) == 1


def test_future_b_path_is_rejected_and_value_labels_cannot_enable_bidding(tmp_path) -> None:
    with np.testing.assert_raises(RuntimeError):
        _reject_future_b_path(tmp_path / "split" / "future_b" / "part-00000.csv")
    frame = pd.DataFrame({"raw_pctr": [.1,.2], "raw_pctcvr": [.01,.02], "calibrated_pctr": [.1,.2], "calibrated_pctcvr": [.01,.02], "click": [0,1], "conversion": [0,1], "conversion_value_eur": [0., 100.]})
    report, _, _ = simulate_attribution(frame, config=SimulationConfig(candidates_per_auction=2, value_prediction_column="conversion_value_eur"), calibrated_available=True)
    assert report["value_based_bidding"]["available"] is False


def test_cli_runs_full_synthetic_sanity_without_future_b(tmp_path, monkeypatch) -> None:
    attribution_root, calibration, metrics, output = (tmp_path / name for name in ("attribution", "calibration", "metrics", "business"))
    prediction = calibration / "predictions" / "sanity" / "calibration_eval"; prediction.mkdir(parents=True)
    pd.DataFrame({"event_id": ["e1", "e2", "e3", "e4"], "timestamp": [10, 11, 12, 13], "click": [0, 1, 0, 1], "conversion": [0, 0, 0, 1], "raw_pctr": [.1,.2,.3,.4], "raw_pctcvr": [.01,.04,.09,.16]}).to_csv(prediction / "part-00000.csv", index=False)
    future_a = attribution_root / "split" / "future_a"; future_a.mkdir(parents=True)
    pd.DataFrame({"event_id": ["e1", "e2", "e3", "e4"], "campaign": ["c"] * 4, "cost": [1.] * 4}).to_csv(future_a / "part-00000.csv", index=False)
    metrics.mkdir(); (metrics / "calibration_metrics_sanity.json").write_text('{"selected_calibrator": {"ctr": "raw", "ctcvr": "raw"}}')
    search = tmp_path / "search_predictions"; search.mkdir()
    pd.DataFrame({"user_id": ["u1", "u2"], "product_id": ["p1", "p2"], "conversion_label": [0, 1], "has_conversion_value": [0, 1], "conversion_value_eur": [0., 10.], "pCVR_clicked": [.1, .3], "predicted_conditional_value": [5., 10.], "expected_value_per_click": [.5, 3.]}).to_csv(search / "part-00000.csv", index=False)
    config = {"project": {"seed": 7}, "attribution_preprocessing": {"temporal_output_dir": str(attribution_root)}, "attribution_esmm": {"enabled": True, "hash_buckets": {"user_id": 17, "campaign_id": 17, "categories": 17}}, "attribution_calibration": {"output_dir": str(calibration), "metrics_dir": str(metrics)}, "business_simulation": {"output_dir": str(output), "candidates_per_auction": 2, "total_budget": 10, "search_prediction_path": str(search), "sanity": {"max_rows": 4}}}
    config_path = tmp_path / "config.yaml"; config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["run_ads_business_simulation.py", "--config", str(config_path), "--stage", "sanity"])
    business_main()
    report = yaml.safe_load((output / "metrics" / "auction_metrics.json").read_text())
    assert report["future_b_read_for_policy_selection"] is False
    assert report["future_b_read_for_bidding_experiment"] is False
    assert (output / "tables" / "budget_curve.csv").is_file()
    assert report["search_conversion"]["available"] is True
    assert (output / "tables" / "search_conversion_value_policy_comparison.csv").is_file()
    for name in ("bidding_policy_comparison.json", "pacing_comparison.json", "spend_trajectory.csv", "bidding_summary.md"):
        assert (output / "metrics" / name).is_file()
