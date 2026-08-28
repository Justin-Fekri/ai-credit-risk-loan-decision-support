from credit_risk.data import generate_borrowers
from credit_risk.pipeline import fit_bundle, predict_pd
from credit_risk.policy import recommend, risk_band
from credit_risk.train import subgroup_diagnostics


def test_generate_borrowers_has_required_columns():
    frame = generate_borrowers(200, seed=0)
    assert len(frame) == 200
    assert frame["defaulted"].isin([0, 1]).all()
    assert 0.04 < frame["defaulted"].mean() < 0.45
    assert frame["application_id"].is_unique


def test_pipeline_learns_signal():
    frame = generate_borrowers(800, seed=1)
    train = frame.iloc[:600]
    test = frame.iloc[600:]
    bundle = fit_bundle(train, random_state=1)
    pd_hat = predict_pd(bundle.pipeline, test)
    assert pd_hat.shape == (200,)
    assert pd_hat.min() >= 0
    assert pd_hat.max() <= 1
    # Model should rank actual defaults higher on average than non-defaults.
    defaults = pd_hat[test["defaulted"].to_numpy() == 1]
    goods = pd_hat[test["defaulted"].to_numpy() == 0]
    if len(defaults) and len(goods):
        assert defaults.mean() > goods.mean()


def test_policy_gates():
    action, _ = recommend(0.05, 740, 0.22)
    assert action == "approve"
    action, _ = recommend(0.55, 700, 0.30)
    assert action == "decline"
    action, _ = recommend(0.12, 570, 0.25)
    assert action == "decline"
    assert risk_band(0.06) == "low"
    assert risk_band(0.50) == "severe"


def test_subgroup_diagnostics_cover_audit_groups():
    frame = generate_borrowers(400, seed=4)
    diagnostics = subgroup_diagnostics(frame, frame["defaulted"].to_numpy(dtype=float))
    assert {row["group"] for row in diagnostics["age_band"]} == {"21-34", "35-49", "50+"}
    assert sum(row["n"] for row in diagnostics["age_band"]) == len(frame)
    assert diagnostics["employment_status"]
