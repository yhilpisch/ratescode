# Data Inventory

This inventory distinguishes official source snapshots from synthetic and
pedagogical CSV inputs used by the examples. Official datasets carry
same-stem `.meta.json` sidecars with source, retrieval, transformation,
schema, redistribution, and chapter-usage information.

| File | Classification | Metadata sidecar |
|------|----------------|------------------|
| `ch02_cashflows.csv` | teaching/pedagogical snapshot | no |
| `ch03_bond_cashflows.csv` | teaching/pedagogical snapshot | no |
| `ch04_us_treasury_par_yield_curve_official.csv` | official snapshot | yes |
| `ch04_us_treasury_real_yield_curve_official.csv` | official snapshot | yes |
| `ch04_zero_curve.csv` | teaching/pedagogical snapshot | no |
| `ch05_ecb_estr_official.csv` | official snapshot | yes |
| `ch05_money_market_quotes.csv` | teaching/pedagogical snapshot | no |
| `ch05_nyfed_reference_rates_official.csv` | official snapshot | yes |
| `ch05_overnight_rates.csv` | teaching/pedagogical snapshot | no |
| `ch06_swap_rates.csv` | teaching/pedagogical snapshot | no |
| `ch07_sovereign_bonds.csv` | teaching/pedagogical snapshot | no |
| `ch07_sovereign_curve.csv` | teaching/pedagogical snapshot | no |
| `ch07_us_treasury_curve_risk_panel_official.csv` | official snapshot | yes |
| `ch08_futures_contract.csv` | teaching/pedagogical snapshot | no |
| `ch08_futures_hedge_inputs.csv` | teaching/pedagogical snapshot | no |
| `ch09_swap_inputs.csv` | teaching/pedagogical snapshot | no |
| `ch10_bond_portfolio.csv` | teaching/pedagogical snapshot | no |
| `ch10_portfolio_profiles.csv` | teaching/pedagogical snapshot | no |
| `ch11_ecb_aaa_yield_curve_official.csv` | official snapshot | yes |
| `ch11_yield_curve_history.csv` | teaching/pedagogical snapshot | no |
| `ch12_initial_zero_curve.csv` | teaching/pedagogical snapshot | no |
| `ch12_short_rate_params.csv` | teaching/pedagogical snapshot | no |
| `ch13_forward_curve_snapshot.csv` | teaching/pedagogical snapshot | no |
| `ch13_forward_vol_assumptions.csv` | teaching/pedagogical snapshot | no |
| `ch14_caplet_inputs.csv` | teaching/pedagogical snapshot | no |
| `ch14_swaption_inputs.csv` | teaching/pedagogical snapshot | no |
| `ch14_synthetic_vol_surface.csv` | synthetic teaching data | no |
| `ch15_callable_bond_terms.csv` | teaching/pedagogical snapshot | no |
| `ch15_rate_tree_inputs.csv` | teaching/pedagogical snapshot | no |
| `ch16_credit_spread_panel.csv` | teaching/pedagogical snapshot | no |
| `ch16_rating_transition_matrix.csv` | teaching/pedagogical snapshot | no |
| `ch16_synthetic_issuer_panel.csv` | synthetic teaching data | no |
| `ch17_cds_quotes.csv` | teaching/pedagogical snapshot | no |
| `ch17_cds_schedule.csv` | teaching/pedagogical snapshot | no |
| `ch17_discount_curve.csv` | teaching/pedagogical snapshot | no |
| `ch17_recovery_assumptions.csv` | teaching/pedagogical snapshot | no |
| `ch18_merton_inputs.csv` | teaching/pedagogical snapshot | no |
| `ch18_merton_scenarios.csv` | teaching/pedagogical snapshot | no |
| `ch19_allocation_candidates.csv` | teaching/pedagogical snapshot | no |
| `ch19_credit_portfolio.csv` | teaching/pedagogical snapshot | no |
| `ch19_rating_transition_matrix.csv` | teaching/pedagogical snapshot | no |
| `ch19_spread_stress_scenarios.csv` | teaching/pedagogical snapshot | no |
| `ch20_borrower_metrics.csv` | teaching/pedagogical snapshot | no |
| `ch20_covenant_snippets.csv` | teaching/pedagogical snapshot | no |
| `ch20_private_credit_loans.csv` | teaching/pedagogical snapshot | no |
| `ch20_private_credit_stress_scenarios.csv` | teaching/pedagogical snapshot | no |
| `ch21_risk_portfolio.csv` | teaching/pedagogical snapshot | no |
| `ch21_shock_scenarios.csv` | teaching/pedagogical snapshot | no |
| `ch21_us_rates_inflation_shock_history_official.csv` | official snapshot | yes |
| `ch22_factor_changes.csv` | teaching/pedagogical snapshot | no |
| `ch22_portfolio_exposures.csv` | teaching/pedagogical snapshot | no |
| `ch22_stress_scenarios.csv` | teaching/pedagogical snapshot | no |
| `ch23_bond_roundtrip.csv` | teaching/pedagogical snapshot | no |
| `ch23_curve_quotes.csv` | teaching/pedagogical snapshot | no |
| `ch23_swap_repricing.csv` | teaching/pedagogical snapshot | no |
| `ch24_ml_feature_panel.csv` | teaching/pedagogical snapshot | no |
| `ch24_model_comparison.csv` | teaching/pedagogical snapshot | no |
| `ch25_curve_ml_panel.csv` | teaching/pedagogical snapshot | no |
| `ch25_curve_regimes.csv` | teaching/pedagogical snapshot | no |
| `ch25_us_treasury_curve_ml_panel_official.csv` | official snapshot | yes |
| `ch26_credit_classification_metrics.csv` | teaching/pedagogical snapshot | no |
| `ch26_credit_ml_panel.csv` | teaching/pedagogical snapshot | no |
| `ch27_portfolio_signal_panel.csv` | teaching/pedagogical snapshot | no |
| `ch27_stress_alerts.csv` | teaching/pedagogical snapshot | no |
| `ch28_prompt_templates.csv` | teaching/pedagogical snapshot | no |
| `ch28_report_validation.csv` | teaching/pedagogical snapshot | no |
| `ch28_structured_extraction.csv` | teaching/pedagogical snapshot | no |
| `ch29_central_bank_statements.csv` | teaching/pedagogical snapshot | no |
| `ch29_curve_reaction_snapshot.csv` | teaching/pedagogical snapshot | no |
| `ch29_fomc_curve_event_panel_official.csv` | official snapshot | yes |
| `ch29_fomc_documents_official.csv` | official snapshot | yes |
| `ch29_policy_tone_outputs.csv` | teaching/pedagogical snapshot | no |
| `ch30_covenant_extractions.csv` | teaching/pedagogical snapshot | no |
| `ch30_credit_risk_summary.csv` | teaching/pedagogical snapshot | no |
| `ch30_sec_credit_filings_official.csv` | official snapshot | yes |
| `ch30_synthetic_credit_memos.csv` | synthetic teaching data | no |
| `ch31_agent_tasks.csv` | teaching/pedagogical snapshot | no |
| `ch31_agent_validation_log.csv` | teaching/pedagogical snapshot | no |
| `ch31_tool_outputs.csv` | teaching/pedagogical snapshot | no |
| `ch32_portfolio_report_metrics.csv` | teaching/pedagogical snapshot | no |
| `ch32_report_draft.csv` | teaching/pedagogical snapshot | no |
| `ch32_report_validation.csv` | teaching/pedagogical snapshot | no |

## Maintenance notes

- Keep complete `.meta.json` sidecars for official data snapshots.
- Keep synthetic files deterministic and clearly identified in prose where
  interpretation depends on data provenance.
- Do not replace frozen publication datasets with live API pulls without
  updating source notes, validation evidence, and affected examples.
