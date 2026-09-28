# Notebooks Overview

This folder contains companion notebooks for all chapters and appendices in
*Python & AI for Rates, Bonds, and Credit*. Each notebook follows the book's
visual identity, explains the calculation before running code, includes a
guided mini-lab, and ends with short exercises. Code cells are unexecuted in
the distributed notebooks.

## Run in Google Colab

Open any notebook and select its **Open in Google Colab** link. The setup cell
clones the public [ratescode repository](https://github.com/yhilpisch/ratescode)
into a fresh Colab runtime, or reuses an existing checkout. Run the notebook
from the top; the setup resolves the repository paths and installs only the
core packages if they are missing. No Google Drive mount or credentials are
required.

The Colab setup assumes the repository's default `main` branch. If you are
working from another branch or a local checkout, run the setup cell there; it
detects the existing repository root. Examples use the bundled, frozen data
snapshots rather than live market feeds.

## Chapter notebooks

- `ch01_fixed_income_landscape.ipynb` — Chapter 1: The Fixed-Income Landscape and Problem Map.
- `ch02_cash_flows_discounting_present_value.ipynb` — Chapter 2: Cash Flows, Discounting, and Present Value.
- `ch03_bond_mathematics_market_conventions.ipynb` — Chapter 3: Bond Mathematics and Market Conventions.
- `ch04_yield_curves_zero_rates_forward_rates.ipynb` — Chapter 4: Yield Curves, Zero Rates, and Forward Rates.
- `ch05_money_markets_short_rate_instruments.ipynb` — Chapter 5: Money Markets and Short-Rate Instruments.
- `ch06_curve_bootstrapping_multi_curve_logic.ipynb` — Chapter 6: Curve Bootstrapping and Multi-Curve Logic.
- `ch07_government_bonds_sovereign_curve_risk.ipynb` — Chapter 7: Government Bonds and Sovereign Curve Risk.
- `ch08_bond_futures_duration_hedging.ipynb` — Chapter 8: Bond Futures and Duration Hedging.
- `ch09_interest_rate_swaps.ipynb` — Chapter 9: Interest-Rate Swaps.
- `ch10_bond_portfolio_construction.ipynb` — Chapter 10: Bond Portfolio Construction.
- `ch11_yield_curve_dynamics_factor_models.ipynb` — Chapter 11: Yield-Curve Dynamics and Factor Models.
- `ch12_short_rate_models_vasicek_cir_hull_white.ipynb` — Chapter 12: Short-Rate Models: Vasicek, CIR, and Hull-White.
- `ch13_hjm_lmm_perspectives.ipynb` — Chapter 13: HJM and LMM Perspectives.
- `ch14_caps_floors_swaptions.ipynb` — Chapter 14: Caps, Floors, and Swaptions.
- `ch15_callable_bonds_embedded_options.ipynb` — Chapter 15: Callable Bonds and Embedded Options.
- `ch16_credit_spreads_ratings_default_risk.ipynb` — Chapter 16: Credit Spreads, Ratings, and Default Risk.
- `ch17_reduced_form_credit_models_cds.ipynb` — Chapter 17: Reduced-Form Credit Models and CDS.
- `ch18_structural_credit_models.ipynb` — Chapter 18: Structural Credit Models.
- `ch19_credit_portfolios_spread_risk_management.ipynb` — Chapter 19: Credit Portfolios and Spread-Risk Management.
- `ch20_private_credit_analytics.ipynb` — Chapter 20: Private Credit Analytics.
- `ch21_interest_rate_spread_risk.ipynb` — Chapter 21: Interest-Rate and Spread Risk.
- `ch22_var_expected_shortfall_stress_testing.ipynb` — Chapter 22: VaR, Expected Shortfall, and Stress Testing.
- `ch23_model_risk_calibration_risk_verification.ipynb` — Chapter 23: Model Risk, Calibration Risk, and Verification.
- `ch24_machine_learning_workflow_fixed_income.ipynb` — Chapter 24: Machine Learning Workflow for Fixed Income.
- `ch25_ml_yield_curves_rates.ipynb` — Chapter 25: ML for Yield Curves and Rates.
- `ch26_ml_credit_spreads_default_risk.ipynb` — Chapter 26: ML for Credit Spreads and Default Risk.
- `ch27_ml_portfolio_risk_signals.ipynb` — Chapter 27: ML for Portfolio and Risk Signals.
- `ch28_genai_workflows_fixed_income.ipynb` — Chapter 28: GenAI Workflows for Fixed Income.
- `ch29_central_bank_macro_document_intelligence.ipynb` — Chapter 29: Central-Bank and Macro Document Intelligence.
- `ch30_credit_documents_covenants_research_notes.ipynb` — Chapter 30: Credit Documents, Covenants, and Research Notes.
- `ch31_tool_using_agents_fixed_income_analysis.ipynb` — Chapter 31: Tool-Using Agents for Fixed-Income Analysis.
- `ch32_automated_reporting_communication.ipynb` — Chapter 32: Automated Reporting and Communication.

## Appendix notebooks

- `appx_a_stochastic_processes_brownian_motion.ipynb` — Appx A Stochastic Processes Brownian Motion.
- `appx_b_ito_calculus_refresher.ipynb` — Appx B Ito Calculus Refresher.
- `appx_c_risk_neutral_valuation.ipynb` — Appx C Risk Neutral Valuation.
- `appx_d_change_of_numeraire_forward_measures.ipynb` — Appx D Change Of Numeraire Forward Measures.
- `appx_e_affine_term_structure_models.ipynb` — Appx E Affine Term Structure Models.
- `appx_f_numerical_methods_fixed_income.ipynb` — Appx F Numerical Methods Fixed Income.
- `appx_g_optimization_refresher.ipynb` — Appx G Optimization Refresher.
- `appx_h_mathematical_notation_reference.ipynb` — Appx H Mathematical Notation Reference.

For local use, install the packages in `requirements.txt` and launch Jupyter
from this directory or the repository root. The setup cells detect both
locations.
