This repository contains a comprehensive Quantitative Finance framework developed in MatLab to analyze equity markets (Nasdaq & NYSE) and implement advanced Portfolio Optimization strategies. The project transitions from raw data processing to complex allocation models, including Black-Litterman and Bayesian approaches.

Project Overview
The analysis is structured into three main phases:

Statistical Profiling: Comparison of daily vs. monthly return distributions, examining the "fat tails" (kurtosis) and the convergence toward normality.

Market Dynamics: Correlation analysis across different time frequencies to identify systemic risk and diversification potential.

Optimal Allocation: Implementation of several portfolio construction methodologies under realistic constraints (e.g., Long-Only, UCITS weight caps).

Features & Models
The codebase (Q1.m) includes the following financial engineering implementations:

Mean-Variance Optimization (MV): Markowitz's classic framework to find the Tangency Portfolio.

Global Minimum Variance (GMV): A risk-focused allocation strategy independent of return expectations.

Black-Litterman Model: Integrating market equilibrium with subjective views to produce robust posterior estimates.

Bayesian Allocation: Using prior distributions to stabilize the estimation of expected returns and reduce sensitivity to historical noise.

Ensemble Strategy (COMBO): A linear combination of models to maximize the Sharpe Ratio and improve out-of-sample stability.

Key Findings
Frequency Effect: The analysis demonstrates how daily leptokurtosis diminishes at a monthly frequency, validating the Central Limit Theorem in a financial context.

Robustness: The Black-Litterman and COMBO models consistently outperform basic Mean-Variance strategies by reducing concentration risk and extreme weight fluctuations.

How to Use
Ensure you have MatLab installed with the Optimization Toolbox.

Place the dataset dataexam_jan26.xlsx in the root directory.

Run the master script Q1.m to generate the full suite of statistical tables and comparative plots.
