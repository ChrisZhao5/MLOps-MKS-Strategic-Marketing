# MKS Strategic Marketing: Data Drift Monitoring 🚀

## Overview
This repository contains a production-grade data drift monitoring pipeline designed for the MKS Strategic Marketing context. In MLOps, deploying a model is only the first step; ensuring its ongoing reliability in a dynamic production environment is critical. 

This project implements statistical monitoring to detect data drift, ensuring that the machine learning models continue to perform accurately as real-world data distributions evolve over time.

## Core File
* `mks_drift_report.py`: The core Python script responsible for continuous data validation. It implements statistical tests (such as the Kolmogorov-Smirnov test) to compare baseline distributions against incoming production data, generating actionable drift reports.

## Key Features
* **Statistical Rigor:** Utilizes robust statistical methods to quantify shifts in feature distributions.
* **Automated Reporting:** Generates clear, interpretable reports flagging features that have drifted beyond acceptable thresholds.
* **Production-Ready:** Designed to be easily integrated into broader MLOps pipelines (e.g., Airflow, GitHub Actions) as a continuous validation gate.

## Business Value
For strategic marketing, acting on outdated customer or market data can lead to suboptimal campaigns and wasted budget. This drift detection module acts as an early warning system, prompting model retraining only when statistically necessary, thereby optimizing compute resources and maintaining high prediction accuracy.

---
*Built for the MKS MLOps workflow.*
