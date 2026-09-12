# Predicting Climate-Adaptation Outcomes Using Machine Learning
 
A forward-time evolutionary simulation of a diploid population under increasing heat stress, combined with machine learning models trained to predict the expected final frequency of a heat-tolerance allele from environmental and demographic starting conditions.
 
---
 
## What this project does
 
Real-world evolution takes thousands of generations to observe. This project replaces that wait with a mechanistic simulation — one that explicitly models survival selection, Mendelian inheritance, genetic drift, migration, mutation, and food-limited population dynamics — and then trains ML models to learn which combinations of starting conditions lead to adaptation, stagnation, or population collapse.
 
The result is a surrogate model: given a new set of environmental parameters, it predicts the expected evolutionary outcome without running the simulation.
 
---
 
## How it works
 
```
Initial conditions (temperature rate, food, selection strength, population size, ...)
                          │
                          ▼
     Forward-time diploid simulation over N generations
          ├── Survival selection (heat + allele-dependent)
          ├── Random mating + Mendelian inheritance
          ├── Rare mutation
          ├── Migration
          └── Environment-dependent carrying capacity
                          │
                          ▼
     Final allele frequency (averaged over 3 stochastic replicates)
                          │
                          ▼
     ML model learns: input conditions → expected outcome
```
 
**600 parameter scenarios** were sampled across the ranges below. Each was run with **3 independent random seeds** and averaged to estimate the expected outcome rather than a single stochastic draw.
 
---
 
## Input features
 
| Feature | Range | Meaning |
|---|---|---|
| `initial_allele_frequency` | 0.03 – 0.70 | Starting fraction of heat-tolerance alleles |
| `temperature_rate` | 0.002 – 0.045 | Temperature anomaly added per generation |
| `selection_strength` | 0.03 – 0.35 | Survival advantage conferred by the allele under heat |
| `food_availability` | 0.35 – 1.00 | Relative resource availability |
| `migration_rate` | 0.00 – 0.20 | Fraction of population replaced by migrants per generation |
| `population_size` | 90 – 280 | Initial number of individuals |
| `generations` | 70 – 170 | Simulation duration |
| `final_temperature_anomaly` | derived | `temperature_rate × generations` |
 
**Target:** `final_allele_frequency` — expected frequency of the heat-tolerance allele after the final generation.
 
---
 
## Models and results
 
Two regression models were trained on an 80/20 train/test split.
 
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Ridge Regression | 0.0526 | 0.0655 | 0.775 |
| **Random Forest** | **0.0299** | **0.0455** | **0.891** |
 
Random Forest outperforms Ridge across all metrics, confirming that evolutionary outcomes involve non-linear interactions — heat stress and selection strength interact multiplicatively, small populations amplify genetic drift, and migration can either rescue or dilute a beneficial allele depending on conditions.
 
---
 
## Feature importance
  
Cumulative temperature anomaly dominates (0.49), followed by migration rate (0.23) and temperature rate (0.14). Initial allele frequency and population size have relatively low importance — the trajectory of warming and whether migrants dilute the allele matters more than starting conditions.
 
---
 
## Survival model
 
Each individual's per-generation survival probability:
 
```
P(survival) = clip(b + f·F - h·T + s·T·(g/2), 0.03, 0.98)
```
 
| Symbol | Meaning |
|---|---|
| `b` | Baseline survival (0.75) |
| `F` | Food availability |
| `f` | Food effect coefficient (0.18) |
| `T` | Current heat stress (temperature_rate × generation) |
| `h` | Heat harm coefficient (0.35) |
| `s` | Selection strength |
| `g` | Individual genotype (0, 1, or 2 copies of the allele) |
 
Carrying capacity is environment-dependent: it shrinks with accumulated heat stress and scales with food availability, so population collapse is a real outcome under extreme parameter combinations.
 
---
 
## Scope and limitations
 
- The heat-tolerance allele is a simplified single-gene proxy. Most real traits are polygenic.
- All parameters are modelling assumptions, not measured biological constants.
- The dataset is entirely synthetic. Prediction accuracy applies only within the simulation's own rules.
- This project supports scenario analysis under explicit assumptions — not deterministic forecasting of real biological evolution.
---
 
## Running the project
 
```bash
pip install -r requirements.txt
python simulation.py   # generates dataset + trajectory figures
python train.py        # trains models + generates all remaining figures
```
 
All outputs are written to `outputs/`.
 
**Dependencies:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`
 
---
