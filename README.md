# Problem-Set-3-Making-Money-with-ML
## Problem description
This repository contains the full workflow and documentation for Problem Set 3 of the course Big Data and Machine Learning for Applied Economics (MECA 4107) at Universidad de los Andes.

It contains the full development of a housing-price prediction system for the Chapinero district in Bogotá. We build a unified database combining nearly 39,000 listings from Properati with rich textual and geospatial features engineered from raw descriptions and curated IDECA layers. The workflow integrates data cleaning, attribute extraction using regular expressions, spatial feature construction, and dimensionality reduction of neighborhood amenities. Multiple model families—linear, regularized, tree-based, ensemble methods, neural networks, and Super Learner—are implemented under a spatial block cross-validation framework to ensure geographic generalization rather than learning neighborhood-specific noise.

Across the seven required model classes, our best-performing approach is a tuned XGBoost model trained with spatial block CV, achieving a Kaggle MAE of approximately 170 million COP. Its performance reflects its ability to capture nonlinear interactions across structural characteristics, text-derived features, and accessibility to urban amenities while avoiding overfitting to localized clusters. The results highlight the value of combining domain knowledge with robust, flexible machine-learning methods for scalable and reliable valuation in Chapinero

-----------------------------------------------------------------------------------------------------------------------------------------------------------------
## Repository Structure
data/ (Contains all datasets used throughout the project)

--> raw:  original Properati files and external data sources

--> processed: cleaned and merged intermediate datasets

--> spatial: – IDECA layers, shapefiles, and geographic resources used for spatial feature engineering

--> final: modeling-ready dataset with all engineered features

scripts/(R scripts that execute each stage of the pipeline)

--> download:scripts for retrieving and parsing raw data

--> cleaning: preprocessing, standardization, deduplication, text-based attribute extraction

--> spatial: construction of spatial distances, amenity-accessibility indices, PCA components

models/ – implementations of OLS/Elastic Net, CART, Random Forest, XGBoost, neural networks, and Super Learner

-----------------------------------------------------------------------------------------------------------------------------------------------------------------
## Instructions

Run "dataDownload" to replicated data

Each model can run with the already saved final data base, thus you only have to choose your favorite
