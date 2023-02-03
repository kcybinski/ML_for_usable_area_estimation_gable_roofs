# ML_for_usable_area_estimation_gable_roofs

## [<img src="https://sciprofiles.com/images/logos/doi.svg" width="20"/>](https://doi.org/10.3390/rs15030863) Machine Learning of Usable Area of Gable-Roof Residential Buildings Based On Topographic Data
This repository contains all the necessary data and code to reproduce results and figures from paper ["Machine Learning of Usable Area of Gable-Roof Residential Buildings Based On Topographic Data"](https://www.mdpi.com/2072-4292/15/3/863) by L. Dawid, K. Cybiński, and Ż. Stręk, namely:
- Folder `Gable-roof` contains the datasets we used to train, and test our Machine Learning (ML) models
- Folder `models` contains the models we chose as performing the best, which's performance has been presented in the paper on Figs 5-7
- Jupyter notebook `Direct_comparison.ipynb`, which has a section including model training, and comparison of Linear Regression, and Neural Network (NN) performance on different dataset combinations
- `auxiliary_funcs.py` is a technical file containing NN and Linear Regression training routines, as well as other function definitions necessary to reproduce Figs 5-7
- Jupyter Notebook `MC.ipynb` is dedicated to reproducing the LiDAR height detection error estimation, and reproduction of Figs 2-4

Code has been written by Kacper Cybiński (University of Warsaw)
