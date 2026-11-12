# Linear Regression by Gradient Descent: Wine Quality

Group 1 class project for **MAI391 – Mathematics for Machine Learning**.

## Overview
This project predicts the quality score of red wine from its physicochemical properties using linear regression. The model is trained with gradient descent (plain and with momentum / heavy-ball) and written from scratch in **numpy**. We compare it with the closed-form normal equation, and use the eigenvalues of the Hessian (the condition number) to explain how the step size and momentum affect convergence.

## Dataset
- **Source:** [Wine Quality (red wine), UCI Machine Learning Repository](https://doi.org/10.24432/C56S3T) (CC BY 4.0)
- **Size:** 1,599 samples, 11 features
- **Target:** wine quality score (regression)
- **Split:** 1,279 training / 320 test (fixed seed 42)

The data file is `data/winequality-red.csv`.

## Methods
- Linear regression with mean squared error loss
- Gradient descent and heavy-ball momentum, with the step size chosen from the Hessian eigenvalues
- Normal equation as the exact reference solution
- Feature standardization and degree-2 polynomial features
- scikit-learn used only as a cross-check

## Repository Structure
```
├── 3 Assignments/     # separate course assignments (not part of the main project)
├── data/              # dataset
├── src/
│   ├── gd_linreg.py                 # core algorithm (numpy only)
│   ├── test_core.py                 # 7 tests, from a hand-solved example to real data
│   ├── run_experiments.py           # runs all experiments, writes src/output/
│   └── CP_Group1_GD_LinReg.ipynb    # step-by-step notebook
├── Report.pdf         # full written report
├── slides.pdf         # presentation slides
├── Contribution.pdf   # member contributions
├── requirements.txt   # Python dependencies
└── README.md
```

> **Note:** `3 Assignments/` holds the three course assignments completed before the main project. They are included for submission only.

## How to Run
```bash
git clone https://github.com/comma-loop/Maths-For-Machine-Learning-Project.git
cd Maths-For-Machine-Learning-Project
pip install -r requirements.txt
cd src
python test_core.py          # run the tests
python run_experiments.py    # generate all tables and figures into src/output/
```
To follow the work step by step, open `CP_Group1_GD_LinReg.ipynb` in Jupyter and choose Run All.

## Results
All tables, figures and numbers are in `Report.pdf` and in `src/output/` (`results.txt`, `tables/`, `figures/`).

## Team
| Name | Student ID |
|------|------------|
| [Name] | [ID] |

See `Contribution.pdf` for who did what.

## Acknowledgments
Cortez, P., Cerdeira, A., Almeida, F., Matos, T. and Reis, J. (2009). Wine Quality. UCI Machine Learning Repository. https://doi.org/10.24432/C56S3T
