## Setup

```
pipenv install
```

## Run examples

```
pipenv run python -m examples.example --dataset iris
pipenv run python -m examples.example --dataset moons --plot
pipenv run python -m examples.example --dataset circles --plot
```

## Benchmark

The following command

```bash
pipenv run python -W ignore -m examples.benchmark --random-state 0
```

outputs

```
The performance compariton:
--------------------
dataset=moons, n_samples=100, n_features=2
Logistic Regression: 0.85
Logistic Regression (nonlinear): 0.94
AdaBoost Classifier: 0.92
AdaBoost Classifier (nonlinear): 0.93
Simple Classifier: 0.8
Simple Classifier (nonlinear): 0.92
AdaBoost Simple Classifier: 0.91
AdaBoost Simple Classifier (nonlinear): 0.95
====================
dataset=circles, n_samples=100, n_features=2
Logistic Regression: 0.45
Logistic Regression (nonlinear): 0.9
AdaBoost Classifier: 0.9
AdaBoost Classifier (nonlinear): 0.88
Simple Classifier: 0.4
Simple Classifier (nonlinear): 0.88
AdaBoost Simple Classifier: 0.76
AdaBoost Simple Classifier (nonlinear): 0.91
====================
dataset=linearly separable, n_samples=100, n_features=2
Logistic Regression: 0.95
Logistic Regression (nonlinear): 0.96
AdaBoost Classifier: 0.89
AdaBoost Classifier (nonlinear): 0.97
Simple Classifier: 0.95
Simple Classifier (nonlinear): 0.95
AdaBoost Simple Classifier: 0.97
AdaBoost Simple Classifier (nonlinear): 0.97
====================

The computational time comparison:
--------------------
dataset=moons, n_samples=100, n_features=2
Logistic Regression: 0.00041961669921875 seconds
Logistic Regression (nonlinear): 0.0010848045349121094 seconds
AdaBoost Classifier: 0.06579995155334473 seconds
AdaBoost Classifier (nonlinear): 0.09208011627197266 seconds
Simple Classifier: 0.000202178955078125 seconds
Simple Classifier (nonlinear): 0.0007181167602539062 seconds
AdaBoost Simple Classifier: 0.030386924743652344 seconds
AdaBoost Simple Classifier (nonlinear): 0.08698391914367676 seconds
====================
dataset=circles, n_samples=100, n_features=2
Logistic Regression: 0.00030803680419921875 seconds
Logistic Regression (nonlinear): 0.001074075698852539 seconds
AdaBoost Classifier: 0.0614471435546875 seconds
AdaBoost Classifier (nonlinear): 0.09307098388671875 seconds
Simple Classifier: 0.00019097328186035156 seconds
Simple Classifier (nonlinear): 0.0006070137023925781 seconds
AdaBoost Simple Classifier: 0.030407190322875977 seconds
AdaBoost Simple Classifier (nonlinear): 0.08505487442016602 seconds
====================
dataset=linearly separable, n_samples=100, n_features=2
Logistic Regression: 0.0004889965057373047 seconds
Logistic Regression (nonlinear): 0.0014030933380126953 seconds
AdaBoost Classifier: 0.05795001983642578 seconds
AdaBoost Classifier (nonlinear): 0.09112811088562012 seconds
Simple Classifier: 0.0001957416534423828 seconds
Simple Classifier (nonlinear): 0.0005719661712646484 seconds
AdaBoost Simple Classifier: 0.03024888038635254 seconds
AdaBoost Simple Classifier (nonlinear): 0.08540225028991699 seconds
====================
```
