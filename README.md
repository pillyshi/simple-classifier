## Setup

### Install as a project
```
pipenv install
```

### Install as a package
```
pipenv install -e /path/to/simple-classifier
```

## Run examples

```
pipenv run python -m examples.adaboost --dataset iris --random-state 0
pipenv run python -m examples.adaboost --dataset moons --random-state 0 --plot
pipenv run python -m examples.adaboost --dataset circles --random-state 0 --plot
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
Logistic Regression: 0.0005271434783935547 seconds
Logistic Regression (nonlinear): 0.0012478828430175781 seconds
AdaBoost Classifier: 0.05403017997741699 seconds
AdaBoost Classifier (nonlinear): 0.08444905281066895 seconds
Simple Classifier: 0.0051250457763671875 seconds
Simple Classifier (nonlinear): 0.005229949951171875 seconds
AdaBoost Simple Classifier: 0.040856122970581055 seconds
AdaBoost Simple Classifier (nonlinear): 0.07993602752685547 seconds
====================
dataset=circles, n_samples=100, n_features=2
Logistic Regression: 0.0003142356872558594 seconds
Logistic Regression (nonlinear): 0.0010678768157958984 seconds
AdaBoost Classifier: 0.05792498588562012 seconds
AdaBoost Classifier (nonlinear): 0.08445096015930176 seconds
Simple Classifier: 0.004865884780883789 seconds
Simple Classifier (nonlinear): 0.004904031753540039 seconds
AdaBoost Simple Classifier: 0.038384199142456055 seconds
AdaBoost Simple Classifier (nonlinear): 0.08080601692199707 seconds
====================
dataset=linearly separable, n_samples=100, n_features=2
Logistic Regression: 0.0003249645233154297 seconds
Logistic Regression (nonlinear): 0.0010859966278076172 seconds
AdaBoost Classifier: 0.05123090744018555 seconds
AdaBoost Classifier (nonlinear): 0.0893399715423584 seconds
Simple Classifier: 0.0045511722564697266 seconds
Simple Classifier (nonlinear): 0.005218029022216797 seconds
AdaBoost Simple Classifier: 0.040426015853881836 seconds
AdaBoost Simple Classifier (nonlinear): 0.08140897750854492 seconds
====================
```
