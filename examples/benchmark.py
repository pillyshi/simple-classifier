# Referrence:
# http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

import time
import argparse
import warnings
import numpy as np
from sklearn.datasets import (
    make_classification,
    make_circles, make_moons
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from simple_classifier import SimpleClassifier


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-state', type=int, default=None)
    return parser.parse_args()


def get_datasets(random_state):
    X, y = make_classification(
        n_features=2, n_redundant=0, n_informative=2,
        random_state=1, n_clusters_per_class=1)
    rnd = np.random.RandomState(random_state)
    X += 2 * rnd.uniform(size=X.shape)
    linearly_separable = (X, y)
    return [
        ('moons', make_moons(noise=0.3, random_state=random_state)),
        ('circles', make_circles(noise=0.2, factor=0.5, random_state=random_state)),
        ('linearly separable', linearly_separable)
    ]


def nonlinearize(classifier, random_state):
    return Pipeline([
        ('transformer', RBFSampler(random_state=random_state)),
        ('classifier', classifier)
    ])


def get_classifiers(random_state):
    names = [
        'Logistic Regression',
        'Logistic Regression (nonlinear)',
        'AdaBoost Classifier',
        'AdaBoost Classifier (nonlinear)',
        'Simple Classifier',
        'Simple Classifier (nonlinear)',
        'AdaBoost Simple Classifier',
        'AdaBoost Simple Classifier (nonlinear)',
    ]
    classifiers = [
        LogisticRegression(random_state=random_state),
        nonlinearize(LogisticRegression(random_state=random_state), random_state),
        AdaBoostClassifier(random_state=random_state),
        nonlinearize(AdaBoostClassifier(random_state=random_state), random_state),
        SimpleClassifier(transformer='mean'),
        nonlinearize(SimpleClassifier('mean'), random_state),
        AdaBoostClassifier(SimpleClassifier(transformer='mean-random', random_state=random_state), algorithm='SAMME', random_state=random_state),
        nonlinearize(AdaBoostClassifier(SimpleClassifier(transformer='mean-random', random_state=random_state), algorithm='SAMME', random_state=random_state), random_state)
    ]
    params = [
        {'C': np.logspace(-4, 1, 20)},
        {'transformer__gamma': np.logspace(-4, 1, 20), 'classifier__C': np.logspace(-4, 1, 20)},
        {},
        {'transformer__gamma': np.logspace(-4, 1, 20)},
        {},
        {'transformer__gamma': np.logspace(-4, 1, 20)},
        {'base_estimator__sigma': np.logspace(-4, 0, 20)},
        {'transformer__gamma': np.logspace(-4, 1, 20), 'classifier__base_estimator__sigma': np.logspace(-4, 0, 20)},
    ]
    return zip(names, classifiers, params)


def main(args):
    print('The performance compariton:')
    print('-' * 20)
    for dataset_name, (X, y) in get_datasets(args.random_state):
        print('dataset={}, n_samples={}, n_features={}'.format(dataset_name, X.shape[0], X.shape[1]))
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.random_state)
        for name, classifier, params in get_classifiers(args.random_state):
            gs = GridSearchCV(classifier, params, scoring='accuracy', cv=cv, error_score=0).fit(X, y)
            print('{}: {}'.format(name, gs.best_score_))
        print('=' * 20)
    print()
    print('The computational time comparison:')
    print('-' * 20)
    for dataset_name, (X, y) in get_datasets(args.random_state):
        print('dataset={}, n_samples={}, n_features={}'.format(dataset_name, X.shape[0], X.shape[1]))
        for name, classifier, params in get_classifiers(args.random_state):
            start = time.time()
            classifier.fit(X, y)
            print('{}: {} seconds'.format(name, time.time() - start))
        print('=' * 20)



if __name__ == '__main__':
    args = get_args()
    main(args)
