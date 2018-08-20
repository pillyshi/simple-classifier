import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, make_moons, make_circles
from simple_classifier import EnsembleRandomSimpleClassifier
from sklearn.multiclass import OneVsRestClassifier


def load(dataset, random_state, n_samples):
    datasets = {
        'iris': lambda : load_iris(return_X_y=True),
        'moons': lambda : make_moons(noise=0.3, random_state=random_state, n_samples=n_samples),
        'circles': lambda : make_circles(noise=0.2, factor=0.5, random_state=random_state, n_samples=n_samples)
    }
    if dataset not in datasets:
        assert()
    return datasets[dataset]()


def main(args):
    X, y = load(args.dataset, args.random_state, args.n_samples)

    n_classes = len(np.unique(y))
    if n_classes > 2:
        convert_y = False
    else:
        convert_y = True

    clf = OneVsRestClassifier(EnsembleRandomSimpleClassifier(
        random_state=args.random_state, n_estimators=args.n_estimators, convert_y=convert_y))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=args.random_state)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    print("accuracy:", accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='iris')
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--random-state', type=int, default=None)
    parser.add_argument('--n-estimators', type=int, default=1000)
    args = parser.parse_args()
    main(args)
