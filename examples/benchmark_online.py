import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.datasets import (
    make_classification,
    make_circles, make_moons
)

from simple_classifier import SimpleClassifier


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='iris')
    parser.add_argument('--random-state', type=int, default=None)
    return parser.parse_args()


def get_datasets(name, random_state):
    if name == 'iris':
        return datasets.load_iris(return_X_y=True)
    elif name == 'wine':
        wine = datasets.load_wine()
        return wine.data, wine.target
    elif name == 'digits':
        digits = datasets.load_digits()
        return digits.data, digits.target
    elif name == 'covtype':
        covtype = datasets.fetch_covtype()
        return covtype.data, covtype.target
    elif name == 'breast_cancer':
        breast_cancer = datasets.load_breast_cancer()
        return breast_cancer.data, breast_cancer.target
    assert()


def main(args):
    names = ['SGD', 'SC-mean', 'SC-SRP']
    models = [
        SGDClassifier(random_state=args.random_state),
        SimpleClassifier(transformer='mean', random_state=args.random_state),
        SimpleClassifier(transformer='SRP', random_state=args.random_state)
    ]
    batch_size = 10

    X, y = get_datasets(args.dataset, args.random_state)
    classes = np.unique(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=args.random_state)
    for name, model in zip(names, models):
        n = 0
        starts = np.arange(0, len(y_train), batch_size)
        scores = np.zeros(len(starts))
        for i, start in enumerate(starts):
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            model.partial_fit(X_batch, y_batch, classes=classes)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            n += len(y_batch)
            scores[i] = accuracy
        plt.plot(starts, scores)
    plt.legend(names)
    plt.title('dataset={}, batch_size={}'.format(args.dataset, batch_size))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    args = get_args()
    main(args)
