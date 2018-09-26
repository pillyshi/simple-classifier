import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, make_moons, make_circles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from simple_classifier import SimpleClassifier


def plot(X, y, clf):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    clf.fit(X, y)
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    colors = np.array(['blue', 'red'])
    sns.set()
    plt.scatter(X[:, 0], X[:, 1], c=colors[y])
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='green')
    plt.tight_layout()
    plt.show()


def load(dataset, random_state, n_samples):
    datasets = {
        'iris': lambda: load_iris(return_X_y=True),
        'moons': lambda: make_moons(noise=0.05, random_state=random_state, n_samples=n_samples),
        'circles': lambda: make_circles(noise=0.05, factor=0.5, random_state=random_state, n_samples=n_samples)
    }
    if dataset not in datasets:
        assert()
    return datasets[dataset]()


def main(args):
    X, y = load(args.dataset, args.random_state, args.n_samples)

    names = ['SimpleClassifier (SRP)', 'SimpleClassifier (mean)', 'AdaBoost (SRP)', 'AdaBoost (mean-random)']
    classifiers = [
        SimpleClassifier(transformer='SRP', random_state=args.random_state),
        SimpleClassifier(transformer='mean'),
        AdaBoostClassifier(
            SimpleClassifier(transformer='SRP', random_state=args.random_state),
            algorithm='SAMME', n_estimators=args.n_estimators, random_state=args.random_state),
        AdaBoostClassifier(
            SimpleClassifier(transformer='mean-random', random_state=args.random_state),
            algorithm='SAMME', n_estimators=args.n_estimators, random_state=args.random_state),
    ]
    for name, classifier in zip(names, classifiers):
        pipe = Pipeline([
            ('transformer', RBFSampler(random_state=args.random_state, n_components=100)),
            ('classifier', classifier)
        ])
        print('accuracy ({}):'.format(name),
            cross_val_score(pipe, X, y, scoring='accuracy').mean())

    if args.plot:
        if args.dataset == 'iris':
            return
        plot(X, y, pipe)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='moons')
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--random-state', type=int, default=None)
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    main(args)
