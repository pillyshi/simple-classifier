import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, make_moons, make_circles
from nonlinear_classifier import NonlinearClassifier
from sklearn.multiclass import OneVsRestClassifier


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
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=args.random_state)

    n_classes = len(np.unique(y))
    if n_classes > 2:
        convert_y = False
    else:
        convert_y = True

    clf = OneVsRestClassifier(NonlinearClassifier(
        n_components=X.shape[0],
        random_state=args.random_state,
        convert_y=convert_y
    ))
    clf = GridSearchCV(clf, {'estimator__sigma': np.logspace(-2, 1, 10)}).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('best params:', clf.best_params_)
    print('accuracy:', accuracy)

    if args.plot:
        if args.dataset == 'iris':
            return
        plot(X, y, clf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='iris')
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--random-state', type=int, default=None)
    parser.add_argument('--n-estimators', type=int, default=1000)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    main(args)
