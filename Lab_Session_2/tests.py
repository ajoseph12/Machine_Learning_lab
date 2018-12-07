from passive_agressive import PASVM
import numpy as np
from sklearn.datasets import make_classification, load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

import time
from tqdm import tqdm
plt.rcParams['figure.figsize'] = (15, 8)
plt.style.use('ggplot')


def preprocess_target(y):
    y[y == 0] = -1
    return y


def plot_2d(X, y):

    plt.scatter(X[y == -1, 0], X[y == -1, 1], c="y")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="c")
    plt.show()


def plot_pair(X, y_truth, y_hat):

    titles = ["Groundtruth", "Predicted"]

    for i, y in enumerate([y_truth, y_hat], 1):
        plt.subplot(1, 2, i)
        plt.scatter(X[y == -1, 0], X[y == -1, 1], c="y")
        plt.scatter(X[y == 1, 0], X[y == 1, 1], c="c")
        plt.title(titles[i-1])
    plt.show()


def compare_svm(num_instances):

    # synthetic data
    X, y = make_classification(n_samples=num_instances,
                               flip_y=0.001, n_features=2, n_informative=1, n_redundant=0, n_clusters_per_class=1, random_state=42, class_sep=0.99)
    X, X_test, y, y_test = train_test_split(X, y)
    y = preprocess_target(y)
    y_test = preprocess_target(y_test)
    # custom implementation
    tick = time.time()
    svm = PASVM(C=1, relaxation="first")
    svm.fit(X, y)
    y_hat = svm.predict(X_test)
    accuracy_pa_svm = accuracy_score(y_test, y_hat)
    time_elapsed_pa_svm = time.time() - tick
    # scikit learn svm
    tick = time.time()
    svm = SVC(kernel='linear')
    svm.fit(X, y)
    y_hat = svm.predict(X_test)
    accuracy_sk_svm = accuracy_score(y_test, y_hat)

    time_elapsed_sk_svm = time.time() - tick

    print("TIME: Sklearn {:.3}; Custom {:.3};".format(
        time_elapsed_sk_svm, time_elapsed_pa_svm))

    print("ACCURACY: Sklearn {:.3}; Custom {:.3};".format(
        accuracy_sk_svm, accuracy_pa_svm))

    return time_elapsed_pa_svm, time_elapsed_sk_svm, accuracy_pa_svm, accuracy_sk_svm


def compare_runtime(from_, to, step, plot=True):
    runtime_log_pa = []
    runtime_log_sk = []
    for num_examples in range(from_, to, step):
        pa_svm, sk_svm, *_ = compare_svm(num_examples)
        runtime_log_pa.append(pa_svm)
        runtime_log_sk.append(sk_svm)

    if plot:
        placeholder = list(range(from_, to, step))
        plt.title("Running time batch Scikit SVC vs. Online Inplimentation")
        plt.ylabel("Running time")
        plt.xlabel("Number of training examples")
        plt.plot(placeholder, runtime_log_pa,
                 c="y", label="Running time Custom")
        plt.plot(placeholder, runtime_log_sk, c="c",
                 label="Running time sklearn")

        plt.legend()
        plt.show()


def test_online():

    X, y = shuffle(*load_breast_cancer(return_X_y=True))
    X = MinMaxScaler().fit_transform(X)
    y = preprocess_target(y)

    y_svc = []
    y_classic = []
    y_first = []
    y_second = []

    classic = PASVM()
    first = PASVM(C=.06, relaxation="first")
    second = PASVM(C=0.03, relaxation="second")

    classifiers = [classic, first, second]
    predictions = [y_classic, y_first, y_second]

    for i in range(X.shape[0]):
        for classifier, prediction in zip(classifiers, predictions):
            prediction.append(classifier.predict(X[i].reshape(1, -1))[0][0])
            classifier.fit(X[i, :].reshape(1, -1), y[i].reshape(-1, 1))

    accuracies = []

    for prediction in predictions:
        accuracies.append(accuracy_score(y, prediction))

    return accuracies


def optimize_custom(n_trials):
    best = {
        "classic": {"acc": 0, "C": None},
        "first": {"acc": 0, "C": None},
        "second": {"acc": 0, "C": None}}
    X, y = shuffle(*load_breast_cancer(return_X_y=True))
    X = MinMaxScaler().fit_transform(X)
    y = preprocess_target(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    svm = PASVM()
    svm.fit(X_train, y_train)
    y_hat = svm.predict(X_test)
    acc = accuracy_score(y_test, y_hat)
    best['classic']['acc'] = acc

    c_values = 10**np.random.uniform(-4, -0.1, size=n_trials)

    for relaxation in ["first", "second"]:
        for C in tqdm(c_values):
            acc = []
            kfold = KFold(n_splits=10, shuffle=False, random_state=42)
            for train_idx, test_idx in tqdm(kfold.split(X)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                svm = PASVM(C=C, relaxation=relaxation)
                svm.fit(X_train, y_train)
                y_hat = svm.predict(X_test)
                acc = accuracy_score(y_test, y_hat)
            acc = np.mean(acc)
            if best[relaxation]['acc'] < acc:
                best[relaxation]['acc'] = acc
                best[relaxation]['C'] = C

    return best


def optimize_sklearn(n_trials):
    best = {"C": 0,
            "acc": 0}
    X, y = shuffle(*load_breast_cancer(return_X_y=True))
    X = MinMaxScaler().fit_transform(X)
    y = preprocess_target(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    c_values = 10**np.random.uniform(-4, -0.1, size=n_trials)
    for C in tqdm(c_values):
        svm = SVC(C=C, kernel="linear")
        svm.fit(X_train, y_train)
        y_hat = svm.predict(X_test)
        acc = accuracy_score(y_test, y_hat)
        if best['acc'] < acc:
            best['acc'] = acc
            best['C'] = C
    return best


def cross_validate_results(k, percentage_random):
    X, y = shuffle(*load_breast_cancer(return_X_y=True))
    X = MinMaxScaler().fit_transform(X)
    y = preprocess_target(y)
    accuracy = {
        "sklearn": [],
        "classic": [],
        "first": [],
        "second": [],
        "rbf": [],
        'poly': []
    }
    accuracy_noise = {
        "sklearn": [],
        "classic": [],
        "first": [],
        "second": []
    }

    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    for train_idx, test_idx in tqdm(kfold.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        sk_svm = SVC(C=0.8859)
        sk_svm.fit(X_train, y_train)
        y_hat = sk_svm.predict(X_test)
        accuracy['sklearn'].append(accuracy_score(y_test, y_hat))

        classic = PASVM()
        classic.fit(X_train, y_train)
        y_hat = classic.predict(X_test)
        accuracy['classic'].append(accuracy_score(y_test, y_hat))

        rbf = PASVM(kernel_type='rbf')
        rbf.fit(X_train, y_train)
        y_hat = rbf.predict(X_test)
        accuracy['rbf'].append(accuracy_score(y_test, y_hat))

        poly = PASVM(kernel_type='poly', degree=3)
        poly.fit(X_train, y_train)
        y_hat = poly.predict(X_test)
        accuracy['poly'].append(accuracy_score(y_test, y_hat))

        first = PASVM(C=0.065, relaxation="first")
        first.fit(X_train, y_train)
        y_hat = first.predict(X_test)
        accuracy['first'].append(accuracy_score(y_test, y_hat))

        first = PASVM(C=0.032, relaxation="second")
        first.fit(X_train, y_train)
        y_hat = first.predict(X_test)
        accuracy['second'].append(accuracy_score(y_test, y_hat))

        num_flipped_examples = int(X_train.shape[0]*percentage_random)
        y_train = y_train.reshape(-1, 1)
        y_flipped = y_train[:num_flipped_examples]
        y_flipped[y_flipped == 1] = 0
        y_flipped[y_flipped == -1] = 1
        y_flipped[y_flipped == 0] = -1
        y_train[:num_flipped_examples] = y_flipped
        y_train = y_train.reshape(-1)
        sk_svm = SVC(C=0.8859)
        sk_svm.fit(X_train, y_train)
        y_hat = sk_svm.predict(X_test)
        accuracy_noise['sklearn'].append(accuracy_score(y_test, y_hat))

        classic = PASVM()
        classic.fit(X_train, y_train)
        y_hat = classic.predict(X_test)
        accuracy_noise['classic'].append(accuracy_score(y_test, y_hat))

        first = PASVM(C=0.065, relaxation="first")
        first.fit(X_train, y_train)
        y_hat = first.predict(X_test)
        accuracy_noise['first'].append(accuracy_score(y_test, y_hat))

        first = PASVM(C=0.032, relaxation="second")
        first.fit(X_train, y_train)
        y_hat = first.predict(X_test)
        accuracy_noise['second'].append(accuracy_score(y_test, y_hat))

    accuracy['sklearn'] = np.mean(accuracy['sklearn'])
    accuracy['classic'] = np.mean(accuracy['classic'])
    accuracy['first'] = np.mean(accuracy['first'])
    accuracy['second'] = np.mean(accuracy['second'])

    accuracy_noise['sklearn'] = np.mean(accuracy_noise['sklearn'])
    accuracy_noise['classic'] = np.mean(accuracy_noise['classic'])
    accuracy_noise['first'] = np.mean(accuracy_noise['first'])
    accuracy_noise['second'] = np.mean(accuracy_noise['second'])

    return accuracy, accuracy_noise


if __name__ == "__main__":
    # print(optimize_custom(500))

    print(cross_validate_results(20, .5))
