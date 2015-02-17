# Training set
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn import svm
from classifiers.perceptron import Perceptron
from classes import storyparser, answers, Question
from features import bow
from sklearn.metrics import accuracy_score
from vectors import results, YVectorQA, YVectorQ
from grading import grading

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report


# methods = [dict(name="Baseline (BOW)", score=bow.predict, opts=None)]
def train(stories, solutions, opts=None):
    # TODO this should be imported in this way
    # features = [m["score"](stories, opts=m["opts"]) for m in methods]
    # X = [tuple(t,) for t in np.asarray(features).T]

    X_train = np.array(zip(
        *[feature(stories) for feature in opts["features"]]
    ))
    y_train = np.array(YVectorQA(stories, solutions))

    test_stories = list(storyparser('mc160.dev'))
    test_answers = list(answers('mc160.dev'))
    X_test = np.array(zip(
        *[feature(test_stories) for feature in opts["features"]]
    ))
    y_test = np.array(YVectorQA(test_stories, test_answers))

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y,
    #     test_size=0.3,
    #     random_state=0
    # )

    tuned_parameters = [{
        'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
        'C': [1, 10, 100, 1000]
    }, {
        'kernel': ['linear'], 'C': [1, 10, 100, 1000]
    }]

    clf = svm.SVC(C=4.0, probability=True)
    fit = clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

    print "Best estimator", clf.best_estimator_

    return fit


def predict(stories, opts=None):

    X = np.array(zip(
        *[feature(stories) for feature in opts["features"]]
    ))

    # TODO this should be loaded not calculated
    if (not opts):
        opts = {}

    if ("trainsets" not in opts):
        opts["trainsets"] = ["mc160.dev"]

    if ("train_stories" not in opts or "train_solutions" not in opts):
        opts["train_stories"] = list(storyparser(opts["trainsets"]))
        opts["train_solutions"] = list(answers(opts["trainsets"]))

    svc = train(
        opts["train_stories"],
        opts["train_solutions"],
        opts=opts
    )

    return [x[1] for x in svc.predict_proba(X)]


def svm_qa(stories, solutions, mode=None):
    qa = bow.XVectorQA(stories, norm="sigmoid", sigmoid_k=10, mode=mode)
    X = np.array(zip(
        qa,
        [0] * len(qa)
    ))
    y = np.array(YVectorQA(stories, solutions, mode=mode))
    h=0.01
    C=1.0

    svc = svm.SVC(kernel='linear', C=C).fit(X, y)
    print "Single QA Prediction " + str(svc.predict(X))
    print "Single QA Actual     " + str(y)
    print "Single QA Accuracy   " + str(svc.score(X, y) * 100) + "%"
    print "Single QA Correct   " + str(accuracy_score(y, svc.predict(X), normalize=False))

    results(svc.predict(X), YVectorQ(stories, solutions, mode), verbose=True)
    grading(svc.predict(X), YVectorQ(stories, solutions, mode), verbose=True)

    # Plot
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.show()

if __name__ == "__main__":
    testset = "mc160.dev"
    stories = list(storyparser(testset))
    solutions = list(answers(testset))
    mode = Question.SINGLE

    svm_qa(stories, solutions, mode=mode)
