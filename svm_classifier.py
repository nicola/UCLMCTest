# Training set
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn import linear_model
from classes import storyparser, answers, Question, loadOrPredict, loadPickle
from features import bow
from sklearn.metrics import accuracy_score
from vectors import results, YVectorQA, YVectorQ
from grading import grading
from sklearn import svm

def yy(vector):
    result = []
    for y in vector:
        if y == 0:
            result+=[1,0,0,0]
        elif y == 1:
            result+=[0,1,0,0]
        elif y == 2:
            result+=[0,0,1,0]
        elif y == 3:
            result+=[0,0,0,1]
    return result

# methods = [dict(name="Baseline (BOW)", score=bow.predict, opts=None)]
def train(stories, solutions, opts=None):
    # TODO this should be imported in this way
    # features = [m["score"](stories, opts=m["opts"]) for m in methods]
    # X = [tuple(t,) for t in np.asarray(features).T]

    X = np.array(zip(
        *[loadOrPredict(dict(name=feature.__name__), stories, opts=dict(pickle=True), pickle_label=opts['trainsets']) for feature in opts["features"]]
    ))
    y = np.array(yy(loadPickle("y"+str(opts['trainsets'][0]))))
    C = 3

    return svm.SVC(kernel='linear', C=C, probability=True).fit(X, y)


def predict(stories, opts=None):

    X = np.array(zip(
        *[loadOrPredict(dict(name=feature.__name__), stories, opts=dict(pickle=True), pickle_label=opts['testsets']) for feature in opts["features"]]
    ))

    # TODO this should be loaded not calculated
    if (not opts):
        opts = {}

    if ("trainsets" not in opts):
        opts["trainsets"] = ["mc160.dev"]

    if ("train_stories" not in opts or "train_solutions" not in opts):
        opts["train_stories"] = []
        opts["train_solutions"] = []

    logreg = train(
        opts["train_stories"],
        opts["train_solutions"],
        opts=opts
    )

    return [x[1] for x in logreg.predict_proba(X)]

if __name__ == "__main__":
    testset = "mc160.dev"
    stories = list(storyparser(testset))
    solutions = list(answers(testset))
    mode = Question.SINGLE

    svm_qa(stories, solutions, mode=mode)
