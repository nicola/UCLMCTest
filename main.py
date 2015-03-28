from classes import storyparser, answers, loadOrPredict, Question
from grading import grading
from vectors import results, YVectorQ
from features import bow
from fscoring import fscore160, fscore500
from hypernymy import hypbow, hypbowscore, hyp_qa_select
import classifier as svm
import logistic_regression as logreg
import svm_classifier as svmreg
import numpy as np

testsets = [
    "mc160.dev",
    "mc500.dev",
    "mc160.train",
    "mc500.train",
    ["mc160.dev", "mc160.train"],
    ["mc500.dev", "mc500.train"],
    "mc160.test",
    "mc500.test"
]

def _bowcoref(stories, opts=None):
    return bow.XVectorQA(
        stories,
        norm="question",
        score_f=bow.score,
        bow_f=bow.coref_bow
    )
def _hypbow(stories, opts=None):
    return bow.XVectorQA(
        stories,
        norm="question",
        score_f=hypbowscore,
        mode=opts["mode"] if "mode" in opts else None
    )
def _bowall1(stories, opts=None):
    return bow.XVectorQA(
        stories,
        norm="question",
        score_f=bow.scoreAll,
        select_f=bow.bow_q_select_coref,
        select_limit=3,
        bow_f=bow.coref_bowall
    )
def _bowall2(stories, opts=None):
    return bow.XVectorQA(
        stories,
        norm="question",
        score_f=bow.scoreAll,
        select_f=bow.bow_qa_select_coref,
        select_limit=3,
        bow_f=bow.coref_bowall
    )
def _hypselect(stories, opts=None):
    return bow.XVectorQA(
        stories,
        norm="question",
        score_f=hypbowscore,
        select_f=bow.bow_qa_select_coref,
        select_limit=3
    )
def _hypselect2(stories, opts=None):
    return bow.XVectorQA(
        stories,
        norm="question",
        score_f=bow.scoreAll,
        select_f=hyp_qa_select,
        select_limit=3,
        bow_f=bow.coref_bowall
    )

def _filter160(stories, opts=None):
    return bow.XVectorQA(
        stories,
        norm="question",
        score_f=fscore160,
    )

def _rtefilter160(stories, opts=None):
    return bow.XVectorQA(
        stories,
        norm="question",
        score_f=fscore160,
    )

def _rtefilter500(stories, opts=None):
    return bow.XVectorQA(
        stories,
        norm="question",
        score_f=fscore500,
    )


def _filter500(stories, opts=None):
    return bow.XVectorQA(
        stories,
        norm="question",
        score_f=fscore500,
    )

def bigMixSum(stories, opts=None):
    features = [
        _bowcoref,
        _hypbow,
        _bowall1,
        # _bowall2,
        # _hypselect,
        # _hypselect2
    ]
    vectors = [
        loadOrPredict(
            dict(name=feature.__name__),
            stories,
            opts=dict(pickle=True),
            pickle_label='mc500.dev'
        )
        for feature in features
    ]
    print vectors

    sum_v = vectors[0]
    for v in vectors[1:]:
        sum_v = np.asarray(sum_v) + np.asarray(v)

    return sum_v

methods = [
    # dict(
    #    name="_filter160",
    #    score=_filter160,
    #    opts=dict(
    #        testsets=testsets,
    #        pickle=True
    #    )
    # ),
    # dict(
    #     name="BOW NN",
    #     score=bow.predictAllNN,
    #     opts=dict(
    #         testsets=testsets
    #     )
    # ),
    # dict(
    #     name="BOW NP",
    #     score=bow.predictAllVB,
    #     opts=dict(
    #         testsets=testsets
    #     )
    # ),
    # dict(
    #     name="Baseline (BOW)",
    #     score=bow.predictAll,
    #     select_f=bow.bow_qa_select,
    #     select_limit=3,
    #     opts=dict(
    #         testsets=[
    #             ["mc160.dev", "mc160.train"],
    #             ["mc500.dev", "mc500.train"]
    #         ]
    #     )
    # ),
    # dict(
    #     name="BOW coref",
    #     score=bow.predict,
    #     opts=dict(
    #         testsets=[
    #             ["mc160.dev", "mc160.train"],
    #             ["mc500.dev", "mc500.train"]
    #         ],
    #         bow_f=bow.coref_bow,
    #         mode=Question.MULTIPLE
    #     )
    # ),
    # dict(
    #     name="Baseline (BOW) all",
    #     score=bow.predictAll,
    #     opts=dict(
    #         testsets=testsets
    #     )
    # ),
    # dict(
    #     name="SVM (BOW) train mc160train",
    #     score=svm.predict,
    #     opts=dict(
    #         trainsets=["mc160.train"],
    #         testsets=["mc160.dev"],
    #         features=[bow.predict]
    #     )
    # ),
    # dict(
    #     name="SVM (BOW) train mc500train",
    #     score=svm.predict,
    #     opts=dict(
    #         trainsets=["mc500.train"],
    #         testsets=["mc500.dev"],
    #         features=[bow.predict]
    #     )
    # ),
    # dict(
    #     name="SVM (BOW+BOWall) train mc160train",
    #     score=svm.predict,
    #     opts=dict(
    #         trainsets=["mc160.train"],
    #         testsets=["mc160.dev"],
    #         features=[bow.predict, bow.predictAll]
    #     )
    # ),
    # dict(
    #     name="SVM (BOWall+BOWcomplement) train mc160train",
    #     score=svm.predict,
    #     opts=dict(
    #         trainsets=["mc160.train"],
    #         testsets=["mc160.dev"],
    #         features=[bow.predictComplement, bow.predictAll, bow.predictAllNN, bow.predictAllVB]
    #     )
    # ),
    # dict(
    #     name="SVM (BOW+BOWall) train mc500train",
    #     score=svm.predict,
    #     opts=dict(
    #         trainsets=["mc500.train"],
    #         testsets=["mc500.dev"],
    #         features=[bow.predict, bow.predictAll]
    #     )
    # ),
    # dict(
    #     name="LogReg (BOW+BOWall) mc160train",
    #     score=logreg.predict,
    #     opts=dict(
    #         trainsets=["mc160.train"],
    #         testsets=["mc160.dev"]
    #     )
    # ),
    # dict(
    #     name="LogReg (BOW+BOWall) mc500train",
    #     score=logreg.predict,
    #     opts=dict(
    #         trainsets=["mc500.train"],
    #         testsets=["mc500.dev"]
    #     )
    # ),
    # dict(
    #     name="LogReg (BOWall+BOWComplement+BOWNN+BOWVB) mc160train",
    #     score=logreg.predict,
    #     opts=dict(
    #         features=[
    #             bow.predictAll,
    #             # bow.predictAllNN,
    #             # bow.predictAllVB,
    #             bow.predictComplement,
    #             hypbow
    #         ],
    #         trainsets=["mc160.train"],
    #         testsets=["mc160.dev"],
    #         select_f=bow.bow_qa_select,
    #         select_limit=2
    #     )
    # ),
    # dict(
    #     name="LogReg (BOWall+BOWComplement+BOWNN+BOWVB) mc500train",
    #     score=logreg.predict,
    #     opts=dict(
    #         features=[
    #             bow.predictAll,
    #             bow.predictAllNN,
    #             bow.predictAllVB,
    #             bow.predictComplement
    #         ],
    #         trainsets=["mc500.train"],
    #         testsets=["mc500.dev"]
    #     )
    # ),
    # dict(
    #     name="Hypernym BOW",
    #     score=hypbow,
    #     opts=dict(
    #         testsets=testsets
    #     )
    # ),
    # dict(
    #     name="Baseline (BOW) selection 2",
    #     score=bow.predictAll,
    #     opts=dict(
    #         testsets=testsets,
    #         select_f=bow.bow_qa_select,
    #         select_limit=2
    #     )
    # ),
    # dict(
    #     name="Hypernym BOW selection 2",
    #     score=hypbow,
    #     opts=dict(
    #         testsets=testsets,
    #         select_f=hyp_qa_select,
    #         select_limit=2
    #     )
    # ),
    # dict(
    #     name="_bowcoref",
    #     score=_bowcoref,
    #     opts=dict(
    #         testsets=testsets,
    #         pickle=True
    #     )
    # ),
    # dict(
    #     name="_hypbow",
    #     score=_hypbow,
    #     opts=dict(
    #         testsets=testsets,
    #         pickle=True
    #     )
    # ),
    # dict(
    #     name="_bowall1",
    #     score=_bowall1,
    #     opts=dict(
    #         testsets=testsets,
    #         pickle=True
    #     )
    # ),
    # dict(
    #     name="_bowall2",
    #     score=_bowall2,
    #     opts=dict(
    #         testsets=testsets,
    #         pickle=True
    #     )
    # ),
    # dict(
    #     name="_hypselect2",
    #     score=_hypselect2,
    #     opts=dict(
    #         testsets=["mc160.dev"],
    #         pickle=True
    #     )
    # ),
    # dict(
    #     name="_hypselect2",
    #     score=_hypselect2,
    #     opts=dict(
    #         testsets=testsets,
    #         pickle=True
    #     )
    # ),
    # dict(
    #     name="SVM (bigmix) mc160train",
    #     score=logreg.predict,
    #     opts=dict(
    #         features=[
    #             _bowcoref,
    #             _hypbow,
    #             _bowall1,
    #             _bowall2,
    #             _hypselect,
    #             _hypselect2,
    #             _filter160
    #         ],
    #         trainsets=["mc160.train"],
    #         testsets=["mc160.test"]
    #     )
    # ),
    # dict(
    #     name="SVM (bigmix) mc500train",
    #     score=logreg.predict,
    #     opts=dict(
    #         features=[
    #             _bowcoref,
    #             _hypbow,
    #             _bowall1,
    #             _bowall2,
    #             _hypselect,
    #             _hypselect2,
    #             _filter500
    #         ],
    #         trainsets=["mc500.train"],
    #         testsets=["mc500.test"]
    #     )
    # ),
    # dict(
    #     name="SVM (bigmix) mc160train",
    #     score=svmreg.predict,
    #     opts=dict(
    #         features=[
    #             _bowcoref,
    #             _hypbow,
    #             _bowall1,
    #             _bowall2,
    #             _hypselect,
    #             _hypselect2
    #         ],
    #         trainsets=["mc500.train"],
    #         testsets=["mc500.test"]
    #     )
    # ),
    # dict(
    #     name="SVM (bigmix) mc500train",
    #     score=svmreg.predict,
    #     opts=dict(
    #         features=[
    #             _bowcoref,
    #             _hypbow,
    #             _bowall1,
    #             _bowall2,
    #             _hypselect,
    #             _hypselect2,
    #             _filter500
    #         ],
    #         trainsets=["mc500.train"],
    #         testsets=["mc500.test"]
    #     )
    # )
]

results = {}
for method in methods:
    results[method["name"]] = {}
    for testset in testsets:
        results[method["name"]][str(testset)] = "0"

for method in methods:
    name = method["name"]
    mode = None if "mode" not in method["opts"] else method["opts"]["mode"]

    for testset in method["opts"]["testsets"]:
        stories = list(storyparser(testset))
        solutions = list(answers(testset))
        true = YVectorQ(stories, solutions, mode=mode)
        scores = loadOrPredict(method, stories, method["opts"], pickle_label=str(testset))
        print len(scores), len(true), mode
        grades = grading(scores, true)
        results[name][str(testset)] = sum(grades) / len(grades)
        print results[name][str(testset)]

print "\n"
print "| Description | " + " | ".join([str(x) for x in testsets]) + " |"
print "| " + ("--- | ---" * len(testsets)) + " |"
for method in methods:
    m_results = [results[method["name"]][str(t)] for t in testsets]
    print "| %s | %s |" % (
        method["name"],
        " | ".join([str(r) for r in m_results])
    )
