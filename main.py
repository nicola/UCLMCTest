import sys
from classes import storyparser, Question, answers
from grading import grading
from vectors import results, YVectorQ
from features import bow
import numpy as np

testsets = ["mc160.dev", "mc500.dev", "mc160.train"]

methods = [
    dict(
        name="Baseline (BOW)",
        score=bow.predict,
        opts=None
    )
]

results = {}
for method in methods:
    name = method["name"]
    results[name] = {}

    for testset in testsets:
        stories = list(storyparser(testset))
        solutions = list(answers(testset))
        true = YVectorQ(stories, solutions)

        scores = method["score"](stories, method["opts"])
        grades = grading(scores, true)
        results[name][testset] = sum(grades)/len(grades)

print results

print "| Description | " + " | ".join(testsets) + " |"
print "| " + ("--- | ---" * len(testsets)) + " |"
for method in methods:
    m_results = [results[method["name"]][t] for t in testsets]
    print "| %s | %s |" % (
        method["name"],
        " | ".join([str(r) for r in m_results])
    )