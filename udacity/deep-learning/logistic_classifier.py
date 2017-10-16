import numpy as np

def softmax(scores):
    """
        Softmax is a function that compute real number score
        into probability.
    """
    exps = [np.exp(i) for i in scores]
    sumExp = sum(exps)
    return np.array([i/sumExp for i in exps])

def oneHotEncoding(labels, scores):
    """
        One Hot Encoding is simple classifier to identify relevance
        object defined by greates probability. So, let matched object
        became 1 or absoultly true, else became 0.
    """
    tmp = 0
    highest = 0
    highestI = 0
    for i in range(len(scores)):
        tmp = softmax(scores[i])
        if tmp > highest:
            highest = tmp
            highestI = i
    return labels[highestI]
