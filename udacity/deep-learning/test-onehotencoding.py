import logistic_classifier as lc

labels = ['a', 'b', 'c', 'd']
scores = [[0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]]
print(lc.oneHotEncoding(labels, scores))
