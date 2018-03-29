# Helper class
class Evaluations(object):
    def __init__(self, cm):
        self.cm = cm
        self.tn, self.fp, self.fn, self.tp = self.evaluate()  # True Negative, False Positive, False Negative, True Positive

        # Inspection
        # print(cm)

    def evaluate(self):
        if self.cm.shape == (2, 2):
            return self.cm.ravel()

    def accuracy(self):
        denom = self.tp + self.tn + self.fp + self.fn
        if denom == 0:
            return 0
        x = self.tp + self.tn

        # Inspection
        # print(x)
        
        return x / denom

    def precission(self):
        denom = self.tp + self.fp
        if denom == 0:
            return 0
        return self.tp / denom

    def recall(self):
        denom = self.tp + self.fn
        if denom == 0:
            return 0
        return self.tp / denom

    def f1(self):
        prec = self.precission()
        rec = self.recall()
        denom = prec + rec
        if denom == 0:
            return 0
        return 2 * prec * rec / (prec + rec)
