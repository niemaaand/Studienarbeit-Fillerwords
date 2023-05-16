class LossInfos:
    def __init__(self, name, loss=0, accuracy=0, accuracy_seperated=None, precision=None, recall=None, f1=None, total=0, correct=0, length=0):
        self.name = name
        self.loss = loss
        self.accuracy = accuracy
        self.accuracy_seperated = accuracy_seperated
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.total = total
        self.correct = correct
        self.length = length

    def set_to_zeros(self):
        self.loss = 0
        self.accuracy = 0
        self.accuracy_seperated = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0
        self.total = 0
        self.correct = 0
