import string


class Preprocess:

    def __init__(self):
        self.sentence = None

    def manipulation(self, sentence):
        self.sentence = sentence.lower()
        punctuation = tuple(string.punctuation)
        self.sentence = ''.join(s for s in self.sentence if s not in punctuation)
        return self.sentence
