import os
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from shell.vector import vectorize
import numpy as np
import unicodedata
import nltk

def loadxml(filename, tokenizer):
    root = ET.parse(filename).getroot()
    dataset = []
    for instance in root:
        id = instance.attrib["id"]
        scenario = tokenizer.tokenize(instance.attrib["scenario"])
        passage = tokenizer.tokenize(instance.find("text").text)
        for questioninfo in instance.find("questions"):
            questionid = questioninfo.attrib["id"]
            question = tokenizer.tokenize(questioninfo.attrib["text"])
            answers = []
            label = 0
            for (i, answer) in enumerate(questioninfo):
                answers.append(answer.attrib["text"])
                if i == 1 and answer.attrib["correct"] == "True":
                    label = 1
            answer1 = tokenizer.tokenize(answers[0])
            answer2 = tokenizer.tokenize(answers[1])
            data = [questionid, scenario, passage, question, answer1, answer2, label]
            dataset.append(data)
    return dataset

class ReaderDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.tokenizer = tokenizer
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return vectorize(self.examples[index], self.tokenizer)

class Dictionary(object):
    NULL = '<NULL>'
    UNK = '<UNK>'
    START = 2

    @staticmethod
    def normalize(token):
        return unicodedata.normalize('NFD', token)

    def __init__(self):
        self.tok2ind = {self.NULL: 0, self.UNK: 1}
        self.ind2tok = {0: self.NULL, 1: self.UNK}

    def __len__(self):
        return len(self.tok2ind)

    def __iter__(self):
        return iter(self.tok2ind)

    def __contains__(self, key):
        if type(key) == int:
            return key in self.ind2tok
        elif type(key) == str:
            return self.normalize(key) in self.tok2ind

    def __getitem__(self, key):
        if type(key) == int:
            return self.ind2tok.get(key, self.UNK)
        if type(key) == str:
            return self.tok2ind.get(self.normalize(key),
                                    self.tok2ind.get(self.UNK))

    def __setitem__(self, key, item):
        if type(key) == int and type(item) == str:
            self.ind2tok[key] = item
        elif type(key) == str and type(item) == int:
            self.tok2ind[key] = item
        else:
            raise RuntimeError('Invalid (key, item) types.')

    def add(self, token):
        token = self.normalize(token)
        if token not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[token] = index
            self.ind2tok[index] = token

    def tokens(self):
        """Get dictionary tokens.

        Return all the words indexed by this dictionary, except for special
        tokens.
        """
        tokens = [k for k in self.tok2ind.keys()
                  if k not in {'<NULL>', '<UNK>'}]
        return tokens

    @property
    def unk_id(self):
        return self.tok2ind[self.UNK]

    @property
    def rand_id(self):
        return np.random.randint(2, len(self.tok2ind))

def build_word_dict(examples):
    """Return a dictionary from question and document words in
    provided examples.
    """
    word_dict = Dictionary()
    for ex in examples:
        questionid, scenario, passage, questiontext, answer1, answer2, label = ex
        info = passage + " " + answer1 + " " + answer2 + " " + questiontext
        for w in info.split(" "):
            word_dict.add(w)
    return word_dict