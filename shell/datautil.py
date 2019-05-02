import os
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from shell.vector import vectorize
import numpy as np
import unicodedata

def loadxml(filename, tokenizer):
    root = ET.parse(filename).getroot()
    dataset = []
    for instance in root:
        id = instance.attrib["id"]
        scenario = tokenizer.tokenize(instance.attrib["scenario"])
        passage = tokenizer.tokenize(instance.find("text").text)
        for questioninfo in instance.find("questions"):
            questionid = id + "_" + questioninfo.attrib["id"]
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

class MockDataset():
    def __init__(self, filename, tokenizer):
        passages = {}
        self.question = {}
        self.firstanswer = {}
        self.secondanswer = {}
        for line in open(filename):
            arr = line.strip().split("\t")
            keys = arr[0].split("_")
            passageid = keys[0]
            if keys[1] == 'p':
                if passageid not in passages:
                    passages[passageid] = []
                passages[passageid].append([int(keys[2]), arr[1]])
                continue
            elif keys[1] == 's':
                continue
            tokens = tokenizer.tokenize(arr[1])
            questionid = passageid + "_" + keys[1]
            if keys[2] == 'q':
                self.question[questionid] = tokens
            elif keys[2] == 'a':
                if keys[3] == '0':
                    self.firstanswer[questionid] = tokens
                else:
                    self.secondanswer[questionid] = tokens
        self.passages = {}

        for (passageid, passageinfo) in passages.items():
            passageinfo = sorted(passageinfo)
            self.passages[passageid] = tokenizer.tokenize(" ".join([k[1] for k in passageinfo]))

    def mock(self, example):
        ratio = 0.1
        questionid, scenario, passage, question, answer1, answer2, label = example
        if questionid not in self.question:
            return example
        if np.random.rand() < ratio:
            passageid = questionid.split("_")[0]
            passage = self.passages.get(passageid, passage)
        if np.random.rand() < ratio:
            question = self.question.get(questionid, question)
        if np.random.rand() < ratio:
            answer1 = self.firstanswer.get(questionid, answer1)
        if np.random.rand() < ratio:
            answer2 = self.secondanswer.get(questionid, answer2)

        example = [questionid, scenario, passage, question, answer1, answer2, label]

        return example

class ReaderDataset(Dataset):
    def __init__(self, examples, tokenizer, mockdataset=None, training=False):
        self.tokenizer = tokenizer
        self.examples = examples
        self.mockdataset = mockdataset
        self.training = training

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        if self.training and self.mockdataset is not None:
            example = self.mockdataset.mock(example)
        return vectorize(example, self.tokenizer, self.training)

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