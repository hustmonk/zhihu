import xml.etree.ElementTree as ET
from nltk.tokenize import sent_tokenize

filename = "data/train-data.xml"

root = ET.parse(filename).getroot()
dataset = []
fout = open("data/train-data.txt", "w")

for instance in root:
    id = instance.attrib["id"]
    scenario = instance.attrib["scenario"]
    passage = instance.find("text").text
    fout.write("%s_s\t%s\n" % (id, scenario))
    for (i, sentence) in enumerate(sent_tokenize(passage)):
        fout.write("%s_p_%d\t%s\n" % (id, i, sentence))
    for questioninfo in instance.find("questions"):
        questionid = id + "_" + questioninfo.attrib["id"]

        question = questioninfo.attrib["text"]
        fout.write("%s_q\t%s\n" % (questionid, question))

        answers = []
        label = 0
        for (i, answer) in enumerate(questioninfo):
            answertext = answer.attrib["text"]
            fout.write("%s_a_%d\t%s\n" % (questionid, i, answertext))

fout.close()