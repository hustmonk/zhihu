import xml.etree.ElementTree as ET
import sys
pids = set()
ids = set()

filename = "bad.ids.txt"
if len(sys.argv) > 1:
    filename = sys.argv[1]
for id in open(filename):
    id = id.strip()
    p, q = id.split("_")
    pids.add(p)
    ids.add(id)

filename = "data/dev-data.xml"
root = ET.parse(filename).getroot()
dataset = []
fout = open("dev.bad.txt", "w")
for instance in root:
    id = instance.attrib["id"]
    if id not in pids:
        continue
    fout.write(instance.find("text").text + "\n")
    for questioninfo in instance.find("questions"):
        questionid = id + "_" + questioninfo.attrib["id"]
        if questionid not in ids:
            continue
        fout.write(questioninfo.attrib["text"] + "\n")
        for (i, answer) in enumerate(questioninfo):
            fout.write(answer.attrib["correct"] + "\t" + answer.attrib["text"] + "\n")
    fout.write("\n")
fout.close()