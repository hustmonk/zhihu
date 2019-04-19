import xml.etree.ElementTree as ET
pids = set()
ids = set()
for id in open("shell/bad.ids.txt"):
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