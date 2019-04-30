import os, time

from tsz import Fanyi360
textfile = "data/train-data.txt"
cachefile = "data/train-data.txt.cache"
outfile = "data/train-data.out.txt"
fout = open(outfile, "w")
keys = set()
if os.path.exists(cachefile):
    for line in open(cachefile):
        arr = line.strip().split("\t")
        if len(arr) == 0 or arr[0] in keys:
            continue
        keys.add(arr[0])
        fout.write(line)

fayi = Fanyi360()
for line in open(textfile):
    key, text = line.strip().split("\t")
    if key in keys:
        continue
    try:
        result = fayi.bifanyi(text)
        time.sleep(1)
    except:
        time.sleep(10)
        continue
    if len(result) < 1:
        continue
    fout.write("%s\t%s\n" % (key, result))
    fout.flush()
fout.close()
