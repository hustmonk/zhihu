import urllib.request
import urllib.parse
import json

class Fanyi360(object):
    def fanyi(self, txt, english):
        url = 'http://fanyi.so.com/index/search'

        data = {
            'query': txt,
            'eng': english
        }
        data = urllib.parse.urlencode(data).encode('utf - 8')
        wy = urllib.request.urlopen(url, data)
        html = wy.read().decode('utf - 8')
        ta = json.loads(html)
        print(ta['data']['fanyi'])
        return ta['data']['fanyi']

    def bifanyi(self, txt):
        txt = self.fanyi(txt, "1")
        txt = self.fanyi(txt, "0")
        return txt

if __name__ == "__main__":
    fanyi = Fanyi360()
    result = fanyi.bifanyi(u"My wife and I decided that our home was getting way too small for our growing family ")
