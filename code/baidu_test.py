"""
百度网盘分享链接下载测试脚本（标准库 urllib 版）
尝试从 SECNet 官方分享链接获取 N-Caltech101 processed data
"""
import re
import json
import urllib.request
import urllib.parse

SHARE_URL = 'https://pan.baidu.com/s/1plQ7YyjxYRq14wWvrZu10Q?pwd=u6m1'
PWD = 'u6m1'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
}

# cookie 管理
import http.cookiejar
cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
opener.addheaders = list(HEADERS.items())

def get(url, timeout=30):
    req = urllib.request.Request(url)
    return opener.open(req, timeout=timeout).read().decode('utf-8', 'ignore')

print('=== 1. 访问分享页 ===')
try:
    html = get(SHARE_URL)
    print('页面长度:', len(html))
    print('cookies:', [(c.name, c.value[:20]) for c in cj])
except Exception as e:
    print('访问失败:', e)
    raise SystemExit

# 解析 yunData
m = re.search(r'window\.yunData\s*=\s*({.*?});?\s*</script>', html, re.S)
if not m:
    m = re.search(r'yunData\s*=\s*({.*?})\s*</script>', html, re.S)
if m:
    try:
        yun = json.loads(m.group(1))
        print('yunData keys:', list(yun.keys())[:20])
        for k in ['sign', 'timestamp', 'bdstoken', 'shareid', 'uk', 'sekey', 'randsk', 'surised', 'serverTime']:
            if k in yun:
                v = yun[k]
                print(f'  {k}: {str(v)[:100]}')
    except Exception as e:
        print('yunData parse error:', e)
        idx = html.find('yunData')
        print('raw snippet:', html[idx:idx+600])
else:
    print('未找到 yunData')
    idx = html.find('yunData')
    if idx >= 0:
        print('snippet:', html[max(0,idx-100):idx+600])
    else:
        print('页面开头:', html[:500])
