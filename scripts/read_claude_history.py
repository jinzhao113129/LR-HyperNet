import json
import sys

files = [
    "/home/jnguan/.claude/projects/-home-jnguan-JiangYouxin-ZhaoJin/61970a17-c0fe-4b50-94e6-8ee95f4ec8f0.jsonl",
    "/home/jnguan/.claude/projects/-home-jnguan-JiangYouxin-ZhaoJin/bf12735d-96f9-466c-95d5-ded05008d2d4.jsonl",
]

for fpath in files:
    print(f"\n{'='*60}")
    print(f"FILE: {fpath.split('/')[-1]}")
    print('='*60)
    try:
        with open(fpath) as f:
            lines = f.readlines()
        print(f"Total lines: {len(lines)}")
        # Find last assistant message with text
        for line in reversed(lines):
            try:
                d = json.loads(line)
                if d.get('type') == 'assistant':
                    msg = d.get('message', {})
                    content = msg.get('content', [])
                    for c in content:
                        if c.get('type') == 'text':
                            text = c.get('text', '')
                            if len(text) > 100:
                                print(text[:5000])
                                break
                    else:
                        continue
                    break
            except:
                pass
    except Exception as e:
        print(f"Error: {e}")
