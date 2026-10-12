import json

files = [
    "/home/jnguan/.claude/projects/-home-jnguan-JiangYouxin-ZhaoJin/ed69ede0-7ae7-402a-bf99-f6969cb7b369.jsonl",
    "/home/jnguan/.claude/projects/-home-jnguan-JiangYouxin-ZhaoJin/c94ff036-4680-43cb-934c-a43d68e03bbb.jsonl",
]

for fpath in files:
    print(f"\n{'='*60}")
    print(f"FILE: {fpath.split('/')[-1]}")
    print('='*60)
    try:
        with open(fpath) as f:
            lines = f.readlines()
        print(f"Total lines: {len(lines)}")
        # Find last 2 assistant messages with text > 500 chars
        count = 0
        for line in reversed(lines):
            if count >= 2:
                break
            try:
                d = json.loads(line)
                if d.get('type') == 'assistant':
                    msg = d.get('message', {})
                    content = msg.get('content', [])
                    for c in content:
                        if c.get('type') == 'text':
                            text = c.get('text', '')
                            if len(text) > 500:
                                print(f"\n--- Assistant message (len={len(text)}) ---")
                                print(text[:6000])
                                count += 1
                                break
            except:
                pass
    except Exception as e:
        print(f"Error: {e}")
