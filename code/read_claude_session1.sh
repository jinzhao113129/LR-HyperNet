#!/bin/bash
cd /home/jnguan/JiangYouxin/ZhaoJin
echo '========== 会话1: ed69ede0 (2.7MB, 12:46) =========='
/home/jnguan/.miniconda/envs/bhli/bin/python3 -c "
import json
with open('/home/jnguan/.claude/projects/-home-jnguan-JiangYouxin-ZhaoJin/ed69ede0-7ae7-402a-bf99-f6969cb7b369.jsonl') as f:
    lines = f.readlines()
print(f'总行数: {len(lines)}')
for line in lines:
    try:
        obj = json.loads(line)
        msg_type = obj.get('type','')
        if msg_type == 'user':
            content = obj.get('message',{}).get('content','')
            if isinstance(content, list):
                text = ' '.join([c.get('text','')[:200] for c in content if c.get('type')=='text'])
            else:
                text = str(content)[:200]
            print(f'\n[USER] {text[:150]}')
        elif msg_type == 'assistant':
            content = obj.get('message',{}).get('content',[])
            if isinstance(content, list):
                for c in content:
                    if c.get('type') == 'text':
                        print(f'\n[ASSISTANT] {c[\"text\"][:3000]}')
    except: pass
" 2>&1 | tail -80
