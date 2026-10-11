#!/bin/bash
cd /home/jnguan/JiangYouxin/ZhaoJin
echo '========== 会话2: c94ff036 (874KB, 12:46) =========='
/home/jnguan/.miniconda/envs/bhli/bin/python3 -c "
import json
with open('/home/jnguan/.claude/projects/-home-jnguan-JiangYouxin-ZhaoJin/c94ff036-4680-43cb-934c-a43d68e03bbb.jsonl') as f:
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
            if '背景' in text or '需要你回答' in text:
                print(f'\n[USER] {text[:120]}...')
        elif msg_type == 'assistant':
            content = obj.get('message',{}).get('content',[])
            if isinstance(content, list):
                for c in content:
                    if c.get('type') == 'text':
                        t = c['text']
                        if len(t) > 50:
                            print(f'\n[ASSISTANT] {t[:4000]}')
    except: pass
" 2>&1 | tail -100
