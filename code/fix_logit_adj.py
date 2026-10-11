"""修复 logit_adj 符号 bug：pred + tau*log_prior → pred - tau*log_prior"""
with open('train_lr_hyper.py', 'r', encoding='utf-8') as f:
    content = f.read()

old = 'return cal_loss(pred + tau * log_prior, gold)'
new = 'return cal_loss(pred - tau * log_prior, gold)'

if old in content:
    content = content.replace(old, new)
    with open('train_lr_hyper.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('FIXED: logit_adj sign corrected (+ -> -)')
else:
    print('NOT FOUND: pattern already fixed or different')

# 验证
with open('train_lr_hyper.py', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        if 'log_prior' in line and 'cal_loss' in line:
            print(f'Line {i}: {line.strip()}')
