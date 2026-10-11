"""修改 train_frequency.py 以支持 N-Caltech101"""
with open('train_frequency.py', 'r') as f:
    content = f.read()

# 1. 去掉 num_category 的 choices 限制，默认 101
content = content.replace(
    "parser.add_argument('--num_category', default=10, type=int, choices=[10, 40], help='training on ModelNet10/40')",
    "parser.add_argument('--num_category', default=101, type=int, help='number of categories')"
)
# 2. 默认数据路径
content = content.replace(
    'parser.add_argument("--data_path", type=str, default=\'./data/DVSGesture/\', help="path to eyetracking_log")',
    'parser.add_argument("--data_path", type=str, default=\'./data/\', help="path to data")'
)
# 3. 默认 num_point
content = content.replace(
    "parser.add_argument('--num_point', type=int, default=1024, help='Point Number')",
    "parser.add_argument('--num_point', type=int, default=8192, help='Point Number')"
)
# 4. 设备选择支持 CPU
content = content.replace(
    "device = 'cuda'",
    "device = 'cpu' if args.use_cpu else 'cuda'"
)
content = content.replace(
    "classifier = classifier.cuda()",
    "classifier = classifier.to(device)"
)

with open('train_frequency.py', 'w') as f:
    f.write(content)
print('Modified successfully')

# 验证修改
with open('train_frequency.py', 'r') as f:
    c = f.read()
assert 'default=101' in c, 'num_category not modified'
assert 'default=8192' in c, 'num_point not modified'
assert "args.use_cpu" in c, 'cpu support not added'
print('All modifications verified')
