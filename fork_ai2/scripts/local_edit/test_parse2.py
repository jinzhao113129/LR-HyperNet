import argparse, ast, sys

src = open('train_frequency_ncaltech.py', encoding='utf-8').read()
tree = ast.parse(src)
func = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'parse_args'][0]
code = compile(ast.get_source_segment(src, func), 'parse_args', 'exec')
ns = {}
exec(code, {'argparse': argparse, 'sys': sys}, ns)

# case1: bug版复核（双曲+缩放坐标）
sys.argv = ['train', '--hyperbolic_selector', '--hsel_scaled_coords', '--seed', '7']
a = ns['parse_args']()
assert a.hyperbolic_selector is True and a.hsel_scaled_coords is True and a.seed == 7, 'case1 FAIL'
print('case1 (bug版复核): hsel_scaled_coords =', a.hsel_scaled_coords, '| seed =', a.seed)

# case2: 修复版（只 hyperbolic_selector）
sys.argv = ['train', '--hyperbolic_selector', '--seed', '7']
b = ns['parse_args']()
assert b.hsel_scaled_coords is False, 'case2 FAIL'
print('case2 (修复版): hsel_scaled_coords =', b.hsel_scaled_coords)

# case3: 默认全关（基线行为不变）
sys.argv = ['train', '--seed', '42']
c = ns['parse_args']()
assert c.hsel_scaled_coords is False and c.hyperbolic_selector is False and c.freeze_fps_parameter is False
print('case3 (基线默认): hsel_scaled_coords =', c.hsel_scaled_coords)

# case4: freeze 组合
sys.argv = ['train', '--freeze_fps_parameter', '--seed', '42']
d = ns['parse_args']()
assert d.freeze_fps_parameter is True
print('case4 (freeze): freeze_fps_parameter =', d.freeze_fps_parameter)
print('ALL_PARSE_TESTS_PASS')
