import argparse, ast, types, sys

src = open('train_frequency_ncaltech.py', encoding='utf-8').read()
tree = ast.parse(src)
func = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'parse_args'][0]
code = compile(ast.get_source_segment(src, func), 'parse_args', 'exec')
ns = {}
exec(code, {'argparse': argparse, 'sys': sys}, ns)
sys.argv = ['train', '--freeze_fps_parameter', '--seed', '7', '--hyperbolic_selector']
args = ns['parse_args']()
print('freeze_fps_parameter =', args.freeze_fps_parameter)
print('seed =', args.seed)
print('hyperbolic_selector =', args.hyperbolic_selector)
print('scale_norm_fps =', args.scale_norm_fps)
