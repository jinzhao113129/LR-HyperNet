import torch
p = 'log/classification/secnet_full_hsel_s42_ai2/checkpoints/last_checkpoint.pth'
c = torch.load(p, map_location='cpu')
print('epoch', c.get('epoch'))
print('best_test_acc', c.get('best_test_acc'))
print('best_test_seq', c.get('best_test_seq'))