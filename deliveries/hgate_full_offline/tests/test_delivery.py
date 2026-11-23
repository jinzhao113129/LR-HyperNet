"""本地CPU验收；不把CPU通过写成四卡CUDA验证通过。"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (RECIPE, build_model, check_resume, classification_loss,
                    dump_json, restore_rng, rng_state, seed_all, sha256, verify_frozen)
from data_io import ExactEvalSampler, check_rows, inspect_data
from prepare_data import (append_row, convert_record, create_h5, read_bin,
                          record_list, window_events)
from seal_transfer import seal, verify
from stage2_pipeline import ThreeWindowHead, grouped_features


def binary_fixture(path: Path) -> None:
    """生成三个窗口内合法5字节事件，不下载真实数据。"""
    times = np.concatenate([begin * 1000 + np.arange(160) * 100 for begin in (35, 135, 235)])
    rows = np.zeros((len(times), 5), dtype=np.uint8)
    rows[:, 0] = np.arange(len(times)) % 240
    rows[:, 1] = np.arange(len(times)) % 180
    rows[:, 2] = ((np.arange(len(times)) % 2) << 7) | ((times >> 16) & 127)
    rows[:, 3], rows[:, 4] = (times >> 8) & 255, times & 255
    path.write_bytes(rows.tobytes())


class DeliveryTests(unittest.TestCase):
    """覆盖数据、恢复、模型接线和离线完整性。"""

    def setUp(self) -> None:
        """限制CPU线程并固定随机流。"""
        torch.set_num_threads(2)
        seed_all(42)

    def test_frozen_model_and_gradients(self) -> None:
        """真实三阶段门控前向反向，确认未启用远程分支。"""
        verify_frozen()
        model = build_model(small=True)
        logits = model(torch.rand(2, 4, 128))
        classification_loss(logits, torch.tensor([1, 2])).backward()
        self.assertEqual(tuple(logits.shape), (2, 101))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertFalse(any('LocalRemote' in type(g).__name__ for g in model.local_grouper_list))
        for gate in model.hyper_gates:
            self.assertGreater(float(gate.in_proj.weight.grad.abs().sum()), 0)
        sync = build_model(sync_bn=True, small=True)
        self.assertTrue(any(isinstance(m, torch.nn.SyncBatchNorm) for m in sync.modules()))

    def test_rng_checkpoint_next_step(self) -> None:
        """保存后继续一步与重建恢复后一步逐参数一致，包括AdamW状态。"""
        model = build_model(small=True)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.001)
        x, y = torch.rand(2, 4, 128), torch.tensor([1, 2])
        self.step(model, optimizer, x, y)
        net, opt, state = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()), rng_state()
        self.step(model, optimizer, x, y)
        expected = copy.deepcopy(model.state_dict())
        other = build_model(small=True)
        other.load_state_dict(net, strict=True)
        second = torch.optim.AdamW((p for p in other.parameters() if p.requires_grad), lr=.001)
        second.load_state_dict(opt)
        restore_rng(state)
        self.step(other, second, x, y)
        for name, value in other.state_dict().items():
            self.assertTrue(torch.equal(value, expected[name]), name)

    def step(self, model: object, optimizer: object, x: torch.Tensor, y: torch.Tensor) -> None:
        """执行一次真实模型优化，供恢复语义测试复用。"""
        optimizer.zero_grad(set_to_none=True)
        classification_loss(model(x), y).backward()
        optimizer.step()

    def test_resume_contract_and_schedule(self) -> None:
        """恢复合同拒绝改配方/数据/world，Cosine状态严格往返。"""
        saved = dict(args=dict(RECIPE), world_size=4, rng_states=[{}] * 4, data_fingerprint='abc')
        check_resume(saved, dict(RECIPE), 4, 'abc')
        for config, world, fp in [(dict(RECIPE, seed=7), 4, 'abc'), (RECIPE, 1, 'abc'), (RECIPE, 4, 'wrong')]:
            with self.assertRaises(ValueError):
                check_resume(saved, config, world, fp)
        a = torch.nn.Linear(2, 1)
        opt = torch.optim.AdamW(a.parameters(), lr=.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 150)
        opt.step()
        scheduler.step()
        saved_opt, saved_sched = copy.deepcopy(opt.state_dict()), scheduler.state_dict()
        opt2 = torch.optim.AdamW(torch.nn.Linear(2, 1).parameters(), lr=.001)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, 150)
        opt2.load_state_dict(saved_opt)
        scheduler2.load_state_dict(saved_sched)
        self.assertEqual(scheduler.get_last_lr(), scheduler2.get_last_lr())

    def test_eval_sampler_no_padding(self) -> None:
        """各种不能整除四卡的长度均无重复且无遗漏，包括空rank。"""
        for count in (1, 3, 7, 5226):
            groups = [list(ExactEvalSampler(count, rank, 4)) for rank in range(4)]
            result = [i for g in groups for i in g]
            self.assertEqual(sorted(result), list(range(count)))
            self.assertEqual(len(set(result)), count)

    def test_bin_windows_reproducible(self) -> None:
        """二进制解析及同记录同种子处理结果逐元素一致，三窗元数据齐全。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary_fixture(root / 'record.bin')
            points = read_bin(root / 'record.bin')
            self.assertEqual(points.shape, (480, 4))
            self.assertGreater(len(window_events(points, 1)), 100)
            args = argparse.Namespace(seed=42, num_points=32)
            row = dict(id=7, path='record.bin', label=0, split='train')
            for i in (1, 2):
                with create_h5(root / f'{i}.h5', 32) as out:
                    self.assertEqual(convert_record(root, row, out, args), [])
            with h5py.File(root / '1.h5') as a, h5py.File(root / '2.h5') as b:
                for key in a:
                    np.testing.assert_array_equal(a[key][:], b[key][:])
                self.assertEqual(len(a['label']), 9)
                check_rows(a, 'train', dict(records=[row], skipped=[]))

    def test_record_split_freeze_and_leak_rejection(self) -> None:
        """原始记录分割稳定；外部划分少文件或重复文件必须拒绝。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for i in range(101):
                folder = root / f'class{i:03d}'
                folder.mkdir()
                for j in range(2):
                    (folder / f'{j}.bin').write_bytes(b'12345')
            a, rows = record_list(root, 42, None)
            self.assertEqual((a, rows), record_list(root, 42, None))
            train = {r['path'] for r in rows if r['split'] == 'train'}
            test = {r['path'] for r in rows if r['split'] == 'test'}
            self.assertFalse(train & test)
            dump_json(root / 'split.json', rows)
            self.assertEqual((a, rows), record_list(root, 3, root / 'split.json'))
            dump_json(root / 'bad.json', rows[:-1])
            with self.assertRaises(ValueError):
                record_list(root, 42, root / 'bad.json')

    def test_metadata_completeness_and_duplicate(self) -> None:
        """缺格、重复格、测试增强和同记录错误标签都不能静默通过。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = dict(id=1, label=0, split='test')
            meta = dict(records=[row], skipped=[])
            with create_h5(root / 'x.h5', 32) as out:
                append_row(out, np.zeros((32, 4)), row, 0, 0)
                with self.assertRaises(ValueError):
                    check_rows(out, 'test', meta)
                meta['skipped'] = [dict(record_id=1, window=w, variant=0, split='test') for w in (1, 2)]
                check_rows(out, 'test', meta)
                append_row(out, np.zeros((32, 4)), row, 0, 0)
                with self.assertRaises(ValueError):
                    check_rows(out, 'test', meta)

    def test_h5_dynamic_counts_and_finite(self) -> None:
        """允许与历史行数不同但清单完整的101类数据，拒绝后续H5篡改。"""
        with tempfile.TemporaryDirectory() as temporary:
            root, records, files = Path(temporary), [], {}
            for split, offset in [('train', 0), ('test', 101)]:
                with create_h5(root / f'{split}.h5', 10240) as out:
                    for label in range(101):
                        row = dict(id=offset + label, label=label, split=split)
                        records.append(row)
                        for w in range(3):
                            for v in (range(3) if split == 'train' else range(1)):
                                append_row(out, np.zeros((10240, 4), dtype='f4'), row, w, v)
                    files[split] = dict(rows=len(out['label']))
                files[split]['sha256'] = sha256(root / f'{split}.h5')
            dump_json(root / 'dataset_manifest.json', dict(num_points=10240, classes=list(range(101)),
                      records=records, files=files, skipped=[], split_provenance='synthetic_fixture'))
            self.assertEqual(inspect_data(root)['rows'], dict(train=909, test=303))
            with h5py.File(root / 'test.h5', 'a') as out:
                out['data'][0, 0, 0] = float('nan')
            with self.assertRaises(ValueError):
                inspect_data(root)

    def test_transfer_and_stage2(self) -> None:
        """离线包篡改被拒绝，三窗头梯度与按元数据分组可运行。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'file.txt').write_text('ok')
            seal(root)
            verify(root)
            (root / 'file.txt').write_text('changed')
            with self.assertRaises(ValueError):
                verify(root)
            with h5py.File(root / 'feat.h5', 'x') as out:
                for key, value in dict(feat=np.ones((3, 1080), dtype='f4'), label=[1]*3,
                                       mark=[9]*3, window=[2, 0, 1], variant=[0]*3).items():
                    out[key] = value
            x, y, ids = grouped_features(root / 'feat.h5', False)
            self.assertEqual(x.shape, (1, 3, 1080))
            self.assertEqual(ids.tolist(), [9])
        head = ThreeWindowHead()
        loss = classification_loss(head(torch.rand(2, 3, 1080)), torch.tensor([1, 2]))
        loss.backward()
        self.assertGreater(float(head.attention.weight.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
