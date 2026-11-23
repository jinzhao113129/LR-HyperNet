"""四卡SyncBN训练单独双曲门控；支持只读验证、测速与完整epoch边界续训。"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from common import (RECIPE, build_model, check_resume, classification_loss,
                    dump_json, restore_rng, rng_state, seed_all, sha256, training_code_hash)
from data_io import EventH5, ExactEvalSampler, inspect_data


def parse_args() -> argparse.Namespace:
    """正式训练冻结配方；测速允许单卡batch16和四卡batch4公平比较。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--mode', choices=['train', 'benchmark', 'evaluate'], default='train')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accum', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--eval-batch', type=int, default=4)
    args = parser.parse_args()
    if min(args.batch_size, args.accum, args.steps, args.eval_batch) < 1 or args.workers < 0:
        parser.error('batch/accum/steps须为正，workers非负')
    if args.mode == 'evaluate' and args.checkpoint is None:
        parser.error('evaluate需要--checkpoint')
    if args.resume and args.mode != 'train':
        parser.error('--resume只用于train')
    return args


def initialize(args: argparse.Namespace) -> tuple:
    """只使用torchrun分配的卡，不改变系统设置，不终止其他作业。"""
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if not torch.cuda.is_available():
        raise RuntimeError('DDP实跑需要CUDA；CPU测试请运行tests')
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', timeout=timedelta(hours=1))
    if args.mode == 'train' and (world != 4 or args.batch_size != 4 or args.accum != 2):
        raise ValueError('正式配方固定四卡、每卡4、累积2；其他配置仅允许benchmark')
    torch.backends.cudnn.benchmark = False
    seed_all(args.seed + rank)
    if rank == 0:
        if args.resume:
            if args.resume.resolve() != (args.out / 'last_checkpoint.pth').resolve():
                raise ValueError('只能从该out下的last_checkpoint.pth恢复')
        else:
            args.out.mkdir(parents=True, exist_ok=False)
        data_report = inspect_data(args.data)
        dump_json(args.out / 'data_check.json', data_report)
    else:
        data_report = None
    shared = [data_report]
    dist.broadcast_object_list(shared, src=0)
    dist.barrier()
    return rank, local, world, shared[0]


def make_loaders(args: argparse.Namespace, rank: int, world: int) -> tuple:
    """训练丢弃不足整批的尾部；测试不补齐，保证每行恰好评估一次。"""
    train_data, test_data = EventH5(args.data / 'train.h5'), EventH5(args.data / 'test.h5')
    sampler = DistributedSampler(train_data, world, rank, seed=args.seed, drop_last=True)
    generator = torch.Generator().manual_seed(args.seed + rank)
    options = dict(num_workers=args.workers, pin_memory=True, generator=generator)
    train = DataLoader(train_data, batch_size=args.batch_size, sampler=sampler,
                       drop_last=True, **options)
    test = DataLoader(test_data, batch_size=args.eval_batch,
                      sampler=ExactEvalSampler(len(test_data), rank, world), **options)
    return train, test, sampler, generator


def train_epoch(model: DDP, loader: DataLoader, optimizer: torch.optim.Optimizer,
                accum: int, limit: int = 0) -> dict:
    """实际窗口长度归一化尾梯度；中间步不做DDP梯度归约。"""
    model.train()
    steps = min(limit, len(loader)) if limit else len(loader)
    totals = torch.zeros(3, device='cuda', dtype=torch.float64)
    start = time.perf_counter()
    for i, (data, labels, _) in enumerate(loader):
        if i >= steps:
            break
        if i % accum == 0:
            optimizer.zero_grad(set_to_none=True)
        window = min(accum, steps - (i // accum) * accum)
        finish = (i + 1) % accum == 0 or i + 1 == steps
        context = nullcontext() if finish else model.no_sync()
        with context:
            data, labels = data.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            logits = model(data)
            loss = classification_loss(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError('非有限训练loss')
            (loss / window).backward()
        if finish:
            if limit and i < accum:
                check_gate_gradients(model.module)
            optimizer.step()
        totals += torch.stack([loss.detach().double() * len(labels),
                               (logits.argmax(-1) == labels).sum().double(),
                               torch.tensor(len(labels), device='cuda', dtype=torch.float64)])
        if dist.get_rank() == 0 and (i + 1) % 100 == 0:
            print(f'batch={i + 1}/{steps} loss={loss.item():.5f}', flush=True)
    dist.all_reduce(totals)
    torch.cuda.synchronize()
    elapsed = torch.tensor(time.perf_counter() - start, device='cuda')
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return dict(loss=(totals[0] / totals[2]).item(), accuracy=(100 * totals[1] / totals[2]).item(),
                samples=int(totals[2]), seconds=elapsed.item(), batches=steps)


def check_gate_gradients(model: torch.nn.Module) -> None:
    """测速必须确认三个门控投影均有有效梯度，不能仅启动空作业。"""
    for i, gate in enumerate(model.hyper_gates):
        grad = gate.in_proj.weight.grad
        if grad is None or not torch.isfinite(grad).all() or grad.abs().sum() == 0:
            raise RuntimeError(f'stage{i}门控未获得有效梯度')


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, seed: int) -> dict:
    """绕过DDP前向以容许不同rank测试批数；同步buffer后不逐批通信。"""
    state = rng_state()
    seed_all(seed + dist.get_rank())
    for buffer in model.buffers():
        dist.broadcast(buffer, src=0)
    model.eval()
    total = torch.zeros(3, device='cuda', dtype=torch.float64)
    for data, labels, _ in loader:
        labels = labels.cuda(non_blocking=True)
        logits = model(data.cuda(non_blocking=True))
        total += torch.stack([classification_loss(logits, labels).double() * len(labels),
                              (logits.argmax(-1) == labels).sum().double(),
                              torch.tensor(len(labels), device='cuda', dtype=torch.float64)])
    dist.all_reduce(total)
    restore_rng(state)
    if int(total[2]) != len(loader.dataset):
        raise ValueError('测试样本计数不一致')
    return dict(loss=(total[0] / total[2]).item(), accuracy=(100 * total[1] / total[2]).item(),
                samples=int(total[2]))


def load_checkpoint(path: Path) -> dict:
    """仅载入自己生成且可信的checkpoint，允许Python/NumPy随机状态。"""
    return torch.load(path, map_location='cpu', weights_only=False)


def save_checkpoint(model: DDP, optimizer: object, scheduler: object, epoch: int,
                    recipe: dict, report: dict, args: argparse.Namespace, best: float,
                    loader_rng: torch.Generator, is_best: bool) -> None:
    """收集所有rank随机流并仅由rank0原子保存，保存位置为完整epoch边界。"""
    local = dict(rng=rng_state(), loader=loader_rng.get_state())
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, local)
    if dist.get_rank() == 0:
        payload = dict(net=model.module.state_dict(), args=recipe, epoch=epoch,
                       optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                       world_size=dist.get_world_size(), rng_states=states, best_test_acc=best,
                       data_fingerprint=report['manifest_sha256'], raw_only=True,
                       training_code_sha256=training_code_hash(),
                       architecture='frozen_hgate_syncbn', source_manifest_sha256=sha256(
                           Path(__file__).with_name('source_manifest.json')))
        for name in ['last_checkpoint.pth'] + (['best_checkpoint.pth'] if is_best else []):
            temporary = args.out / (name + '.tmp')
            torch.save(payload, temporary)
            temporary.replace(args.out / name)
    dist.barrier()


def setup_training(args: argparse.Namespace, recipe: dict, report: dict, local: int) -> tuple:
    """模型严格恢复；SyncBN不改变保存键，优化器仅接受实际可训练参数。"""
    model = build_model(sync_bn=True).cuda(local)
    model = DDP(model, device_ids=[local], broadcast_buffers=False,
                find_unused_parameters=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 150)
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    if checkpoint:
        check_resume(checkpoint, recipe, dist.get_world_size(), report['manifest_sha256'])
        if checkpoint.get('training_code_sha256') != training_code_hash():
            raise ValueError('续训的训练/读取代码与原checkpoint不同')
        model.module.load_state_dict(checkpoint['net'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        if checkpoint['source_manifest_sha256'] != sha256(Path(__file__).with_name('source_manifest.json')):
            raise ValueError('续训模型代码不同')
    return model, optimizer, scheduler, checkpoint


def benchmark(args: argparse.Namespace, model: DDP, train: DataLoader, optimizer: object) -> None:
    """两批预热后测真实输入；报告同步通信和AdamW开销，不写模型权重。"""
    train_epoch(model, train, optimizer, args.accum, 2)
    torch.cuda.reset_peak_memory_stats()
    dist.barrier()
    result = train_epoch(model, train, optimizer, args.accum, args.steps)
    peaks = [None] * dist.get_world_size()
    dist.all_gather_object(peaks, torch.cuda.max_memory_allocated() / 1024 ** 3)
    result.update(samples_per_second=result['samples'] / result['seconds'], peak_gib=peaks,
                  world_size=dist.get_world_size(), micro_batch=args.batch_size,
                  accumulation=args.accum, sync_bn=True, gate_gradients_checked=True,
                  includes_optimizer_and_communication=True)
    if dist.get_rank() == 0:
        dump_json(args.out / 'benchmark.json', result)
        print(json.dumps(result), flush=True)


def fit(args: argparse.Namespace, model: DDP, optimizer: object, scheduler: object,
        checkpoint: dict | None, loaders: tuple, recipe: dict, report: dict) -> None:
    """固定150轮，不早停；每轮保存可恢复状态与raw测试结果。"""
    train, test, sampler, loader_rng = loaders
    start, best = (checkpoint['epoch'] + 1, checkpoint['best_test_acc']) if checkpoint else (0, -1.)
    if checkpoint:
        state = checkpoint['rng_states'][dist.get_rank()]
        restore_rng(state['rng'])
        loader_rng.set_state(state['loader'])
    for epoch in range(start, 150):
        sampler.set_epoch(epoch)
        output = train_epoch(model, train, optimizer, args.accum)
        scheduler.step()
        validation = evaluate(model.module, test, args.seed)
        improved = validation['accuracy'] > best
        best = max(best, validation['accuracy'])
        item = dict(epoch=epoch + 1, train=output, test=validation, best_raw=best,
                    lr=scheduler.get_last_lr(), effective_batch=32)
        if dist.get_rank() == 0:
            with (args.out / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(item) + '\n')
            print(json.dumps(item), flush=True)
        save_checkpoint(model, optimizer, scheduler, epoch, recipe, report, args, best,
                        loader_rng, improved)
    if dist.get_rank() == 0:
        dump_json(args.out / 'completed.json', dict(epochs=150, best_raw=best,
                  primary_checkpoint='last_checkpoint.pth', stage2_not_started=True))


def main() -> None:
    """入口只处理本次实验，异常由torchrun终止同作业其他rank。"""
    args = parse_args()
    rank, local, world, report = initialize(args)
    recipe = dict(RECIPE, batch_size=args.batch_size, grad_accum=args.accum, seed=args.seed,
                  eval_batch=args.eval_batch, workers=args.workers)
    if rank == 0:
        dump_json(args.out / 'launch.json', dict(recipe=recipe, world_size=world,
                  mode=args.mode, data=report, resume=str(args.resume), pid=os.getpid(),
                  torch=torch.__version__, cuda=torch.version.cuda))
    loaders = make_loaders(args, rank, world)
    model, optimizer, scheduler, checkpoint = setup_training(args, recipe, report, local)
    if args.mode == 'benchmark':
        benchmark(args, model, loaders[0], optimizer)
    elif args.mode == 'evaluate':
        saved = load_checkpoint(args.checkpoint)
        if saved['data_fingerprint'] != report['manifest_sha256']:
            raise ValueError('评估数据不同于训练数据')
        model.module.load_state_dict(saved['net'], strict=True)
        result = evaluate(model.module, loaders[1], args.seed)
        if rank == 0:
            dump_json(args.out / 'evaluation.json', result)
    else:
        fit(args, model, optimizer, scheduler, checkpoint, loaders, recipe, report)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
