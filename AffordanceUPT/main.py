"""
Utilities for training, testing and caching results
for HICO-DET and V-COCO evaluations.

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import torch
import random
import warnings
import argparse
import numpy as np
import wandb
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

from upt import build_detector
from utils import custom_collate, CustomisedDLE, DataFactory

warnings.filterwarnings("ignore")

def main(rank, args):
    if rank == 0:
        wandb.init(project="upt-subset_v2", config=args)

    dist.init_process_group(
        backend="nccl",
        #backend="gloo",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )

    # Fix seed
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.set_device(rank)

    trainset = DataFactory(name=args.dataset, partition=args.partitions[0], data_root=args.data_root)
    testset = DataFactory(name=args.dataset, partition=args.partitions[1], data_root=args.data_root)
    print("DataFactory created")
    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        sampler=DistributedSampler(
            trainset, 
            num_replicas=args.world_size, 
            rank=rank)
    )
    test_loader = DataLoader(
        dataset=testset,
        collate_fn=custom_collate, batch_size=1,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        sampler=torch.utils.data.SequentialSampler(testset)
    )
    print("Loader created")

    args.human_idx = 1
    object_to_target = train_loader.dataset.dataset.object_to_verb
    args.num_classes = 2

    upt = build_detector(args, object_to_target)
    if rank == 0:
        wandb.watch(upt, log_freq=args.print_interval)

    if os.path.exists(args.resume):
        print(f"=> Rank {rank}: continue from saved checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        upt.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"=> Rank {rank}: start from a randomly initialised model")

    print("CustomisedDLE")

    if rank == 0:
        outdir = args.output_dir + "/" + wandb.run.name + "_rank0/"
    else:
        outdir = args.output_dir + "/rank1/"
    #outdir = args.output_dir + "/rank1/"
    engine = CustomisedDLE(
        upt, train_loader,
        max_norm=args.clip_max_norm,
        num_classes=args.num_classes,
        print_interval=args.print_interval,
        find_unused_parameters=False,
        cache_dir=outdir
        #cache_dir=args.output_dir
    )

    """
    if args.cache:
        if args.dataset == 'hicodet':
            engine.cache_hico(test_loader, args.output_dir)
        elif args.dataset == 'vcoco':
            engine.cache_vcoco(test_loader, args.output_dir)
        return
    """

    if args.eval:
        if args.dataset == 'vcoco':
            raise NotImplementedError(f"Evaluation on V-COCO has not been implemented.")
        eval1, my_dict = engine.test_hico(test_loader)
        ap, max_rec = eval1
        #avg_prec = eval2

        # Fetch indices for rare and non-rare classes
        num_anno = torch.as_tensor(trainset.dataset.anno_interaction)
        rare = torch.nonzero(num_anno < 10).squeeze(1)
        non_rare = torch.nonzero(num_anno >= 10).squeeze(1)
        ap = ap.detach().cpu().numpy()
        ap[ap == 0] = np.nan
        max_rec = max_rec.detach().cpu().numpy()
        max_rec[max_rec == 0] = np.nan

        print(ap)
        print(max_rec)
        print(
            f"The mAP is {np.nanmean(ap):.4f},"
            f" rare: {np.nanmean(ap[rare]):.4f},"
            f" none-rare: {np.nanmean(ap[non_rare]):.4f}"
            f" The Recall is {np.nanmean(max_rec):.4f},"
            f" rare: {np.nanmean(max_rec[rare]):.4f},"
            f" none-rare: {np.nanmean(max_rec[non_rare]):.4f}"
        )
        print("========")
        print(my_dict)
        return

    for p in upt.detector.parameters():
        p.requires_grad = False

    param_dicts = [{
        "params": [p for n, p in upt.named_parameters() if "interaction_head" in n and p.requires_grad]
    }]

    optim = torch.optim.AdamW(
        param_dicts, lr=args.lr_head,
        weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, args.lr_drop)
    # Override optimiser and learning rate scheduler
    engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler)
    engine.save_checkpoint()
    print("Start Engine")
    engine(args.epochs)

    if rank == 0:
        print("Finish WANDB1")
        wandb.finish(0)


if __name__ == '__main__':
    # nohup python main.py > log-3-hyperopt.txt 2>&1 &
    # CUDA_VISIBLE_DEVICES=1 python main.py --world-size 1 --eval --resume /mnt/hydra/ssd4/team/henlein/upt-eval/models/desired-lovebird-3/ckpt_41940_20.pt
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr-head', default=0.00013, type=float)  # <----
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--weight-decay', default=0.00047, type=float)  # <----
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr-drop', default=82, type=int)  # <----
    parser.add_argument('--clip-max-norm', default=0.18, type=float)  # <----
    """
    parser.add_argument('--lr-head', default=1e-4, type=float)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr-drop', default=10, type=int)
    parser.add_argument('--clip-max-norm', default=0.1, type=float)
    """
    parser.add_argument('--repr-dim', default=512, type=int)
    parser.add_argument('--hidden-dim', default=256, type=int)
    parser.add_argument('--enc-layers', default=6, type=int)
    parser.add_argument('--dec-layers', default=6, type=int)
    parser.add_argument('--dim-feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num-queries', default=100, type=int)
    parser.add_argument('--pre-norm', action='store_true')

    parser.add_argument('--no-aux-loss', dest='aux_loss', action='store_false')
    parser.add_argument('--set-cost-class', default=1, type=float)
    parser.add_argument('--set-cost-bbox', default=5, type=float)
    parser.add_argument('--set-cost-giou', default=2, type=float)
    parser.add_argument('--bbox-loss-coef', default=5, type=float)
    parser.add_argument('--giou-loss-coef', default=2, type=float)
    parser.add_argument('--eos-coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    parser.add_argument('--alpha', default=0.25, type=float)  # <---
    parser.add_argument('--gamma', default=0.85, type=float)  # <---
    """
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.2, type=float)
    """
    parser.add_argument('--dataset', default='hicodet', type=str)
    parser.add_argument('--partitions', nargs='+', default=['train2015', 'test2015'], type=str)
    parser.add_argument('--num-workers', default=2, type=int)
    #parser.add_argument('--data-root', default='./hicodet')
    parser.add_argument('--data-root', default='/mnt/hydra/ssd4/team/henlein/HicoDetDataset')

    # training parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--port', default='1234', type=str)
    parser.add_argument('--seed', default=66, type=int)
    parser.add_argument('--pretrained', default='', help='Path to a pretrained detector')
    parser.add_argument('--resume', default='', help='Resume from a model')
    #parser.add_argument('--output-dir', default='checkpoints')
    parser.add_argument('--output-dir', default="/mnt/hydra/ssd4/team/henlein/upt-anno-v2/models")
    parser.add_argument('--print-interval', default=500, type=int)
    parser.add_argument('--world-size', default=2, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--sanity', action='store_true')
    parser.add_argument('--box-score-thresh', default=0.8, type=float)
    parser.add_argument('--fg-iou-thresh', default=0.5, type=float)
    parser.add_argument('--min-instances', default=3, type=int)
    parser.add_argument('--max-instances', default=15, type=int)

    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port
    print(args)
    mp.spawn(main, nprocs=args.world_size, args=(args,))

