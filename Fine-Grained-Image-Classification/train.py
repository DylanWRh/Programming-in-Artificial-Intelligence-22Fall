import argparse
import os
import logging
from utils.data_utils import get_loader
from utils.scheduler import WarmupCosineSchedule
import time
from tqdm import tqdm

from models import configs
from models.modeling import ViT

import torch
import numpy as np


logger = logging.getLogger(__name__)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup(args):
    config = configs.get_b16_config()

    model = ViT(config, zero_head=True)
    model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f'Training params {args}')
    logger.info('Total params: \t%2.1fM' % num_params)

    return args, model


def compute_acc(preds, labels):
    return (preds == labels).mean()


def valid(args, model, test_loader, global_step):
    eval_losses = AverageMeter()

    logger.info('----------Validating----------')
    logger.info(f' Total validation steps: {test_loader}')
    logger.info(f' Validation batch size: {args.eval_batch_size}')

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc='Validating... (loss=X.X)',
                          bar_format='{l_bar}{r_bar}',
                          dynamic_ncols=True)
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)

            eval_loss = loss_fct(logits, y)
            eval_loss = eval_loss.mean()
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = compute_acc(all_preds, all_label)
    accuracy = torch.tensor(accuracy).to(args.device)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d\n" % global_step)
    logger.info("Valid Loss: %2.5f\n" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f\n" % accuracy)

    with open('output/log.txt', 'a', encoding='utf-8') as f:
        f.write('\n')
        f.write('Validation Results\n')
        f.write('Global Steps: %d\n' % global_step)
        f.write('Valid Loss: %2.5f\n' % eval_losses.avg)
        f.write('Valid Accuracy: %2.5f\n' % accuracy)

    return accuracy


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    checkpoint = {'model': model_to_save.state_dict()}
    torch.save(checkpoint, model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def train(args, model):
    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset
    train_loader, test_loader = get_loader(args)

    # Optimizer and Scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)

    # Training
    # logger output
    logger.info('----------Training----------')
    logger.info(f' Total optimization steps: {args.num_steps}')
    logger.info(f' Train batch size: {args.train_batch_size}')

    model.zero_grad()
    losses = AverageMeter()
    best_acc = 0
    global_step = 0
    start_time = time.time()

    while True:
        model.train()

        epoch_iterator = tqdm(train_loader,
                              desc='Training (X / X Steps) (loss=X.X)',
                              bar_format='{l_bar}{r_bar}',
                              dynamic_ncols=True)

        all_preds, all_label = [], []
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch

            loss, logits = model(x, y)
            loss = loss.mean()

            preds = torch.argmax(logits, dim=-1)

            if len(all_preds) == 0:
                all_preds.append(preds.detach().cpu().numpy())
                all_label.append(y.detach().cpu().numpy())
            else:
                all_preds[0] = np.append(
                    all_preds[0], preds.detach().cpu().numpy(), axis=0
                )
                all_label[0] = np.append(
                    all_label[0], y.detach().cpu().numpy(), axis=0
                )

            loss.backward()

            losses.update(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()

            epoch_iterator.set_description(
                f'Training ({step} / {args.num_steps})' + ('(loss)=%2.5f' % losses.val)
            )

            global_step += 1
            # Validation
            if global_step % args.eval_every == 0 or global_step == 1:
                with torch.no_grad():
                    acc = valid(args, model, test_loader, global_step)
                if best_acc < acc:
                    save_model(args, model)
                    best_acc = acc
                logger.info('best acc so far: %f' % best_acc)
                model.train()

            if global_step % args.num_steps == 0:
                break

        all_preds, all_label = all_preds[0], all_label[0]
        accuracy = compute_acc(all_preds, all_label)
        accuracy = torch.tensor(accuracy).to(args.device)
        accuracy = accuracy.detach().cpu().numpy()
        logger.info("train accuracy so far: %f" % accuracy)
        losses.reset()
        if global_step % args.num_steps == 0:
            break

    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")
    end_time = time.time()
    logger.info("Total Training Time: \t%f" % ((end_time - start_time) / 3600))

    with open('output/res.txt','a') as f:
        f.write('Best Acc: \t%f\n' % best_acc)
        f.write('Total training time: \t%f\n' % ((end_time - start_time) / 3600))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',               type=str,   default='default')

    # args of files
    parser.add_argument('--data_root',          type=str,   default='dataset/CUB_200_2011')
    parser.add_argument('--pretrained_dir',     type=str,   default='ViT-B_16.npz')
    parser.add_argument('--output_dir',         type=str,   default='output')

    # args of training
    parser.add_argument('--train_batch_size',   type=int,   default=16)
    parser.add_argument('--num_steps',          type=int,   default=10000)

    # args of validating
    parser.add_argument('--eval_every',         type=int,   default=100)
    parser.add_argument('--eval_batch_size',    type=int,   default=4)

    # args of optimizer and scheduler
    parser.add_argument('--learning_rate',      type=float, default=3e-2)
    parser.add_argument('--warmup_steps',       type=int,   default=500)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    args, model = setup(args)

    train(args, model)


if __name__ == '__main__':
    main()
