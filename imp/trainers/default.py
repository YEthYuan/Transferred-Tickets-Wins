import time
import torch
import torch.nn.functional as F
import tqdm
import abc

from robustness.tools.helpers import has_attr
from torch.utils.tensorboard import SummaryWriter

__all__ = ["train", "validate", "modifier"]

cifar_mean = (0.4914, 0.4822, 0.4465)
cifar_std = (0.2471, 0.2435, 0.2616)

svhn_mean = (0.5, 0.5, 0.5)
svhn_std = (0.5, 0.5, 0.5)

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def train_adv(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]",
    )

    if args.attack_type == 'pgd':
        args.alpha = 2
    elif args.attack_type == 'fgsm-rs':
        args.alpha = 1.25 * args.epsilon

    if args.set == 'ImageNet':
        mean = imagenet_mean
        std = imagenet_std
    elif args.set == 'svhn':
        mean = svhn_mean
        std = svhn_std
    elif 'cifar' in args.set:
        mean = cifar_mean
        std = cifar_std
    else:
        print('Plz specify mean and std in the trainer for dataset:', args.set)
        exit()

    mu = torch.tensor(mean).view(3, 1, 1).cuda()
    std = torch.tensor(std).view(3, 1, 1).cuda()

    upper_limit = ((1 - mu) / std)
    lower_limit = ((0 - mu) / std)

    # epsilon = (args.epsilon / 255.) / std
    # alpha = (args.alpha / 255.) / std
    if args.constraint == 'Linf':
        epsilon = args.epsilon / 255.
        alpha = args.alpha / 255.
    elif args.constraint == 'L2':
        epsilon = args.epsilon
        alpha = args.alpha / 255.
    else:
        epsilon = args.epsilon / 255.
        alpha = epsilon * 2.5 / 5

    ones = torch.ones_like(mu).cuda()
    # switch to train mode
    model.train()

    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    end = time.time()
    for i, (X, y) in tqdm.tqdm(
            enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        # measure data loading time
        data_time.update(time.time() - end)

        if epoch < args.warmup:
            warmup_lr(epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args)

        X = X.cuda()
        y = y.cuda()

        delta = torch.zeros_like(X).cuda()

        if 'fgsm' in args.attack_type:
            if args.attack_type == 'fgsm-rs':
                # for j in range(len(epsilon)):
                #     delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                for j in range(len(mu)):
                    delta[:, j, :, :].uniform_(-epsilon, epsilon)
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta.requires_grad = True

            output = model(X + delta[:X.size(0)])
            loss = F.cross_entropy(output, y)

            loss.backward()
            if constraint == 'Linf':
                delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon * ones, epsilon * ones)
                delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
            elif constraint == 'L2':
                l = len(X.shape) - 1
                g_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, *([1]*l))
                scaled_g  = grad / (g_norm + 1e-10)
                delta.data = delta.data + alpha * scaled_g
                delta.data = delta.renorm(p=2, dim=0, maxnorm=epsilon)
                delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
            else:
                print('Wrong Constraint')
                exit()

        elif args.attack_type == 'pgd':
            for j in range(len(mu)):
                    delta[:, j, :, :].uniform_(-epsilon, epsilon)
            delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta.requires_grad = True

            for _ in range(args.attack_iters):
                output = model(X + delta)
                loss = criterion(output, y)

                loss.backward()

                if constraint == 'Linf':
                delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon * ones, epsilon * ones)
                delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
                elif constraint == 'L2':
                    l = len(X.shape) - 1
                    g_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, *([1]*l))
                    scaled_g  = grad / (g_norm + 1e-10)
                    delta.data = delta.data + alpha * scaled_g
                    delta.data = delta.renorm(p=2, dim=0, maxnorm=epsilon)
                    delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
                else:
                    print('Wrong Constraint')
                    exit()

        else:
            print('Wrong attack type:', args.attack_type)
            exit()

        delta = delta.detach()

        # compute output
        output = model(X + delta[:X.size(0)])

        loss = criterion(output, y)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, y, topk=(1, 5))
        losses.update(loss.item(), X.size(0))
        top1.update(acc1.item(), X.size(0))
        top5.update(acc5.item(), X.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            t = (num_batches * epoch + i) * batch_size
            progress.display(i)
            progress.write_to_tensorboard(writer, prefix="train_adv", global_step=t)

    return top1.avg, top5.avg


def fgsm(gradz, step_size):
    return step_size * torch.sign(gradz)


global_noise_data = None


def train_adv_free(train_loader, model, criterion, optimizer, epoch, args, writer):
    global global_noise_data

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]",
    )

    if args.attack_type == 'pgd':
        args.alpha = 2
    elif args.attack_type == 'fgsm-rs':
        args.alpha = 1.25 * args.epsilon

    if args.set == 'ImageNet':
        mean = imagenet_mean
        std = imagenet_std
    elif args.set == 'SVHN':
        mean = svhn_mean
        std = svhn_std
    elif 'CIFAR' in args.set:
        mean = cifar_mean
        std = cifar_std
    else:
        print('Plz specify mean and std in the trainer for dataset:', args.set)
        exit()

    mu = torch.tensor(mean).view(3, 1, 1).cuda()
    std = torch.tensor(std).view(3, 1, 1).cuda()

    upper_limit = ((1 - mu) / std)
    lower_limit = ((0 - mu) / std)

    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std

    # switch to train mode
    model.train()

    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    end = time.time()
    for i, (X, y) in tqdm.tqdm(
            enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        if epoch < args.warmup:
            warmup_lr(epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args)

        # measure data loading time
        data_time.update(time.time() - end)

        X = X.cuda()
        y = y.cuda()

        if global_noise_data is None:
            global_noise_data = torch.zeros_like(X).cuda()

        for j in range(args.n_repeats):
            noise_batch = torch.autograd.Variable(global_noise_data[0:X.size(0)], requires_grad=True).cuda()
            noise_batch.data = clamp(noise_batch, lower_limit - X, upper_limit - X)

            output = model(X + noise_batch)
            loss = criterion(output, y)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, y, topk=(1, 5))
            losses.update(loss.item(), X.size(0))
            top1.update(acc1.item(), X.size(0))
            top5.update(acc5.item(), X.size(0))

            optimizer.zero_grad()
            loss.backward()

            if args.task == 'ft_full' and args.ft_full_mode == 'only_zero':
                for n, m in model.named_modules():
                    if hasattr(m, "clear_subset_grad"):
                        m.clear_subset_grad()

            if args.task == 'ft_full' and args.ft_full_mode == 'decay_on_zero':
                for n, m in model.named_modules():
                    if hasattr(m, "weight_decay_custom"):
                        m.weight_decay_custom(args.weight_decay, args.weight_decay_on_zero)

            if args.task == 'ft_full' and args.ft_full_mode == 'low_lr_zero':
                for n, m in model.named_modules():
                    if hasattr(m, "lr_scale_zero"):
                        m.lr_scale_zero(args.lr_scale_zero)

            if args.discard_mode:
                for n, m in model.named_modules():
                    if hasattr(m, "clear_low_score_grad"):
                        m.clear_low_score_grad()

            pert = fgsm(noise_batch.grad, epsilon)
            global_noise_data[0:X.size(0)] += pert.data
            global_noise_data.data = clamp(global_noise_data, -epsilon, epsilon)

            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            t = (num_batches * epoch + i) * batch_size
            progress.display(i)
            progress.write_to_tensorboard(writer, prefix="train_adv", global_step=t)

    return top1.avg, top5.avg


def validate_adv(val_loader, model, criterion, args, writer, epoch):
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    if args.set == 'ImageNet':
        mean = imagenet_mean
        std = imagenet_std
    elif args.set == 'svhn':
        mean = svhn_mean
        std = svhn_std
    elif 'cifar' in args.set:
        mean = cifar_mean
        std = cifar_std
    else:
        print('Plz specify mean and std in the trainer for dataset:', args.set)
        exit()

    mu = torch.tensor(mean).view(3, 1, 1).cuda()
    std = torch.tensor(std).view(3, 1, 1).cuda()

    upper_limit = ((1 - mu) / std)
    lower_limit = ((0 - mu) / std)

    # epsilon = (args.epsilon / 255.) / std
    # # alpha = (args.alpha / 255.) / std
    # alpha = (2 / 255.) / std
    
    if args.constraint == 'Linf':
        epsilon = args.epsilon / 255.
        alpha = 2 / 255.
    elif args.constraint == 'L2':
        epsilon = args.epsilon
        alpha = 2 / 255.
    else:
        epsilon = args.epsilon / 255.
        alpha = epsilon * 2.5 / 5

    end = time.time()
    for i, (X, y) in tqdm.tqdm(
            enumerate(val_loader), ascii=True, total=len(val_loader)
    ):

        X = X.cuda()
        y = y.cuda()

        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, lower_limit, upper_limit, constraint = args.constraint, attack_iters=3, restarts=1)
        # compute output
        output = model(X + pgd_delta)

        loss = criterion(output, y)

        # measure accuracy and record loss
        model_logits = output[0] if (type(output) is tuple) else output
        if has_attr(args, "custom_accuracy"):
            acc1, acc5 = args.custom_accuracy(model_logits, y)
        else:
            acc1, acc5 = accuracy(model_logits, y, topk=(1, 5))

        losses.update(loss.item(), X.size(0))
        top1.update(acc1.item(), X.size(0))
        top5.update(acc5.item(), X.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    progress.display(len(val_loader))

    if writer is not None:
        progress.write_to_tensorboard(writer, prefix="test_adv", global_step=epoch)

    return top1.avg, top5.avg


def attack_pgd(model, X, y, epsilon, alpha, lower_limit, upper_limit, constraint, attack_iters=20, restarts=1):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    ones = torch.ones((3,1,1)).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(3):
            delta[:, i, :, :].uniform_(-epsilon, epsilon)
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            loss = F.cross_entropy(output, y)
            loss.backward()

            grad = delta.grad.detach()
            if constraint == 'Linf':
                delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon * ones, epsilon * ones)
                delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
            elif constraint == 'L2':
                l = len(X.shape) - 1
                g_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, *([1]*l))
                scaled_g  = grad / (g_norm + 1e-10)
                delta.data = delta.data + alpha * scaled_g
                delta.data = delta.renorm(p=2, dim=0, maxnorm=epsilon)
                delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
            else:
                print('Wrong Constraint')
                exit()
            
        with torch.no_grad():
            all_loss = F.cross_entropy(model(X + delta), y, reduction='none').detach()
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
            max_loss = torch.max(max_loss, all_loss)

    return max_delta


def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]",
    )

    # switch to train mode
    model.train()

    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    end = time.time()
    for i, (X, y) in tqdm.tqdm(
            enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        if epoch < args.warmup:
            warmup_lr(epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args)

        # measure data loading time
        data_time.update(time.time() - end)

        X = X.cuda()
        y = y.cuda()

        # compute output
        output = model(X)

        loss = criterion(output, y)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, y, topk=(1, 5))
        losses.update(loss.item(), X.size(0))
        top1.update(acc1.item(), X.size(0))
        top5.update(acc5.item(), X.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            t = (num_batches * epoch + i) * batch_size
            progress.display(i)
            progress.write_to_tensorboard(writer, prefix="train", global_step=t)

    return top1.avg, top5.avg


def validate(val_loader, model, criterion, args, writer, epoch):
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (X, y) in tqdm.tqdm(
                enumerate(val_loader), ascii=True, total=len(val_loader)
        ):

            X = X.cuda()

            y = y.cuda()

            # compute output
            output = model(X)

            loss = criterion(output, y)

            # measure accuracy and record loss
            model_logits = output[0] if (type(output) is tuple) else output
            if has_attr(args, "custom_accuracy"):
                acc1, acc5 = args.custom_accuracy(model_logits, y)
            else:
                acc1, acc5 = accuracy(model_logits, y, topk=(1, 5))

            losses.update(loss.item(), X.size(0))
            top1.update(acc1.item(), X.size(0))
            top5.update(acc5.item(), X.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display(len(val_loader))

        if writer is not None:
            progress.write_to_tensorboard(writer, prefix="test", global_step=epoch)

    return top1.avg, top5.avg


def modifier(args, epoch, model):
    return


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def warmup_lr(epoch, step, optimizer, one_epoch_step, args):
    overall_steps = args.warmup * one_epoch_step
    current_steps = epoch * one_epoch_step + step

    lr = args.lr * current_steps / overall_steps
    lr = min(lr, args.lr)
    # print(" => set warmup lr=", lr)

    for p in optimizer.param_groups:
        p['lr'] = lr


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, tqdm_writer=True):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if not tqdm_writer:
            print("\t".join(entries))
        else:
            tqdm.tqdm.write("\t".join(entries))

    def write_to_tensorboard(
            self, writer: SummaryWriter, prefix="train", global_step=None
    ):
        for meter in self.meters:
            avg = meter.avg
            val = meter.val
            if meter.write_val:
                writer.add_scalar(
                    f"{prefix}/{meter.name}_val", val, global_step=global_step
                )

            if meter.write_avg:
                writer.add_scalar(
                    f"{prefix}/{meter.name}_avg", avg, global_step=global_step
                )

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class Meter(object):
    @abc.abstractmethod
    def __init__(self, name, fmt=":f"):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def update(self, val, n=1):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass


class AverageMeter(Meter):
    """ Computes and stores the average and current value """

    def __init__(self, name, fmt=":f", write_val=True, write_avg=True):
        self.name = name
        self.fmt = fmt
        self.reset()

        self.write_val = write_val
        self.write_avg = write_avg

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

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class VarianceMeter(Meter):
    def __init__(self, name, fmt=":f", write_val=False):
        self.name = name
        self._ex_sq = AverageMeter(name="_subvariance_1", fmt=":.02f")
        self._sq_ex = AverageMeter(name="_subvariance_2", fmt=":.02f")
        self.fmt = fmt
        self.reset()
        self.write_val = False
        self.write_avg = True

    @property
    def val(self):
        return self._ex_sq.val - self._sq_ex.val ** 2

    @property
    def avg(self):
        return self._ex_sq.avg - self._sq_ex.avg ** 2

    def reset(self):
        self._ex_sq.reset()
        self._sq_ex.reset()

    def update(self, val, n=1):
        self._ex_sq.update(val ** 2, n=n)
        self._sq_ex.update(val, n=n)

    def __str__(self):
        return ("{name} (var {avg" + self.fmt + "})").format(
            name=self.name, avg=self.avg
        )
