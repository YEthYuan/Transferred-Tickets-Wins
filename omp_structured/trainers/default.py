import time
import torch
import torch.nn.functional as F
import tqdm
import abc

from robustness.tools.helpers import has_attr

from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter


__all__ = ["train", "validate", "modifier"]

cifar_mean = (0.4914, 0.4822, 0.4465)
cifar_std = (0.2471, 0.2435, 0.2616)

svhn_mean = (0.5, 0.5, 0.5)
svhn_std = (0.5, 0.5, 0.5)

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)


def batch_multiply_tensor_by_vector(vector, batch_tensor):
    """Equivalent to the following
    for ii in range(len(vector)):
        batch_tensor.data[ii] *= vector[ii]
    return batch_tensor
    """
    return (
            batch_tensor.transpose(0, -1) * vector).transpose(0, -1).contiguous()


def batch_multiply(float_or_vector, tensor):
    if isinstance(float_or_vector, torch.Tensor):
        assert len(float_or_vector) == len(tensor)
        tensor = batch_multiply_tensor_by_vector(float_or_vector, tensor)
    elif isinstance(float_or_vector, float):
        tensor *= float_or_vector
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor


def clamp_by_pnorm(x, p, r):
    assert isinstance(p, float) or isinstance(p, int)
    norm = get_norm_batch(x, p)
    if isinstance(r, torch.Tensor):
        assert norm.size() == r.size()
    else:
        assert isinstance(r, float)
    factor = torch.min(r / norm, torch.ones_like(norm))
    return batch_multiply(factor, x)


def normalize_by_pnorm(x, p=2, small_constant=1e-6):
    """
    Normalize gradients for gradient (not gradient sign) attacks.
    # TODO: move this function to utils
    :param x: tensor containing the gradients on the input.
    :param p: (optional) order of the norm for the normalization (1 or 2).
    :param small_constant: (optional float) to avoid dividing by zero.
    :return: normalized gradients.
    """
    # loss is averaged over the batch so need to multiply the batch
    # size to find the actual gradient of each input sample

    assert isinstance(p, float) or isinstance(p, int)
    norm = get_norm_batch(x, p)
    norm = torch.max(norm, torch.ones_like(norm) * small_constant)
    return batch_multiply(1. / norm, x)


def get_norm_batch(x, p):
    batch_size = x.size(0)
    return x.abs().pow(p).view(batch_size, -1).sum(dim=1).pow(1. / p)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def train_adv(train_loader, model, criterion, optimizer, epoch, args, writer, log):
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
    
    mu = torch.tensor(mean).view(3,1,1).cuda()
    std = torch.tensor(std).view(3,1,1).cuda()
    
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

        X = X.cuda()
        y = y.cuda()

        delta = torch.zeros_like(X).cuda()
        
        if 'fgsm' in args.attack_type:
            if args.attack_type == 'fgsm-rs':
                for j in range(len(mu)):
                    delta[:, j, :, :].uniform_(-epsilon, epsilon)
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta.requires_grad = True
            
            output = model(X + delta[:X.size(0)])
            loss = F.cross_entropy(output, y)

            loss.backward()

            if args.constraint == 'Linf':
                grad = delta.grad.detach()
                delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon * ones, epsilon * ones)
                delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
            elif args.constraint == 'L2':
                grad = delta.grad.detach()
                grad = normalize_by_pnorm(grad)
                delta.data = delta.data + batch_multiply(alpha, grad)
                delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
                delta.data = clamp_by_pnorm(delta.data, 2, epsilon)
            else:
                print('Wrong Attack Constraint!')
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

                if args.constraint == 'Linf':
                    grad = delta.grad.detach()
                    delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon * ones, epsilon * ones)
                    delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
                elif args.constraint == 'L2':
                    grad = delta.grad.detach()
                    grad = normalize_by_pnorm(grad)
                    delta.data = delta.data + batch_multiply(alpha, grad)
                    delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
                    delta.data = clamp_by_pnorm(delta.data, 2, epsilon)
                else:
                    print('Wrong Attack Constraint!')
                    exit()
        else:
            print('Wrong attack type:', args.attack_type)
            exit()
        
        delta = delta.detach()

        # compute output
        output = model(X+delta[:X.size(0)])

        loss = criterion(output, y)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, y, topk=(1, 5))
        losses.update(loss.item(), X.size(0))
        top1.update(acc1.item(), X.size(0))
        top5.update(acc5.item(), X.size(0))

        # compute gradient and do SGD step
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
    return step_size*torch.sign(gradz)

global_noise_data = None
def train_adv_free(train_loader, model, criterion, optimizer, epoch, args, writer, log):
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
    elif args.set == 'svhn':
        mean = svhn_mean
        std = svhn_std
    elif 'cifar' in args.set:
        mean = cifar_mean
        std = cifar_std
    else:
        print('Plz specify mean and std in the trainer for dataset:', args.set)
        exit()
    
    mu = torch.tensor(mean).view(3,1,1).cuda()
    std = torch.tensor(std).view(3,1,1).cuda()

    upper_limit = ((1 - mu)/ std)
    lower_limit = ((0 - mu)/ std)
    
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
    
    mu = torch.tensor(mean).view(3,1,1).cuda()
    std = torch.tensor(std).view(3,1,1).cuda()

    upper_limit = ((1 - mu)/ std)
    lower_limit = ((0 - mu)/ std)
    
    # epsilon = (args.epsilon / 255.) / std
    # # alpha = (args.alpha / 255.) / std
    # alpha = (2 / 255.) / std
    
    if args.constraint == 'Linf':
        epsilon = args.epsilon / 255.
        alpha = 2 / 255.
    elif args.constraint == 'L2':
        epsilon = args.epsilon
        alpha = 2 / 255.
    
    end = time.time()
    for i, (X, y) in tqdm.tqdm(
        enumerate(val_loader), ascii=True, total=len(val_loader)
    ):

        X = X.cuda()
        y = y.cuda()

        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, lower_limit, upper_limit, attack_iters=3, restarts=1)
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


def attack_pgd(model, X, y, epsilon, alpha, lower_limit, upper_limit, attack_iters=20, restarts=1):
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
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()

            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon * ones, epsilon * ones)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        
        with torch.no_grad():
            all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
            max_loss = torch.max(max_loss, all_loss)
            
    return max_delta


def train(train_loader, model, criterion, optimizer, epoch, args, writer, log):
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
