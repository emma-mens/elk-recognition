import argparse
import time
import paramparse

import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from packaging import version

from tensorboardX import SummaryWriter

import yolov3_test_orig as yolov3_test  # Import test.py to get mAP after each epoch
from yolov3_models import *
from yolov3_utils.datasets import *
from yolov3_utils.utils import *

from YOLOv3TrainParams import YOLOv3TrainParams, HyperParams

update_scheduler_first = (version.parse(torch.__version__) < version.parse("1.1.0"))


def train(
        params,
        hyp,
        net_cfg,
        data_cfg,
        img_size=416,
        resume=False,
        epochs=273,  # 500200 batches at bs 64, dataset length 117263
        batch_size=16,
        accumulate=1,
        multi_scale=False,
        freeze_backbone=False,
        transfer=False,  # Transfer learning (train only YOLO layers)
        weights_dir='',
        pretrained_weights='',
        # Mixed precision training https://github.com/NVIDIA/apex
        # install help: https://github.com/NVIDIA/apex/issues/259
        mixed_precision=0,
        writer=None,
        load_sep=' ',
):
    """

    :param HyperParams hyp:
    :param net_cfg:
    :param data_cfg:
    :param img_size:
    :param resume:
    :param epochs:
    :param batch_size:
    :param accumulate:
    :param multi_scale:
    :param freeze_backbone:
    :param transfer:
    :param weights_dir:
    :param pretrained_weights:
    :param mixed_precision:
    :return:
    """
    init_seeds()

    print('pretrained_weights', pretrained_weights)
    if not pretrained_weights:
        pretrained_weights = 'pretrained_weights' + os.sep

    if not weights_dir:
        weights_dir = 'weights'

    print('weights: {}'.format(weights_dir))

    backup_weights = os.path.join(weights_dir, 'backup')
    latest = os.path.join(weights_dir, 'latest.pt')
    # best = os.path.join(weights, 'best.pt')

    if writer is None:
        writer = SummaryWriter(logdir=params.weights)

    if not os.path.isdir(weights_dir):
        os.makedirs(weights_dir)

    if not os.path.isdir(backup_weights):
        os.makedirs(backup_weights)

    device = torch_utils.select_device()

    if multi_scale:
        img_size = round((img_size / 32) * 1.5) * 32  # initiate with maximum multi_scale size
        # opt.num_workers = 0  # bug https://github.com/ultralytics/yolov3/issues/174
    else:
        torch.backends.cudnn.benchmark = True  # unsuitable for multiscale

    # Configure run
    data_dict = parse_data_cfg(data_cfg)
    train_path = data_dict['train']
    nc = int(data_dict['classes'])  # number of classes

    # Initialize model
    model = Darknet(net_cfg, img_size).to(device)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=hyp.lr0,
                          momentum=hyp.momentum,
                          weight_decay=hyp.weight_decay)

    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_loss = float('inf')
    nf = int(model.module_defs[model.yolo_layers[0] - 1]['filters'])  # yolo layer size (i.e. 255)
    if resume:  # Load previously saved model
        if transfer:  # Transfer learning
            chkpt = torch.load(pretrained_weights + 'yolov3-spp.pt', map_location=device)
            model.load_state_dict({k: v for k, v in chkpt['model'].items() if v.numel() > 1 and v.shape[0] != 255},
                                  strict=False)
            for p in model.parameters():
                p.requires_grad = True if p.shape[0] == nf else False

        else:  # resume from latest.pt
            print('Loading weights from: {}'.format(latest))
            chkpt = torch.load(latest, map_location=device)  # load checkpoint
            model.load_state_dict(chkpt['model'])

        start_epoch = chkpt['epoch'] + 1
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_loss = chkpt['best_loss']
        del chkpt

    else:  # Initialize model with backbone (optional)
        if '-tiny.cfg' in net_cfg:
            cutoff = load_darknet_weights(model, pretrained_weights + 'yolov3-tiny.conv.15')
        else:
            cutoff = load_darknet_weights(model, pretrained_weights + 'darknet53.conv.74')

    # Scheduler https://github.com/ultralytics/yolov3/issues/238
    # lf = lambda x: 1 - x / epochs  # linear ramp to zero
    # lf = lambda x: 10 ** (hyp['lrf'] * x / epochs)  # exp ramp
    lf = lambda x: 1 - 10 ** (hyp.lrf * (1 - x / epochs))  # inverse exp ramp
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch=start_epoch - 1)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[218, 245], gamma=0.1, last_epoch=start_epoch-1)

    # # Plot lr schedule
    # y = []
    # for _ in range(epochs):
    #     scheduler.step()
    #     y.append(optimizer.param_groups[0]['lr'])
    # plt.plot(y, label='LambdaLR')
    # plt.xlabel('epoch')
    # plt.xlabel('LR')
    # plt.tight_layout()
    # plt.savefig('LR.png', dpi=300)

    # Dataset
    dataset = LoadImagesAndLabels(train_path,
                                  img_size,
                                  batch_size,
                                  augment=True,
                                  rect=False,
                                  multi_scale=multi_scale,
                                  elk_vs_all=params.elk_vall)

    # Initialize distributed training
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend=params.backend, init_method=params.dist_url,
                                world_size=params.world_size, rank=params.rank)
        model = torch.nn.parallel.DistributedDataParallel(model)
        # sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    # Dataloader
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=params.num_workers,
                            shuffle=False,  # disable rectangular training if True
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)

    if mixed_precision:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        print('Using mixed_precision training')

    # Remove old results
    for f in glob.glob('*_batch*.jpg') + glob.glob('results.txt'):
        os.remove(f)

    # Start training
    model.hyp = hyp.__dict__  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    model_info(model)
    nb = len(dataloader)
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0)  # P, R, mAP, F1, test_loss
    n_burnin = min(round(nb / 5 + 1), 1000)  # burn-in batches
    t, t0 = time.time(), time.time()
    _test_iter = 0

    print('Saving weights to: {}'.format(weights_dir))
    best_chkpt = ''
    for epoch in range(start_epoch, epochs):
        model.train()
        print(('\n%8s%12s' + '%10s' * 7) % ('Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'targets', 'time'))

        if update_scheduler_first:
            # Update scheduler
            scheduler.step()

        # Freeze backbone at epoch 0, unfreeze at epoch 1 (optional)
        if freeze_backbone and epoch < 2:
            for name, p in model.named_parameters():
                if int(name.split('.')[1]) < cutoff:  # if layer < 75
                    p.requires_grad = False if epoch == 0 else True

        # Update image weights (optional)
        w = model.class_weights.cpu().numpy() * (1 - maps)  # class weights

        # print('dataset.labels: {}'.format(dataset.labels))
        # print('nc: {}'.format(nc))
        # print('w: {}'.format(w))
        # print('maps: {}'.format(maps))
        # print('model.class_weights: {}'.format(model.class_weights))

        image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
        dataset.indices = random.choices(range(dataset.n_files), weights=image_weights,
                                         k=dataset.n_files)  # random weighted index

        mloss = torch.zeros(5).to(device)  # mean losses
        for i, (_, imgs, targets, _, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            # Plot images with bounding boxes
            # TODO uncomment to allow plotting
#             if epoch == 0 and i == 0:
#                 plot_images(imgs=imgs, targets=targets, fname='train_batch0.jpg')

            # SGD burn-in
            if epoch == 0 and i <= n_burnin:
                lr = hyp.lr0 * (i / n_burnin) ** 4
                for x in optimizer.param_groups:
                    x['lr'] = lr
            print('lr', lr)

            # Run model
            pred = model(imgs)

            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model)
            if torch.isnan(loss):
                print('WARNING: nan loss detected, ending training')
                return results

            # Compute gradient
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Accumulate gradient for x batches before optimizing
            if (i + 1) % accumulate == 0 or (i + 1) == nb:
                optimizer.step()
                optimizer.zero_grad()

            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            s = ('%8s%12s' + '%10.3g' * 7) % (
                '%g/%g' % (epoch, epochs - 1),
                '%g/%g' % (i, nb - 1), *mloss, len(targets), time.time() - t)
            t = time.time()
            print(s)

            lxy, lwh, lconf, lcls, _loss = loss_items.cpu().numpy()
            _iter = i + (nb - 1) * epoch
            writer.add_scalar('train/total_loss', _loss, _iter)
            writer.add_scalar('train/xy_loss', lxy, _iter)
            writer.add_scalar('train/wh_loss', lwh, _iter)
            writer.add_scalar('train/conf_loss', lconf, _iter)
            writer.add_scalar('train/class_loss', lcls, _iter)
            writer.add_scalar('train/mean_loss', mloss[-1], _iter)

            # Multi-Scale training (320 - 608 pixels) every 10 batches
            if multi_scale and (i + 1) % 10 == 0:
                dataset.img_size = random.choice(range(10, 20)) * 32
                print('multi_scale img_size = %g' % dataset.img_size)

        if not update_scheduler_first:
            # Update scheduler
            scheduler.step()

        # Calculate mAP (always test final epoch, skip first 5 if opt.nosave)
        if not (params.notest or (params.nosave and epoch < 10)) or epoch == epochs - 1:
            _test_iter += 1
            with torch.no_grad():
                results, maps = yolov3_test.test(
                    net_cfg,
                    data_cfg,
                    batch_size=batch_size,
                    img_size=img_size,
                    model=model,
                    conf_thres=0.1,
                    load_sep=load_sep,
                    elk_vall=params.elk_vall
                )

            test_mp, test_mr, test_map, test_mf1, test_mloss = results
            writer.add_scalar('test/mp', test_mp, _test_iter)
            writer.add_scalar('test/mr', test_mr, _test_iter)
            writer.add_scalar('test/map', test_map, _test_iter)
            writer.add_scalar('test/mf1', test_mf1, _test_iter)
            writer.add_scalar('test/loss', test_mloss, _test_iter)
            # for _class in maps:
            #     writer.add_scalar('test/{}_map'.format(_class), maps[_class], _test_iter)

        # Write epoch results
        with open(os.path.join(weights_dir, 'results.txt'), 'a') as file:
            file.write(s + '%11.3g' * 5 % results + '\n')  # P, R, mAP, F1, test_loss

        # Update best loss
        test_loss = results[4]
        if test_loss < best_loss:
            best_loss = test_loss

        # Save training results
        save = (not params.nosave) or (epoch == epochs - 1)
        if save:
            # Create checkpoint
            chkpt = {'epoch': epoch,
                     'best_loss': best_loss,
                     'model': model.module.state_dict() if type(
                         model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                     'optimizer': optimizer.state_dict()}

            # Save latest checkpoint
            torch.save(chkpt, latest)

            # Save best checkpoint
            if best_loss == test_loss:
                if best_chkpt:
                    print('Removing old best_chkpt: {}'.format(best_chkpt))
                    os.remove(best_chkpt)
                best_chkpt = os.path.join(weights_dir, 'best_{}.pt'.format(epoch))
                torch.save(chkpt, best_chkpt)

            # Save backup every 10 epochs (optional)
            if epoch > 0 and epoch % 10 == 0:
                torch.save(chkpt, os.path.join(backup_weights, 'epoch_{:d}.pt'.format(epoch)))

            # Delete checkpoint
            del chkpt

    dt = (time.time() - t0) / 3600
    print('%g epochs completed in %.3f hours.' % (epoch - start_epoch, dt))
    return results


def print_mutation(hyp, results):
    # Write mutation results
    a = '%11s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%11.4g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%11.3g' * len(results) % results  # results (P, R, mAP, F1, test_loss)
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))
    with open('evolve.txt', 'a') as f:
        f.write(c + b + '\n')


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--epochs', type=int, default=273, help='number of epochs')
    # parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    # parser.add_argument('--accumulate', type=int, default=1, help='accumulate gradient x batches before optimizing')
    # parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    # parser.add_argument('--data-cfg', type=str, default='data/coco.data', help='coco.data file path')
    # parser.add_argument('--multi-scale', action='store_true', help='random image sizes per batch 320 - 608')
    # parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    # parser.add_argument('--resume', action='store_true', help='resume training flag')
    # parser.add_argument('--transfer', action='store_true', help='transfer learning flag')
    # parser.add_argument('--num-workers', type=int, default=4, help='number of Pytorch DataLoader workers')
    # parser.add_argument('--dist-url', default='tcp://127.0.0.1:9999', type=str, help='distributed training init method')
    # parser.add_argument('--rank', default=0, type=int, help='distributed training node rank')
    # parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
    # parser.add_argument('--backend', default='nccl', type=str, help='distributed backend')
    # parser.add_argument('--nosave', action='store_true', help='do not save training results')
    # parser.add_argument('--notest', action='store_true', help='only test final epoch')
    # parser.add_argument('--evolve', action='store_true', help='run hyperparameter evolution')
    # parser.add_argument('--var', default=0, type=int, help='debug variable')
    # parser.add_argument('--pretrained_weights', type=str, default='pretrained_weights', help='pretrained_weights path')
    # parser.add_argument('--weights', type=str, default='weights', help='weights path')
    # parser.add_argument('--mixed_precision', type=int, default=0, help='mixed_precision training')
    # paramparse.fromParser(parser, 'YOLOv3TrainParams')
    # opt = parser.parse_args()
    # print(opt)

    params = YOLOv3TrainParams()
    paramparse.process(params)

    load_sep = params.load_sep

    if load_sep == '0':
        load_sep = ' '
    elif load_sep == '1':
        load_sep = '\t'

    if params.evolve:
        params.notest = True  # save time by only testing final epoch
        params.nosave = True  # do not save checkpoints

    writer = SummaryWriter(logdir=params.weights)
    try:
        # Train
        results = train(
            params,
            params.hyp,
            params.net_cfg,
            params.data_cfg,
            img_size=params.img_size,
            resume=params.resume or params.transfer,
            transfer=params.transfer,
            epochs=params.epochs,
            batch_size=params.batch_size,
            accumulate=params.accumulate,
            multi_scale=params.multi_scale,
            weights_dir=params.weights,
            pretrained_weights=params.pretrained_weights,
            mixed_precision=params.mixed_precision,
            writer=writer,
            load_sep=load_sep,
        )
    except KeyboardInterrupt:
        writer.close()
        plot_results(folder=params.weights)

    # Evolve hyperparameters (optional)
    if params.evolve:
        hyp = params.hyp.__dict__

        best_fitness = results[2]  # use mAP for fitness

        # Write mutation results
        print_mutation(hyp, results)

        gen = 50  # generations to evolve
        for _ in range(gen):

            # Mutate hyperparameters
            old_hyp = hyp.copy()
            init_seeds(seed=int(time.time()))
            s = [.3, .3, .3, .3, .3, .3, .3, .03, .3]
            for i, k in enumerate(hyp.keys()):
                x = (np.random.randn(1) * s[i] + 1) ** 1.1  # plt.hist(x.ravel(), 100)
                hyp[k] = hyp[k] * float(x)  # vary by about 30% 1sigma

            # Clip to limits
            keys = ['lr0', 'iou_t', 'momentum', 'weight_decay']
            limits = [(1e-4, 1e-2), (0, 0.90), (0.70, 0.99), (0, 0.01)]
            for k, v in zip(keys, limits):
                hyp[k] = np.clip(hyp[k], v[0], v[1])

            # Determine mutation fitness
            results = train(
                params,
                params.hyp,
                params.net_cfg,
                params.data_cfg,
                img_size=params.img_size,
                resume=params.resume or params.transfer,
                transfer=params.transfer,
                epochs=params.epochs,
                batch_size=params.batch_size,
                accumulate=params.accumulate,
                multi_scale=params.multi_scale,
                weights_dir=params.weights,
                pretrained_weights=params.pretrained_weights,
                mixed_precision=params.mixed_precision,
                load_sep=load_sep,
            )
            mutation_fitness = results[2]

            # Write mutation results
            print_mutation(hyp.__dict__, results)

            # Update hyperparameters if fitness improved
            if mutation_fitness > best_fitness:
                # Fitness improved!
                print('Fitness improved!')
                best_fitness = mutation_fitness
            else:
                hyp = old_hyp.copy()  # reset hyp to

            # # Plot results
            # import numpy as np
            # import matplotlib.pyplot as plt
            # a = np.loadtxt('evolve_1000val.txt')
            # x = a[:, 2] * a[:, 3]  # metric = mAP * F1
            # weights = (x - x.min()) ** 2
            # fig = plt.figure(figsize=(14, 7))
            # for i in range(len(hyp)):
            #     y = a[:, i + 5]
            #     mu = (y * weights).sum() / weights.sum()
            #     plt.subplot(2, 5, i+1)
            #     plt.plot(x.max(), mu, 'o')
            #     plt.plot(x, y, '.')
            #     print(list(hyp.keys())[i],'%.4g' % mu)


if __name__ == '__main__':
    main()
