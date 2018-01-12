import argparse
import copy
import random
import os
import json
import torch
import numpy as np
import tqdm

from contextlib import contextmanager
from torch import FloatTensor
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim
import torchvision.utils

import PIL.Image as PImg

import car_dataset
import figure as fig
import sampler as S
import unet as U

ALPHA = 0.8  # transparency level of prediction mask on sample collage
SAMPLE_DIR = 'samples'


class NManager(object):
    def __init__(self, root, name):
        self.model_dir = os.path.join(root, name)
        self.model_state_path = os.path.join(self.model_dir, "state.json")
        self.name = name

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            print("New net will live in %s" % self.model_dir)

        if os.path.exists(self.model_state_path):
            print('Restoring model from', self.model_state_path)
            self.restore()
            self.registered = True

        else:
            self.registered = False

    def __last_model_path__(self):
        return os.path.join(self.model_dir, "model%d.pth" % self.iteration)

    def registered(self):
        return self.registered

    def get_net(self):
        return self.net

    def restore(self):
        state = json.load(open(self.model_state_path, 'r'))
        self.iteration = state['iteration']
        self.name = state['name']

        self.net = torch.load(self.__last_model_path__())
        print("Net %s was restored from checkpoint %d" % (self.name, self.iteration))
        self.iteration += 1
        return self.net

    def register_net(self, net):
        assert not self.registered, "Network is already registered"
        assert isinstance(net, U.Unet), "Net must be Unet instance"

        self.iteration = 0
        self.net = net
        self.registered = True

    def save(self):
        print("\ncheckpoint at: %d" % self.iteration)
        json.dump({'iteration': self.iteration, 'name': self.name}, open(self.model_state_path, 'w'))
        torch.save(self.net, self.__last_model_path__())

    @contextmanager
    def session(self, n_steps):
        it = self.iterator(n_steps)
        try:
            yield it, self.iteration

        finally:
            self.save()

    def iterator(self, n_steps):
        start = self.iteration
        def cond():
            if n_steps <= 0:
                return True
            else:
                return self.iteration - start <= n_steps

        while cond():
            yield self.iteration
            self.iteration += 1

def batch_generator(sampler, batch_size):
    batch_features = []
    batch_target = []

    for b in range(batch_size):
        X, Y = next(sampler)
        batch_features.append(X)
        batch_target.append(Y)

    x = np.array(batch_features)
    y = np.array(batch_target)
    return x,y

def main():
    parser = argparse.ArgumentParser(description="Train U-net")

    parser.add_argument('--name', type=str, default='unknown',
                        help='network name')

    parser.add_argument('--model_dir', type=str, required=True,
                        help='Where network will be saved and restored')

    parser.add_argument("--lr",
                        type=float,
                        default=1e-4,
                        help="Adam learning rate")

    parser.add_argument("--batch_size",
                        type=int,
                        default=5,
                        help="Batch size")


    parser.add_argument("--input_size",
                        type=int,
                        default=324,
                        help="Input size of the image will fed into network. Input_size = 16*n + 4, Default: 324")

    parser.add_argument("--output_size",
                        type=int,
                        default=116,
                        help="size of the image produced by network. Default: 116")

    parser.add_argument("--n_steps",
                        type=int,
                        default=0,
                        help="Number of the steps. Default: 0 means infinity steps.")

    parser.add_argument("--dataset_dir",
                        type=str,
                        default="../dataset/trainset")

    parser.add_argument("--pretrained_vgg",
                        type=str,
                        choices=['yes', 'no'],
                        default="yes",
                        help="Use pretrained vgg weigth")

    parser.add_argument("--fix_vgg",
                        type=str,
                        choices=['yes', 'no'],
                        default="yes",
                        help="Fix vgg weights while learning")

    parser.add_argument("--introspect_every",
                        type=int,
                        help="Dump picture and result every n-th batch")

    parser.add_argument("--plot_loss_every",
                        type=int,
                        help="Dump picture and result every n-th batch")

    args = parser.parse_args()

    net_name = args.name
    model_dir = args.model_dir
    learning_rate = args.lr
    batch_size = args.batch_size
    net_input_size = args.input_size
    net_output_size = args.output_size
    n_steps = args.n_steps
    dataset_dir = args.dataset_dir
    pretrained_vgg = args.pretrained_vgg == 'yes'
    fix_vgg = args.fix_vgg == 'yes'

    print("Load dataset")
    dataset = car_dataset.CarDataset(dataset_dir)

    #for i in range(50):
    #    print(dataset[i][0].shape)


    print("Initialize network manager")
    network_manager = NManager(model_dir, net_name)

    if network_manager.registered:
        net = network_manager.get_net()
    else:
        print("Use pretrained weights %s" % str(pretrained_vgg))
        net = U.Unet(vgg_pretrained=pretrained_vgg, n_classes=12)
        network_manager.register_net(net)

    print("Move to GPU")
    net.cuda()

    def get_features(x): return x[0]
    def get_target(x): return x[1][None,:,:]

    train_sampler = S.Sampler(dataset, get_features, get_target,
                                    net_input_size, net_output_size, rotate_amplitude=5,
                                    random_crop=True, reflect=True)()

#    next(train_sampler)

    if fix_vgg:
        parameters = list(net.bn.parameters()) + list(net.decoder.parameters()) + list(net.conv1x1.parameters())
    else:
        parameters = net.parameters()

    print("LR: %f" % learning_rate)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    # logger = SummaryWriter(tb_log_dir + "/" + net_name)

    loss_acc = None
    log_fig = None
    if args.plot_loss_every is not None:
        loss_acc = fig.Accumulator(with_std=True)
        log_fig = fig.Figure(accums={'loss': loss_acc}, title='Learning curves')

    if args.introspect_every is not None:
        os.makedirs(SAMPLE_DIR, exist_ok=True)

    print("Start learning")
    with network_manager.session(n_steps) as (iterator, initial_step):
        for step in tqdm.tqdm(iterator, initial=initial_step):
            batch_features, batch_target = batch_generator(train_sampler, batch_size)

            batch_features = Variable(torch.from_numpy(batch_features)).cuda()
            batch_target = Variable(torch.from_numpy(np.int64(batch_target[:,0,:,:]))).cuda()

            predicted = net.forward(batch_features)

            loss = nn.NLLLoss2d()(predicted, batch_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.introspect_every is not None and step % args.introspect_every == 0:
                # Just RGB image
                img_data = (batch_features[0].data.cpu().permute(1, 2, 0).numpy().clip(0, 255))
                # Three first channels of annotation rendered as R,G,B
                target = batch_target[0]
                # tgt_data = target.data.cpu().numpy()
                # tgt_ch0 = tgt_data == 0
                # tgt_ch1 = tgt_data == 1
                # tgt_ch2 = tgt_data == 2
                # tgt_data = (255 * np.stack([tgt_ch0,tgt_ch1,tgt_ch2], axis=2)).astype(np.uint8)

                # Three first channels of prediction rendered as R,G,B
                pred = predicted[0]
                # pred_data = (pred.data.cpu().permute(1, 2, 0).numpy()[:, :, :2] * 255).astype(np.uint8)

                # Heatmap of correctness, more red -> more confidently incorrect, more green -> more confidently correct
                pred_confidence, pred_class = pred.max(dim=0)
                true_pred_mask = (pred_class == target).type_as(pred_confidence)
                correct_preds = pred_confidence * true_pred_mask
                incorrect_preds = pred_confidence * (1-true_pred_mask)
                corr_pred_data = (torch.stack([incorrect_preds, correct_preds, torch.zeros_like(correct_preds)], dim=2)*255).data.cpu().numpy()

                border0 = (img_data.shape[0] - corr_pred_data.shape[0]) // 2
                border1 = (img_data.shape[1] - corr_pred_data.shape[1]) // 2

                img_pred_zone = img_data[border0:border0+corr_pred_data.shape[0], border1:border1+corr_pred_data.shape[1], :]
                corr_collage = img_pred_zone * ALPHA + corr_pred_data * (1-ALPHA)

                PImg.fromarray(img_data.astype(np.uint8)).save(os.path.join(SAMPLE_DIR, '%05d_input.png' % step))
                #PImg.fromarray(tgt_data).save(os.path.join(SAMPLE_DIR, '%05d_tgt.png' % step))
                #PImg.fromarray(pred_data).save(os.path.join(SAMPLE_DIR, '%05d_pred.png' % step))
                PImg.fromarray(corr_pred_data.astype(np.uint8)).save(os.path.join(SAMPLE_DIR, '%05d_corr.png' % step))
                PImg.fromarray(corr_collage.astype(np.uint8)).save(os.path.join(SAMPLE_DIR, '%05d_collage.png' % step))


            tqdm.tqdm.write("step: %d, loss: %f, lg(lr): %f" % (step, loss.data[0], np.log(learning_rate)/np.log(10)))

            if args.plot_loss_every is not None:
                loss_acc.append(loss.data.cpu()[0])

                if step % args.plot_loss_every == 0:
                    loss_acc.accumulate()
                    log_fig.plot_accums()
                    log_fig.draw()

            if step % 1000 == 0:
                network_manager.save()

if __name__ == "__main__":
    main()
