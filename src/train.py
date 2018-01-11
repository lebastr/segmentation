import argparse
# import dataset as DS
import os
import json
import torch
import unet as U
import sampler as S
import numpy as np
import tqdm
import copy
import random

from contextlib import contextmanager
from torch import FloatTensor
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim
import torchvision.utils
import car_dataset

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
        print("Use pretrained weihts %s" % str(pretrained_vgg))
        net = U.Unet(vgg_pretrained=pretrained_vgg, n_classes=14)
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

            tqdm.tqdm.write("step: %d, loss: %f, lg(lr): %f" % (step, loss.data[0], np.log(learning_rate)/np.log(10)))

            if step % 1000 == 0:
                network_manager.save()

if __name__ == "__main__":
    main()
