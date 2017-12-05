import argparse
import dataset as DS
import os
import json
import torch
import unet as U
import batchgen as BG
import numpy as np
import tqdm

from torch import FloatTensor
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim
from tensorboard import SummaryWriter


class NManager(object):
    def __init__(self, root, name):
        self.model_dir = root + "/" + name
        self.model_state_path = self.model_dir + "/state.json"
        self.name = name

        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
            print("Net will be live in %s" % self.model_dir)
            
        if os.path.exists(self.model_state_path):
            self.restore()
            self.registered = True

        else:
            self.registered = False

    def __last_model_path__(self):
        return self.model_dir + "/model%d.pth" % self.iteration

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

def batch_generator(learning_point_generator, batch_size):
    batch_features = []
    batch_target = []

    for b in range(batch_size):
        X, Y = learning_point_generator.next()
        batch_features.append(X)
        batch_target.append(Y)

    return np.array(batch_features), np.float32(batch_target)

def get_iterator(start, n_steps):
    if n_steps == 0:
        i = start
        while True:
            yield i
            i += 1

    else:
        for i in xrange(start, start+n_steps):
            yield i

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
                        help="Input size of the image will fed into network. Input_size = 16*n + 4")

    parser.add_argument("--tb_log_dir",
                        type=str,
                        required=True,
                        help="Tensorboard log dir")

    parser.add_argument("--iteration_n",
                        type=int,
                        default=0,
                        help="Iteration's number. 0 mean inf")

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
    input_size = args.input_size
    tb_log_dir = args.tb_log_dir
    iteration_n = args.iteration_n
    dataset_dir = args.dataset_dir
    pretrained_vgg = args.pretrained_vgg == 'yes'
    fix_vgg = args.fix_vgg == 'yes'

    print("Load dataset")
    dataset = DS.DataSet(dataset_dir)

    train = dataset.train_images()
    test = dataset.test_images()

    print("Initialize network manager")
    network_manager = NManager(model_dir, net_name)
    if network_manager.registered:
        net = network_manager.get_net()
    else:
        net = U.Unet(vgg_pretrained=pretrained_vgg)
        network_manager.register_net(net)

    print("Move to GPU")
    net.cuda()

    def get_features(x):
        return x.get_ndarray([DS.ChannelRGB_PanSharpen])

    def get_target(x):
        return x.get_interior_mask()

    learning_point_generator = BG.LearningPointGenerator(train, get_features, get_target,
                                                         input_size, input_size - 208, rotate_amplitude=10,
                                                         random_crop=True, reflect=True)()

    if fix_vgg:
        parameters = list(net.bn.parameters()) + list(net.decoder.parameters()) + list(net.conv1x1.parameters())
    else:
        parameters = net.parameters()

    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    logger = SummaryWriter(tb_log_dir + "/" + net_name)

    print("Start learning")
    try:
        for _ in tqdm.tqdm(get_iterator(network_manager.iteration, iteration_n)):
            iteration = network_manager.iteration
            batch_features, batch_target = batch_generator(learning_point_generator, batch_size)
            
            batch_features = Variable(FloatTensor(batch_features)).cuda()
            batch_target = Variable(FloatTensor(batch_target)).cuda()

            predicted = net.forward(batch_features)

            loss = F.binary_cross_entropy(predicted, batch_target)

            loss.backward()
            optimizer.step()

            logger.add_scalar('loss', loss.data[0], iteration)
            network_manager.iteration += 1

    except BaseException as e:
        network_manager.save()

if __name__ == "__main__":
    main()
