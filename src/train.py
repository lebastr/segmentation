import argparse
import dataset as DS
import os
import json
import torch
import unet as U
import sampler as S
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

def batch_generator(learning_point_generator, batch_size):
    batch_features = []
    batch_target = []

    for b in range(batch_size):
        X, Y = learning_point_generator.next()
        batch_features.append(X)
        batch_target.append(Y)

    return np.array(batch_features), np.float32(batch_target)

def eval_metrics(predicted, ground_truth):
    loss = F.binary_cross_entropy(predicted, ground_truth)

    predicted_mask = (predicted.data > 0.5).int()
    ground_truth_mask = (ground_truth.data > 0.5).int()

    tp_mask = predicted_mask & ground_truth_mask
    fp_mask = predicted_mask - tp_mask
    fn_mask = ground_truth_mask - tp_mask

    tp = float(tp_mask.sum())

    relevant = float(ground_truth_mask.sum())
    selected = float(predicted_mask.sum())

    precision = tp / selected if selected >= 1 else None
    recall = tp / relevant if relevant >= 1 else None

    f1 = 2*precision*recall/(precision + recall) if (precision is not None) and (recall is not None) and (precision+recall >= 1) else None

    return {'tp_mask': tp_mask, 'fp_mask': fp_mask, 'fn_mask': fn_mask,
            'tp': tp, 'relevant': relevant, 'selected': selected,
            'precision': precision, 'recall': recall, 'f1': f1, 'loss': loss }


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
    input_size = args.input_size
    tb_log_dir = args.tb_log_dir
    n_steps = args.n_steps
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
        print("Use pretrained weihts %s" % str(pretrained_vgg))
        net = U.Unet(vgg_pretrained=pretrained_vgg)
        network_manager.register_net(net)

    print("Move to GPU")
    net.cuda()

    def get_features(x):
        return x.get_ndarray([DS.ChannelRGB_PanSharpen])

    def get_target(x):
        return x.get_interior_mask()

    learning_point_generator = S.Sampler(train, get_features, get_target,
                                         input_size, input_size - 208, rotate_amplitude=20,
                                         random_crop=True, reflect=True)()

    if fix_vgg:
        parameters = list(net.bn.parameters()) + list(net.decoder.parameters()) + list(net.conv1x1.parameters())
    else:
        parameters = net.parameters()

    print("LR: %f" % learning_rate)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    logger = SummaryWriter(tb_log_dir + "/" + net_name)

    print("Start learning")
    try:
        for step in tqdm.tqdm(network_manager.iterator(n_steps), initial=network_manager.iteration):
            batch_features, batch_target = batch_generator(learning_point_generator, batch_size)
            
            batch_features = Variable(FloatTensor(batch_features)).cuda()
            batch_target = Variable(FloatTensor(batch_target)).cuda()

            predicted = net.forward(batch_features)

            metrics = eval_metrics(predicted, batch_target)
            loss = metrics['loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            def log_if_not_none(name, v):
                if v is not None:
                    logger.add_scalar(name, v, step)

            logger.add_scalar('loss', loss.data[0], step)
            log_if_not_none('precision', metrics['precision'])
            log_if_not_none('recall', metrics['recall'])
            log_if_not_none('f1', metrics['f1'])
            logger.add_scalar('relevant', metrics['relevant'], step)

            if step % 1000 == 0:
                network_manager.save()

    except KeyboardInterrupt as e:
        network_manager.save()

    except BaseException as e:
        network_manager.save()
        raise e
        
if __name__ == "__main__":
    main()
