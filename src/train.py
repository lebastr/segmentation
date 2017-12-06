import argparse
import dataset as DS
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
import torch.optim
import torchvision.utils
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
        X, Y = sampler.next()
        batch_features.append(X)
        batch_target.append(Y)

    return np.array(batch_features), np.float32(batch_target)

def eval_precision_recall_f1(**kwargs):
    tp = kwargs['tp']
    selected = kwargs['selected']
    relevant = kwargs['relevant']

    precision = tp / selected if selected >= 1 else None
    recall = tp / relevant if relevant >= 1 else None

    f1 = 2*precision*recall/(precision + recall) \
        if (precision is not None) and (recall is not None) and (precision+recall >= 1) else None

    metrics = copy.copy(kwargs)
    metrics.update({'precision': precision, 'recall': recall, 'f1': f1})
    return metrics

def eval_base_metrics(predicted, ground_truth):
    loss = F.binary_cross_entropy(predicted, ground_truth)

    predicted_mask = (predicted.data > 0.5).int()
    ground_truth_mask = (ground_truth.data > 0.5).int()

    tp_mask = predicted_mask & ground_truth_mask
    fp_mask = predicted_mask - tp_mask
    fn_mask = ground_truth_mask - tp_mask

    tp = float(tp_mask.sum())

    relevant = float(ground_truth_mask.sum())
    selected = float(predicted_mask.sum())

    return {'tp_mask': tp_mask, 'fp_mask': fp_mask, 'fn_mask': fn_mask,
            'tp': tp, 'relevant': relevant, 'selected': selected, 'loss': loss }

def average_metrics(net, sampler, batch_size, n_samples):
    avg_loss = 0.0
    tp = 0.0
    relevant = 0.0
    selected = 0.0

    count = 0
    remainder = n_samples
    while remainder > 0:
        bsize = batch_size if remainder > batch_size else remainder

        batch_features, batch_target = batch_generator(sampler, bsize)

        batch_features = Variable(FloatTensor(batch_features)).cuda()
        batch_target = Variable(FloatTensor(batch_target)).cuda()

        predicted = net.forward(batch_features)

        metrics = eval_base_metrics(predicted, batch_target)

        avg_loss += metrics['loss'].data[0] * bsize

        tp += metrics['tp']
        relevant += metrics['relevant']
        selected += metrics['selected']

        remainder -= batch_size
        count += 1

    avg_metrics = eval_precision_recall_f1(tp=tp, selected=selected, relevant=relevant)
    avg_metrics.update({'loss' : avg_loss / n_samples })
    return avg_metrics

def log_metrics(tb_logger, prefix, metrics, step):
    def log_if_not_none(name, v):
        if v is not None:
            tb_logger.add_scalar(name, v, step)

    loss = metrics['loss']
    if isinstance(loss, Variable):
        loss = loss.data[0]

    tb_logger.add_scalar(prefix + '/loss', loss, step)
    log_if_not_none(prefix + '/precision', metrics['precision'])
    log_if_not_none(prefix + '/recall', metrics['recall'])
    log_if_not_none(prefix + '/f1', metrics['f1'])
    tb_logger.add_scalar(prefix + '/relevant', metrics['relevant'], step)

def generate_image(tb_logger, net, name, images, get_features, get_target,
                   net_input_size, net_output_size, step, level=0.09):

    for trial in range(20):
        image = random.choice(images)
        target = get_target(image)
        v = 1.0 * target.sum() / (target.shape[1]*target.shape[2])
        if v > level:
            break

    features = get_features(image)

    heat_map = U.predict(net, net_input_size, net_output_size, features)

    metrics = eval_base_metrics(Variable(torch.from_numpy(np.float32(heat_map))), Variable(torch.from_numpy(np.float32(target))))

    tp_mask = metrics['tp_mask']
    fp_mask = metrics['fp_mask']
    fn_mask = metrics['fn_mask']

    img = torch.zeros((3, 3, target.shape[1], target.shape[2]))
    img[0,1] += tp_mask.float() + fn_mask.float()
    img[0,0] += fp_mask.float() + fn_mask.float()

    img[1,:] += torch.from_numpy(np.float32(heat_map))[0]

    i3c = np.float32(features[:3])
    i3c = i3c / i3c.max()
    
    img[2,:] += torch.from_numpy(i3c)

    tb_logger.add_image(name, torchvision.utils.make_grid(img), step)

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

    parser.add_argument("--validation_freq",
                        type=int,
                        default=100,
                        help="Validation freq. Default 100")

    parser.add_argument("--validation_set_size",
                        type=int,
                        default=20,
                        help="metrics will be averaged by validation_set_size. Default 20")




    args = parser.parse_args()

    net_name = args.name
    model_dir = args.model_dir
    learning_rate = args.lr
    batch_size = args.batch_size
    net_input_size = args.input_size
    net_output_size = args.output_size
    tb_log_dir = args.tb_log_dir
    n_steps = args.n_steps
    dataset_dir = args.dataset_dir
    pretrained_vgg = args.pretrained_vgg == 'yes'
    fix_vgg = args.fix_vgg == 'yes'
    validation_freq = args.validation_freq
    validation_set_size = args.validation_set_size

    print("Load dataset")
    dataset = DS.DataSet(dataset_dir)

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

    train_sampler = S.Sampler(dataset.train_images(), get_features, get_target,
                                         net_input_size, net_output_size, rotate_amplitude=20,
                                         random_crop=True, reflect=True)()

    test_sampler = S.Sampler(dataset.test_images(), get_features, get_target,
                                         net_input_size, net_output_size, rotate_amplitude=20,
                                         random_crop=True, reflect=True)()

    if fix_vgg:
        parameters = list(net.bn.parameters()) + list(net.decoder.parameters()) + list(net.conv1x1.parameters())
    else:
        parameters = net.parameters()

    print("LR: %f" % learning_rate)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    logger = SummaryWriter(tb_log_dir + "/" + net_name)

    print("Start learning")
    with network_manager.session(n_steps) as (iterator, initial_step):
        for step in tqdm.tqdm(iterator, initial=initial_step):
            batch_features, batch_target = batch_generator(train_sampler, batch_size)
            
            batch_features = Variable(FloatTensor(batch_features)).cuda()
            batch_target = Variable(FloatTensor(batch_target)).cuda()

            predicted = net.forward(batch_features)

            train_metrics = eval_base_metrics(predicted, batch_target)
            train_metrics = eval_precision_recall_f1(**train_metrics)

            loss = train_metrics['loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log_metrics(logger, '', train_metrics, step)
            logger.add_scalar('lr', np.log(learning_rate)/np.log(10), step)
            
            if step % 1000 == 0:
                network_manager.save()

            if step % validation_freq == 0:
                test_metrics = average_metrics(net, test_sampler, batch_size, validation_set_size)
                log_metrics(logger, 'val', test_metrics, step)

                avg_train_metrics = average_metrics(net, train_sampler, batch_size, validation_set_size)
                log_metrics(logger, 'avg_train', avg_train_metrics, step)

                generate_image(logger, net, 'val', dataset.test_images(), get_features, get_target,
                               net_input_size, net_output_size, step)

                generate_image(logger, net, 'train', dataset.train_images(), get_features, get_target,
                               net_input_size, net_output_size, step)



if __name__ == "__main__":
    main()
