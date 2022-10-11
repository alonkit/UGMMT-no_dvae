import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import os
import math


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Translator(nn.Module):
    scalar = 100000
    fp_and_emb = 2304
    fp = 2048
    emb = 256
    fp_and_emb_sqrt = int(math.sqrt(fp_and_emb))
    emb_sqrt = int(math.sqrt(emb))
    def __init__(self, n_residual_blocks=4):
        super(Translator, self).__init__()

        # for fp attention
        model0 = [
                  nn.Linear(2048, self.fp, bias=False),
        ]
        self.model0 = nn.Sequential(*model0)

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(1, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling block
        in_features = 64
        out_features = in_features*2
        for _ in range(1):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling block
        out_features = in_features//2
        for _ in range(1):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output convolution
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, 1, 7),
                    nn.Tanh()]

        self.model = nn.Sequential(*model)

        model2 = [  nn.Linear(self.fp_and_emb, self.fp_and_emb//2),
                    nn.BatchNorm1d(num_features=self.fp_and_emb//2),
                    nn.LeakyReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(self.fp_and_emb//2, self.emb)]
        self.model2 = nn.Sequential(*model2)

        model_no_fp = [ nn.Linear(self.emb, self.emb),
                        nn.BatchNorm1d(num_features=self.emb),
                        nn.LeakyReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(self.emb, self.emb)]

        self.model_no_fp = nn.Sequential(*model_no_fp)

    def forward(self, x, fp=None):
        if fp is not None:
            w = self.model0(fp)
            w_prob = F.softmax(w, dim=1)
            fp = self.scalar * fp * w_prob
            x = torch.cat((x, fp), dim=1)
            x = self.model(x.view(x.shape[0], 1, self.fp_and_emb_sqrt, self.fp_and_emb_sqrt))
            x = self.model2(x.view(x.shape[0], -1))
        else:       # for ablation no fp
            x = self.model(x.view(x.shape[0], 1, self.emb_sqrt, self.emb_sqrt))
            x = self.model_no_fp(x.view(x.shape[0], -1))
        return x


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        self.decay_start_epoch = decay_start_epoch
        self.offset = offset
        self.n_epochs = n_epochs

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


# for ablation
class ReplayBuffer():
    def __init__(self, max_size=50):
        self.data = []
        self.max_size = max_size

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


# hold, update and print statistics
class Statistics(object):
    # Total
    loss_epoch = []

    # cycle constraints
    loss_cycle_epoch = []

    # KL for ablation
    loss_kl_epoch = []

    def __init__(self):
        return

    def update(self, loss, cycle_loss, kl_loss):
        # Total
        self.loss_epoch.append(loss)

        # cycle constraints
        self.loss_cycle_epoch.append(cycle_loss)

        # KL for ablation
        self.loss_kl_epoch.append(kl_loss)

    def print(self):
        # Total
        Average_loss = sum(self.loss_epoch) / len(self.loss_epoch)
        print('Average loss = ' + str(Average_loss.tolist()))

        # cycle constraints
        if any(self.loss_cycle_epoch):
            Average_loss_cycle = sum(self.loss_cycle_epoch) / len(self.loss_cycle_epoch)
            print('Average loss_cycle = ' + str(Average_loss_cycle.tolist()))


        # KL for ablation
        if any(self.loss_kl_epoch):
            Average_loss_kl = sum(self.loss_kl_epoch) / len(self.loss_kl_epoch)
            print('Average loss_kl = ' + str(Average_loss_kl.tolist()))


def save_checkpoint(current_criterion, best_criterion, T, E, args):
    # first model or best model so far
    if best_criterion is None or current_criterion > best_criterion:
        best_criterion = current_criterion
        saved_state = dict(best_criterion=best_criterion,
                           T=T.state_dict(),
                           E=E)
        checkpoint_filename_path = args.checkpoints_folder + '/' + args.property + '/checkpoint_model.pth'
        torch.save(saved_state, checkpoint_filename_path)
        print('*** Saved checkpoint in: ' + checkpoint_filename_path + ' ***')
    return best_criterion


def load_checkpoint(args, device):
    checkpoint_filename_path = args.checkpoints_folder + '/' + args.property + '/checkpoint_model.pth'
    if os.path.isfile(checkpoint_filename_path):
        print('*** Loading checkpoint file ' + checkpoint_filename_path)
        saved_state = torch.load(checkpoint_filename_path, map_location=device)

        # create embedding translator
        T_AB = Translator().to(device)
        T_BA = Translator().to(device)

        best_criterion = saved_state.get('best_criterion')
        T_AB.load_state_dict(saved_state['T'])
        T_BA.load_state_dict(saved_state['T_BA'])
        E = saved_state['E']
    return T, E, best_criterion