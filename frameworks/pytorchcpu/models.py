import torchvision
from time import time

import torch
# this turns on auto tuner which optimizes performance
# torch.backends.cudnn.benchmark = True

# print('cuda version=', torch.version.cuda)
# print('cudnn version=', torch.backends.cudnn.version())

torch.set_num_threads(40)

class pytorch_base:

    def __init__(self, model_name, precision, image_shape, batch_size):
        self.model = getattr(torchvision.models, model_name)() if precision == 'fp32' \
            else getattr(torchvision.models, model_name)().half()
        x = torch.rand(batch_size, 3, image_shape[0], image_shape[1])
        self.eval_input = torch.autograd.Variable(x, requires_grad=False) if precision == 'fp32' \
            else torch.autograd.Variable(x, requires_grad=False).half()
        self.train_input = torch.autograd.Variable(x, requires_grad=True) if precision == 'fp32' \
            else torch.autograd.Variable(x, requires_grad=True).half()

    def eval(self, num_iterations, num_warmups):
        self.model.eval()
        durations = []
        for i in range(num_iterations + num_warmups):
            # torch.cuda.synchronize()
            t1 = time()
            self.model(self.eval_input)
            # torch.cuda.synchronize()
            t2 = time()
            if i >= num_warmups:
                durations.append(t2 - t1)
        return durations

    def train(self, num_iterations, num_warmups):
        self.model.train()
        durations = []
        for i in range(num_iterations + num_warmups):
            # torch.synchronize()
            t1 = time()
            self.model.zero_grad()
            out = self.model(self.train_input)
            loss = out.sum()
            loss.backward()
            # torch.synchronize()
            t2 = time()
            if i >= num_warmups:
                durations.append(t2 - t1)
        return durations


class vgg16(pytorch_base):

    def __init__(self, precision, image_shape, batch_size):
        super().__init__('vgg16', precision, image_shape, batch_size)


class resnet152(pytorch_base):

    def __init__(self, precision, image_shape, batch_size):
        super().__init__('resnet152', precision, image_shape, batch_size)
