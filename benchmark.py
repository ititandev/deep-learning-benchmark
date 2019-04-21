import argparse
from collections import OrderedDict
from importlib import import_module
import pickle

import numpy as np

frameworks = [
    'pytorch',
    'tensorflow',
    'caffe2'
]

models = [
    'vgg16',
    'resnet152',
]

batch_sizes = [
    4,
    8,
    16,
    # 32
]

class Benchmark():

    def get_framework_model(self, framework, model):
        framework_model = import_module('.'.join(['frameworks', framework, 'models']))
        return getattr(framework_model, model)

    def benchmark_model(self, mode, framework, model, batch_size=16, precision = 'fp32', image_shape=(224, 224), num_iterations=20, num_warmups=20):
        framework_model = self.get_framework_model(framework, model)(precision, image_shape, batch_size)
        durations = framework_model.eval(num_iterations, num_warmups) if mode == 'eval' else framework_model.train(num_iterations, num_warmups)
        durations = np.array(durations)
        return durations.mean() * 1000

    def benchmark_all(self):
        results = OrderedDict()
        for framework in frameworks:
            results[framework] = self.benchmark_framework(framework)
        return results

    def benchmark_framework(self, framework, num_iterations=20, num_warmups=20):
        results = OrderedDict()

        for batch_size in batch_sizes:
            results[batch_size] = []
            for model in models:
                eval_duration = self.benchmark_model('eval', framework, model, batch_size)
                train_duration = self.benchmark_model('train', framework, model, batch_size)
                print("{}'s {} eval at batch size {}: {}ms avg".format(framework, model, batch_size, round(eval_duration, 1)))
                print("{}'s {} train at batch size {}: {}ms avg".format(framework, model, batch_size, round(train_duration, 1)))
                results[batch_size].append(eval_duration)
                results[batch_size].append(train_duration)
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='framework', required=False)
    parser.add_argument('-i', dest='num_iterations', required=False)
    parser.add_argument('-w', dest='num_warmups', required=False)
    args = parser.parse_args()

    if args.framework:
        print('RUNNING BENCHMARK FOR FRAMEWORK', args.framework)
        results = Benchmark().benchmark_framework(args.framework, args.num_iterations, args.num_warmups)
    else:
        print('RUNNING BENCHMARK FOR FRAMEWORK', frameworks)
        results = Benchmark().benchmark_all()