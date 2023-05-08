import yaml
import itertools
import logging
import utils
import os
from utils import notifier
import json
import statistics as stats

model_pair_to_num_iters_train_inf = {
    ('resnet50', 'resnet50'): (300, 499),
    ('resnet50', 'mobilenet_v2'): (300, 594),
    ('resnet50', 'resnet101'): (300, 282),
    ('resnet50', 'bert'): (700, 235),
    ('resnet50', 'transformer'): (700, 494),

    ('mobilenet_v2', 'resnet50'): (300, 366),
    ('mobilenet_v2', 'mobilenet_v2'): (300, 479),
    ('mobilenet_v2', 'resnet101'): (300, 236),
    ('mobilenet_v2', 'bert'): (700, 192),
    ('mobilenet_v2', 'transformer'): (700, 411),

    ('resnet101', 'resnet50'): (300, 802),
    ('resnet101', 'mobilenet_v2'): (300, 1016),
    ('resnet101', 'resnet101'): (300, 471),
    ('resnet101', 'bert'): (400, 201),
    ('resnet101', 'transformer'): (400, 461),

    ('bert', 'resnet50'): (300, 1115),
    ('bert', 'mobilenet_v2'): (300, 1447),
    ('bert', 'resnet101'): (300, 646),
    ('bert', 'bert'): (300, 208),
    ('bert', 'transformer'): (300, 462),

    ('transformer', 'resnet50'): (300, 1085),
    ('transformer', 'mobilenet_v2'): (300, 1448),
    ('transformer', 'resnet101'): (300, 675),
    ('transformer', 'bert'): (300, 185),
    ('transformer', 'transformer'): (300, 434)
}


model_pair_to_num_iters_train_train = {
    ('resnet50', 'resnet50'): (300, 300),
    ('resnet50', 'mobilenet_v2'): (400, 400),
    ('resnet50', 'resnet101'): (400, 300),
    ('resnet50', 'bert'): (700, 250),
    ('resnet50', 'transformer'): (1000, 300),

    ('mobilenet_v2', 'mobilenet_v2'): (300, 300),
    ('mobilenet_v2', 'resnet101'): (500, 300),
    ('mobilenet_v2', 'bert'): (800, 300),
    ('mobilenet_v2', 'transformer'): (1000, 300),

    ('resnet101', 'resnet101'): (300, 300),
    ('resnet101', 'bert'): (500, 300),
    ('resnet101', 'transformer'): (450, 300),

    ('bert', 'bert'): (300, 300),
    ('bert', 'transformer'): (345, 350),
    ('transformer', 'transformer'): (300, 300)
}

model_pair_to_num_requests_infer_infer = {
    ('resnet50', 'bert'): (1000, 350),
    ('mobilenet_v2', 'bert'): (1000, 200),
    ('resnet101', 'bert'): (500, 350),
    ('bert', 'bert'): (500, 500),
    ('bert', 'transformer'): (350, 500),

    ('resnet101', 'resnet50'): (550, 1000),
    ('mobilenet_v2', 'resnet101'): (1000, 450),
    ('resnet101', 'resnet101'): (1000, 1000),
    ('resnet101', 'transformer'): (500, 500),

    ('resnet50', 'transformer'): (550, 300),
    ('resnet50', 'resnet50'): (1000, 1000),
    ('resnet50', 'retinanet'): (1000, 150),

    ('retinanet', 'retinanet'): (200, 200),
    ('retinanet', 'transformer'): (100, 350),

    ('mobilenet_v2', 'transformer'): (1000, 350),
    ('mobilenet_v2', 'retinanet'): (1000, 90),
    ('mobilenet_v2', 'resnet50'): (1250, 1000),
    ('mobilenet_v2', 'mobilenet_v2'): (1000, 1000),
    ('mobilenet_v2', 'resnet101'): (1000, 450),
    ('transformer', 'transformer'): (500, 500),

}


model_pair_to_num_requests_infer_infer_uniform_high = {
    ('resnet50', 'resnet50'): (1000, 1000),
    ('resnet50', 'mobilenet_v2'): (1000, 1250),
    ('resnet50', 'resnet101'): (1000, 500),
    ('resnet50', 'bert'): (1000, 100),
    ('resnet50', 'transformer'): (1000, 250),

    ('mobilenet_v2', 'mobilenet_v2'): (1000, 1000),
    ('mobilenet_v2', 'resnet101'): (1000, 400),
    ('mobilenet_v2', 'bert'): (1000, 125),
    ('mobilenet_v2', 'transformer'): (1000, 200),

    ('resnet101', 'resnet101'): (1000, 1000),
    ('resnet101', 'bert'): (1000, 200),
    ('resnet101', 'transformer'): (1000, 500),

    ('bert', 'bert'): (500, 500),
    ('bert', 'transformer'): (400, 1000),

    ('transformer', 'transformer'): (500, 500)
}

def generate_configs(default_config, **kwargs):
    for values in itertools.product(*kwargs.values()):
        for kw_id, kw in enumerate(kwargs.keys()):
            value = values[kw_id]
            default_config[kw] = value
        logging.info(f"generated config: {utils.dict2pretty_str(default_config)}")

        unique_name = '-'.join(f'{key}-{value}' for key, value in zip(kwargs.keys(), values))
        yield default_config, unique_name


def run(config, combination_name, times=1, start_id = 0):

    config_file_name = f'gen_conf_{combination_name}.yaml'

    logging.info(f'dump config to {config_file_name}')
    with open(f'./{config_file_name}', 'w') as file:
        yaml.dump(config, file)
    # run python main.py
    logging.info(f'training with this config {times} times')


    for i in range(start_id, start_id + times):
        log_file = f'log_{i}_{combination_name}.log'
        os.system(f"python main.py --log ./{log_file} --config ./{config_file_name}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s: [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%d/%m/%Y %H:%M:%S',
        handlers=[
            # output to console
            logging.StreamHandler(),
        ]
    )
    with open('./config.yaml', 'r') as file:
        default_full_config = yaml.load(file, Loader=yaml.FullLoader)

    # ----configuration region started----
    model0_mode = 'train'
    model1_mode = 'train'

    policy = 'time-slice'


    train_batch_sizes = {
        'resnet50': 32,
        'mobilenet_v2': 64,
        'resnet101': 32,
        'bert': 8,
        'transformer': 8
    }


    combinations = list(model_pair_to_num_iters_train_inf.keys())
    times = 1


    # ----configuration region ended----

    # default_full_config['shared_config']['distribution'] = distribution

    for model0, model1 in combinations:
        default_full_config['model0']['name'] = model0
        # default_full_config['model0']['mode'] = model0_mode
        default_full_config['model1']['name'] = model1 # if model0 != model1 else model1 + '-1'
        # default_full_config['model1']['mode'] = model1_mode
        default_full_config['policy'] = policy

        combination_name = f'{model0_mode}-{model0}-{model1_mode}-{model1}-{policy}'
        run(default_full_config, combination_name, times=times)
        # if model0 != model1:
        #     # if model0 == 'bert':
        #     #     # for training use bert-base
        #     #     default_full_config[model0]['arch'] = 'base'
        #     # if model1 == 'bert':
        #     #     # for evaluation use bert-large
        #     #     default_full_config[model1]['arch'] = 'large'
        #
        #     default_full_config['policy'] = policy
        #     # num_reqs0, num_reqs1 = model_pair_to_num_requests_infer_infer_uniform_high[(model0, model1)]
        #     # default_full_config[model0]['request_rate'] = request_rates[model0]
        #     default_full_config[model1]['request_rate'] = request_rates[model1]
        #
        #     default_full_config[model1]['num_requests'] = models2num_reqs[model0][model1]
        #
        #     default_full_config[model0]['batch_size'] = train_batch_sizes[model0]
        #     default_full_config[model1]['batch_size'] = eval_batch_sizes[model1]
        #
        #     combination_name = f'{model0_mode}-{model0}-{model1_mode}-{model1}-{policy}'
        #     run(default_full_config, combination_name, times=times)
        # else:
        #     model1_with_suffix = model1 + '-1'
        #     if model0 == 'bert':
        #         # for training use bert-base
        #         default_full_config[model0]['arch'] = 'base'
        #     if model1 == 'bert':
        #         # for evaluation use bert-large
        #         default_full_config[model1_with_suffix]['arch'] = 'large'
        #
        #     default_full_config['policy'] = policy
        #     # num_reqs0, num_reqs1 = model_pair_to_num_requests_infer_infer_uniform_high[(model0, model1)]
        #
        #     # default_full_config[model0]['request_rate'] = request_rates[model0]
        #     default_full_config[model1_with_suffix]['request_rate'] = request_rates[model1]
        #
        #     default_full_config[model1_with_suffix]['num_requests'] = models2num_reqs[model0][model1]
        #
        #     default_full_config[model0]['batch_size'] = train_batch_sizes[model0]
        #     default_full_config[model1_with_suffix]['batch_size'] = eval_batch_sizes[model1]
        #
        #
        #     combination_name = f'{model0_mode}-{model0}-{model1_mode}-{model1}-{policy}'
        #     run(default_full_config, combination_name, times=times)



    notifier.notify(
        subject='A set of experiments have finished',
        body=utils.dict2pretty_str({
            'combinations': combinations
        })
    )


