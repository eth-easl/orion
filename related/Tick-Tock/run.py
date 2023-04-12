import yaml
import itertools
import logging
import utils
import os
from utils import notifier
import json


def generate_configs(default_config, **kwargs):
    for values in itertools.product(*kwargs.values()):
        for kw_id, kw in enumerate(kwargs.keys()):
            value = values[kw_id]
            default_config[kw] = value
        logging.info(f"generated config: {utils.dict2pretty_str(default_config)}")

        unique_name = '-'.join(f'{key}-{value}' for key, value in zip(kwargs.keys(), values))
        yield default_config, unique_name


def run(config, combination_name):

    config_file_name = f'gen_conf_{combination_name}.yaml'
    log_file = f'log_{combination_name}.log'
    logging.info(f'dump config to {config_file_name}')
    with open(f'./{config_file_name}', 'w') as file:
        yaml.dump(config, file)
    # run python main.py
    logging.info('training with this config...')
    os.system(f"python main.py --log ./{log_file} --config ./{config_file_name}")
    logging.info('training finished.')

    # report some statistics
    json_file =f'{log_file}.json'
    try:
        with open(f'./{json_file}', 'r') as file:
            dict_data = json.load(file)

        model0 = config['model0']['name']
        model1 = config['model1']['name']
        round_func = lambda val: round(val, 2)

        logging.info(f"results for {model0} and {model1}")
        logging.info(f"duration for each: {round_func(dict_data['duration0'])} / {round_func(dict_data['duration1'])}")
        logging.info(f"duration: {round_func(dict_data['duration'])}")
    except:
        logging.info("the json data file cannot be opened")




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

    policies = ['MPS-process']
    use_dummy_data = True
    request_rate = 0

    model_pair_to_num_iters_train_train = {
        ('resnet50', 'resnet50'): (300, 300),
        ('resnet50', 'mobilenet_v2'): (400, 300),
        ('resnet50', 'resnet101'): (400, 300),
        ('resnet50', 'bert'): (1000, 300),
        ('resnet50', 'transformer'): (1000, 300),

        ('mobilenet_v2', 'mobilenet_v2'): (300, 300),
        ('mobilenet_v2', 'resnet101'): (500, 300),
        ('mobilenet_v2', 'bert'): (1000, 300),
        ('mobilenet_v2', 'transformer'): (1000, 300),

        ('resnet101', 'resnet101'): (300, 300),
        ('resnet101', 'bert'): (500, 300),
        ('resnet101', 'transformer'): (400, 300),

        ('bert', 'bert'): (300, 300),
        ('bert', 'transformer'): (300, 400),
        ('transformer', 'transformer'): (300, 300)
    }

    # model_pair_to_num_requests_infer_infer = {
    #     ('resnet50', 'bert'): (1000, 350),
    #     ('mobilenet_v2', 'bert'): (1000, 200),
    #     ('resnet101', 'bert'): (500, 350),
    #     ('bert', 'bert'): (500, 500),
    #     ('bert', 'transformer'): (350, 500),
    #
    #     ('resnet101', 'resnet50'): (550, 1000),
    #     ('mobilenet_v2', 'resnet101'): (1000, 450),
    #     ('resnet101', 'resnet101'): (1000, 1000),
    #     ('resnet101', 'transformer'): (500, 500),
    #
    #     ('resnet50', 'transformer'): (550, 300),
    #     ('resnet50', 'resnet50'): (1000, 1000),
    #     ('resnet50', 'retinanet'): (1000, 150),
    #
    #     ('retinanet', 'retinanet'): (200, 200),
    #     ('retinanet', 'transformer'): (100, 350),
    #
    #     ('mobilenet_v2', 'transformer'): (1000, 350),
    #     ('mobilenet_v2', 'retinanet'): (1000, 90),
    #     ('mobilenet_v2', 'resnet50'): (1250, 1000),
    #     ('mobilenet_v2', 'mobilenet_v2'): (1000, 1000),
    #     ('mobilenet_v2', 'resnet101'): (1000, 450),
    #     ('transformer', 'transformer'): (500, 500),
    #
    # }

    combinations = list(model_pair_to_num_iters_train_train.keys())
    # combinations = [
    #     ('resnet101', 'resnet101'),
    #     ('resnet101', 'transformer'),
    #
    #     # ('resnet50', 'bert'),
    #     # ('mobilenet_v2', 'bert'),
    #     # ('resnet101', 'bert'),
    #     # ('bert', 'bert'),
    #     # ('bert', 'transformer')
    # ]
    # ----configuration region ended----

    default_full_config['shared_config']['use_dummy_data'] = use_dummy_data
    default_full_config['shared_config']['request_rate'] = request_rate
    for model0, model1 in combinations:
        default_full_config['model0']['name'] = model0
        default_full_config['model0']['mode'] = model0_mode
        default_full_config['model1']['name'] = model1
        default_full_config['model1']['mode'] = model1_mode
        for policy in policies:
            default_full_config['policy'] = policy
            num_iters0, num_iters1 = model_pair_to_num_iters_train_train[(model0, model1)]
            default_full_config[model0]['num_iterations'] = num_iters0
            default_full_config[model1]['num_iterations'] = num_iters1
            combination_name = f'{model0_mode}-{model0}-{model1_mode}-{model1}-{policy}-dummy-{use_dummy_data}'
            run(default_full_config, combination_name)

    notifier.notify(
        subject='A set of experiments have finished',
        body=utils.dict2pretty_str({
            'combinations': combinations
        })
    )


