import yaml
import itertools
import logging
import utils
import os
from utils import notifier


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


def make_ordered_pair(model0, model1):
    if model0 <= model1:
        return model0, model1, True
    else:
        return model1, model0, False


def find_num_requests(model0, model1, library):
    if model0 <= model1:
        return library[(model0, model1)]
    else:
        val1, val0 = library[(model1, model0)]
        return val0, val1


def canonical_name(model_name, model_config):
    if model_name in ['vision', 'vision1']:
        return model_config['arch']
    else:
        return model_name

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
    model0_names = ['vision']
    model1_names = ['vision1']
    model0_modes = ['eval']
    model1_modes = ['eval']
    model_to_kwargs = {
        'transformer': {
            'batch_size': [4],
            'arch': ['base'],
        },
        'vision': {
            'batch_size': [4],
            'arch': ['resnet50', 'mobilenet_v2'],
        },
        'vision1': {
            'batch_size': [4],
            'arch': ['resnet101', 'mobilenet_v2'],
        },
        'bert': {
            'batch_size': [2],
            'arch': ['base']
        },
        'retinanet': {
            'batch_size': [4]
        },
    }

    policies = ['MPS-process']
    skip_identical_models = False
    skip_heterogeneous_models = False
    use_dummy_data = True
    request_rate = 0

    model_pair_to_num_requests = {
        ('resnet101', 'resnet50'): (550, 1000),
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
    # ----configuration region ended----

    default_full_config['shared_config']['use_dummy_data'] = use_dummy_data
    default_full_config['shared_config']['request_rate'] = request_rate
    for model0 in model0_names:
        default_full_config['model0']['name'] = model0
        for model0_mode in model0_modes:
            logging.info(f"model0 {model0} with mode {model0_mode}")
            default_full_config['model0']['mode'] = model0_mode
            model0_default_config = default_full_config[model0]
            for model1 in model1_names:
                default_full_config['model1']['name'] = model1
                for model1_mode in model1_modes:
                    logging.info(f"model1 {model1} with mode {model1_mode}")
                    default_full_config['model1']['mode'] = model1_mode
                    model1_default_config = default_full_config[model1]
                    for policy in policies:
                        logging.info(f'policy {policy}')
                        default_full_config['policy'] = policy
                        if model0 == model1:
                            if skip_identical_models:
                                continue
                            # only generate config once
                            for model_config, name in generate_configs(model0_default_config, **model_to_kwargs[model0]):
                                model0_canonical_name = canonical_name(model0, model_config)
                                num_reqs, _ = find_num_requests(model0_canonical_name, model0_canonical_name, model_pair_to_num_requests)
                                model_config['num_requests'] = num_reqs
                                default_full_config[model0] = model_config
                                combination_name = f'{model0_mode}-{model0}-{model1_mode}-{model1}-{name}-{policy}-dummy-{use_dummy_data}'
                                run(default_full_config, combination_name)
                        else:
                            if skip_heterogeneous_models:
                                continue
                            for model0_config, name0 in generate_configs(model0_default_config, **model_to_kwargs[model0]):
                                for model1_config, name1 in generate_configs(model1_default_config, **model_to_kwargs[model1]):
                                    model0_canonical_name = canonical_name(model0, model0_config)
                                    model1_canonical_name = canonical_name(model1, model1_config)
                                    reqs0, reqs1 = find_num_requests(model0_canonical_name, model1_canonical_name, model_pair_to_num_requests)
                                    model0_config['num_requests'] = reqs0
                                    model1_config['num_requests'] = reqs1
                                    default_full_config[model0] = model0_config
                                    default_full_config[model1] = model1_config
                                    combination_name = f'{model0_mode}-{model0}-{name0}-{model1_mode}-{model1}-{name1}-{policy}-dummy-{use_dummy_data}'
                                    run(default_full_config, combination_name)

    notifier.notify(
        subject='A set of experiments have finished',
        body=utils.dict2pretty_str({
            'model0_names': model0_names,
            'model0_modes': model0_modes,
            'model1_modes': model1_modes,
            'model1_names': model1_names,
            'policies': policies,
            'model_to_kwargs': model_to_kwargs
        })
    )


