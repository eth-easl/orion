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
    model0_names = ['retinanet']
    model1_names = ['retinanet', 'vision', 'transformer', 'bert']

    model_to_kwargs = {
        'transformer': {
            'arch': ['base'],
            'batch_size': [8, 16]
        },
        'vision': {
            'batch_size': [32, 64],
            'arc': ['mobilenet_v2', 'resnet50'],
        },
        'bert': {
            'batch_size': [8, 16],
            'arch': ['base']
        },
        'retinanet': {
            'batch_size': [2, 4]
        },
        'gnmt': {
            'batch_size': [64, 32]
        }
    }

    policies = ['tick-tock', 'temporal']
    skip_identical_models = False
    # ----configuration region ended----

    for model0 in model0_names:
        logging.info(f"model0 {model0}")
        default_full_config['model0'] = model0
        model0_default_config = default_full_config[model0]
        for model1 in model1_names:
            logging.info(f"model1 {model1}")
            default_full_config['model1'] = model1
            model1_default_config = default_full_config[model1]
            for policy in policies:
                logging.info(f'policy {policy}')
                default_full_config['policy'] = policy
                if model0 == model1:
                    if skip_identical_models:
                        continue
                    # only generate config once
                    for model_config, name in generate_configs(model0_default_config, **model_to_kwargs[model0]):
                        default_full_config[model0] = model_config
                        combination_name = f'{model0}-{model1}-{name}-{policy}'
                        run(default_full_config, combination_name)
                else:
                    for model0_config, name0 in generate_configs(model0_default_config, **model_to_kwargs[model0]):
                        for model1_config, name1 in generate_configs(model1_default_config, **model_to_kwargs[model1]):
                            default_full_config[model0] = model0_config
                            default_full_config[model1] = model1_config
                            combination_name = f'{model0}-{name0}-{model1}-{name1}-{policy}'
                            run(default_full_config, combination_name)

    notifier.notify(
        subject='A set of experiments have finished',
        body=utils.dict2pretty_str({
            'model0_names': model0_names,
            'model1_names': model1_names,
            'policies': policies,
            'model_to_kwargs': model_to_kwargs
        })
    )


