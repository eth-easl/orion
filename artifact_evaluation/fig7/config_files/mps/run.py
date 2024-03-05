import yaml
import itertools
import logging
import os

mnames = {
    'resnet50': "ResNet50",
    'mobilenet_v2': "MobileNetV2",
    'resnet101': 'ResNet101',
    'bert': 'BERT',
    'transformer': 'Transformer'
}

def run(model0, model1, config, combination_name, times=1, start_id = 0):

    config_file_name = f'gen_conf_{combination_name}.yaml'

    logging.info(f'dump config to {config_file_name}')
    with open(f'./{config_file_name}', 'w') as file:
        yaml.dump(config, file)
    # run python main.py
    logging.info(f'training with this config {times} times')


    for i in range(start_id, start_id + times):
        log_file = f'log_{i}_{combination_name}.log'
        os.system(f"python3.8 {os.path.expanduser( '~' )}/orion/related/baselines/main.py --config ./{config_file_name}")
        print(f"{combination_name}.log.json")
        os.system(f"cp {combination_name}.log.json ../../results/mps/{mnames[model0]}_{mnames[model1]}_{i}.json")



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
    model0_mode = 'eval'
    model1_mode = 'train'

    policy = 'MPS'

    train_batch_sizes = {
        'resnet50': 32,
        'mobilenet_v2': 64,
        'resnet101': 32,
        'bert': 8,
        'transformer': 8
    }

    eval_batch_sizes = {
        'resnet50': 4,
        'mobilenet_v2': 4,
        'resnet101': 4,
        'bert': 2,
        'transformer': 4
    }

    request_rates = {
        'resnet50': 15,
        'mobilenet_v2': 40,
    }

    num_reqs = {
        'resnet50': 9200,
        'mobilenet_v2': 12000,
    }

    models = ['resnet50', 'mobilenet_v2', 'resnet101', 'bert', 'transformer']
    combinations = itertools.product(models[:2], models)
    times = 3
    start_id = 0
    distribution = 'poisson'


    # ----configuration region ended----

    default_full_config['shared_config']['distribution'] = distribution

    for model0, model1 in combinations:
        default_full_config['models']['model0']['name'] = model0
        default_full_config['models']['model0']['mode'] = model0_mode
        default_full_config['models']['model1']['name'] = model1 if model0 != model1 else model1 + '-1'
        default_full_config['models']['model1']['mode'] = model1_mode
        default_full_config['policy'] = policy

        if model0 != model1:

            default_full_config[model0]['request_rate'] = request_rates[model0]
            default_full_config[model0]['num_iterations'] = num_reqs[model0]

            default_full_config[model0]['batch_size'] = eval_batch_sizes[model0]
            default_full_config[model1]['batch_size'] = train_batch_sizes[model1]

            combination_name = f'{model0_mode}-{model0}{model1_mode}-{model1}'
            run(model0, model1, default_full_config, combination_name, times=times, start_id=start_id)
        else:
            model1_with_suffix = model1 + '-1'

            default_full_config[model0]['request_rate'] = request_rates[model0]
            default_full_config[model0]['num_iterations'] = num_reqs[model0]

            default_full_config[model0]['batch_size'] = eval_batch_sizes[model0]
            default_full_config[model1_with_suffix]['batch_size'] = train_batch_sizes[model1]

            combination_name = f'{model0_mode}-{model0}{model1_mode}-{model1_with_suffix}'
            run(model0, model1, default_full_config, combination_name, times=times, start_id=start_id)