import json
import os
from string import Template

if __name__ == "__main__":
    directory = '/Users/sherlock/programs/gpu_share_data/'
    file_name_template = Template('log_vision-vision-batch_size-${batch_size}-arc-${arc}-${policy}-dummy-True.log.json')
    for policy in ['tick-tock', 'temporal']:
        print(policy + '-table')
        for arc in ['resnet50', 'mobilenet_v2']:
            for batch_size in [32, 64]:
                filename = file_name_template.substitute(
                    policy=policy,
                    batch_size=batch_size,
                    arc=arc
                )
                # print(f'now parse file {filename}')
                file_fullname = os.path.join(directory, filename)
                with open(file_fullname, 'r') as f:
                    data = json.load(f)
                duration = data['duration']
                print(f'{arc} with {batch_size}: {round(duration, 3)}')




