from datetime import datetime
import json


def pretty_time():
    return datetime.now().strftime('%d-%m-%Y-%H-%M-%S')


def dict2pretty_str(dict_data):
    return json.dumps(dict_data, indent=4)


class DummyDataLoader:
    def __init__(self, data, target, iterations):
        self.data = data
        self.target = target
        self.iterations = iterations

    def __iter__(self):
        return ((self.data, self.target) for _ in range(self.iterations))


