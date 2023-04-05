from datetime import datetime
import json
import itertools

def pretty_time():
    return datetime.now().strftime('%d-%m-%Y-%H-%M-%S')


def dict2pretty_str(dict_data):
    return json.dumps(dict_data, indent=4)


class DummyDataLoader:
    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        return itertools.repeat(self.batch)


