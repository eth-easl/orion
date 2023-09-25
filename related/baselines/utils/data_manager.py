import json


class DataManager:
    """
    A class to encapsulate all the logic regarding writing the structured experiment results to a json file.
    """

    def __init__(self, experiment_data_json_file):
        self.experiment_data_json_file = experiment_data_json_file
        # init the file
        self._dump_dict({})

    def write_kv(self, key, value):
        """
        Write the key-value pair to the json data file.

        This method is NOT thread/process-safe, the caller needs a
        synchronization mechanism, e.g. a lock, to ensure at most one writer exists at any time.
        """
        with open(self.experiment_data_json_file, 'r') as f:
            dict_data = json.load(f)

        dict_data[key] = value
        self._dump_dict(dict_data)

    def write_kvs(self, kv_pairs):
        """
        Write many key-value pairs to the json data file.

        This method is NOT thread/process-safe, the caller needs a
        synchronization mechanism, e.g. a lock, to eusure at most one writer exists at any time.
        """
        dict_data = self.read_dict()

        dict_data.update(kv_pairs)
        self._dump_dict(dict_data)

    def _dump_dict(self, dict_data):
        with open(self.experiment_data_json_file, 'w') as f:
            json.dump(dict_data, f, indent=4)

    def read_dict(self):
        with open(self.experiment_data_json_file, 'r') as f:
            dict_data = json.load(f)

        return dict_data
