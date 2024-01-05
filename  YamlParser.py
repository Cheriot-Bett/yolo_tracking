import yaml

class YamlParser:
    def __init__(self):
        self.config = {}

    def merge_from_file(self, file_path):
        with open(file_path, 'r') as file:
            yaml_config = yaml.safe_load(file)
            self.config.update(yaml_config)

    def get_value(self, key):
        return self.config.get(key)