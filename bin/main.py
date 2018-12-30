# Initialize all the required paths.
import init_paths

from py_perception.task_runner import TaskRunner

import os
import yaml

BASE_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../'
CONFIG_FILE_PATH = os.path.join(BASE_DIR, 'config/CONFIG.yaml')


def parse_yml(config_file_path):
    """ Helper to read and parse YAML file.

    :param config_file_path: path where to load file from.
    :return: dict() of the key : value pair contents of the file.
    """
    with open(config_file_path, 'r') as f:
        file = f.read()

    return yaml.load(file)


def main():
    config = parse_yml(CONFIG_FILE_PATH)

    task_runner = TaskRunner(config['TaskConfig'], BASE_DIR)
    task_runner.run_perception_task()


if __name__ == '__main__':
    main()
