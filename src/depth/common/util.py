
import yaml


def yaml_parser(path='../../properties.yml'):
    with open(path) as f:
        props = yaml.load(f, Loader=yaml.FullLoader)
    return props


