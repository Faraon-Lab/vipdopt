from setuptools import setup
import platform
from pathlib import Path
from typing import Iterable, Generator, TypeVar
import itertools
from configparser import ConfigParser

T = TypeVar('T')

PACKAGE_NAME = 'vipdopt'
VERSION = '2.0.1'
SUPPORTED_LUMERICAL_VERSIONS = 'v2[1-4]*'
LUMERICAL_DEFAULT_PATH_WINDOWS = Path('C:\\Program Files\\Lumerical\\')
LUMERICAL_DEFAULT_PATH_LINUX = Path('/opt/lumerical/')
INSTALLATION_CONFIG_FILE = Path('setup.cfg').absolute()
LUMERICAL_LOCATION_FILE = Path('vipdopt/lumerical.cfg').absolute()

def get_dependencies():
    dependencies = []
    with open('build_requirements.txt') as f:
        for line in f:
            dependencies.append(line.rstrip())
    return dependencies


def ladd_to_all(iterable: Iterable[T], val: T) -> Generator[T, None, None]:
    """Add a value to all memebers of a list on the left."""
    for item in iterable:
        yield val + item


def radd_to_all(iterable: Iterable[T], val: T) -> Generator[T, None, None]:
    """Add a value to all memebers of a list on the right."""
    for item in iterable:
        yield item + val


def split_glob(pattern: str) -> Generator[str, None, None]:
    """Convert a glob pattern into multiple patterns if necessary."""
    if '{' in pattern:
        start = pattern.index('{')
        end = pattern.index('}')

        prefix = pattern[:start]
        choices = pattern[start + 1 : end].split(',')
        for choice in choices:
            yield from ladd_to_all(
                split_glob(pattern[end + 1 :]), prefix + choice.strip()
            )
    else:
        yield pattern


def glob_last(search_dir: Path, pattern: str) -> Path:
    """Find the last file to match the given pattern."""
    patterns = list(split_glob(pattern))
    matches = list(itertools.chain.from_iterable(map(search_dir.glob, patterns)))
    if len(matches) == 0:
        msg = f'No path found with pattern {pattern} in {search_dir}'
        raise FileNotFoundError(msg)
    matches.sort()
    return matches[-1]

def find_lumapi_path(base_diretory: Path, version=SUPPORTED_LUMERICAL_VERSIONS):
    """Get the path of lumapi from the latest version of Lumerical."""
    try:
        latest_version = glob_last(base_diretory, version)
    except FileNotFoundError as e:
        raise SystemError(
            f'Lumerical could not be found in default installation directory: {str(base_diretory)}. '
            'If Lumerical is installed in another directory please specify the path to lumapi.py inside "setup.cfg"'
        ) from e
    print('Succesfully located Lumerical API file!')
    return latest_version / 'api/python/lumapi.py'


if __name__ == '__main__':
    with open('README.md') as f:
        readme = f.read()

    platform = platform.system()
    match platform:
        case 'Windows':
            default_directory = LUMERICAL_DEFAULT_PATH_WINDOWS
        case 'Liinux':
            default_directory = LUMERICAL_DEFAULT_PATH_LINUX
        case _:
            raise NotImplementedError(f'vipdopt does not support platform "{platform}"')

    config = ConfigParser()
    config.read(INSTALLATION_CONFIG_FILE)

    with open(INSTALLATION_CONFIG_FILE, 'r') as f:
        print(f.read())

    lumerical_version = config['Lumerical Options']['lumerical_version']
    if lumerical_version == '':
        lumerical_version = SUPPORTED_LUMERICAL_VERSIONS

    lumapi_path = Path(config['Lumerical Options']['lumapi_path'])
    if lumapi_path == Path(''):
        print(f'No lumapi file specified in "setup.cfg". Searching in default installation location: {default_directory}')
        lumapi_path = find_lumapi_path(default_directory, version=lumerical_version)
    elif not lumapi_path.exists():
        print(f'lumapi file "{lumapi_path}" could not be found. Searching in default installation location: {default_directory}')
        lumapi_path = find_lumapi_path(default_directory, version=lumerical_version)
    
    config = ConfigParser(comment_prefixes='/', allow_no_value=True)
    print(f'Proceeding with installation using Lumerical API at path: {str(lumapi_path)}')
    config.read_file(open(LUMERICAL_LOCATION_FILE))
    config.set('Lumerical', 'lumapi_path', str(lumapi_path.absolute()))
    with open(LUMERICAL_LOCATION_FILE, 'w') as f:
        config.write(f)

    setup()