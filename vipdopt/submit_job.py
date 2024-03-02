"""Run the Vipdopt software package."""
import os
import sys

sys.path.append(os.getcwd())    # 20240219 Ian: Only added this so I could debug some things from my local VSCode

import re
import subprocess


def generate_script(filename: str, nnodes: int, *args, **kwargs):
    bash_args = ['bash', 'vipdopt/submit.sh', filename, str(nnodes)]
    bash_args += [str(arg) for arg in args]
    for k, v in kwargs.items():
        bash_args += [f'--{k}', str(v)]

    subprocess.call(bash_args)


if __name__ == '__main__':
    if re.search(r'-h(?!\S)', ' '.join(sys.argv)) or re.search(r'--help', ' '.join(sys.argv)):
        print('Usage: submit_job OUTPUT NUMBER_OF_NODES PROJECT_DIRECTORY [ARGS]...')
        sys.exit()

    generate_script(*sys.argv[1:])
