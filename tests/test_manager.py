
from typing import Any, Pattern
import re

import numpy as np
import pytest

from testing import assert_equal
from vipdopt.manager import WorkloadManager

def test_get_node_list():
    man = WorkloadManager()
    nodes = man.get_slurm_nodes()
    print(nodes)
    assert False

