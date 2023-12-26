"""Tests for vipdopt.utils"""

import logging
import re
from pathlib import Path

import numpy as np
import pytest

from testing import assert_close, assert_equal
from vipdopt.utils import read_config_file, sech, setup_logger

TEST_YAML_PATH = 'vipdopt/configuration/config_example.yml'

LOG_REGEX = \
    r'\d{4}(-\d\d){2} (\d\d:){2}\d{2}.*- (INFO|DEBUG|WARNING|ERROR|CRITICAL) - .*'


@pytest.mark.parametrize(
        'level, msg',
        [
            (logging.INFO, 'short message'),
            (logging.DEBUG, 'looooooooooong message'),
            (logging.WARNING, "cOmPleX MeSSage with \\ weird ' characters '"),
            (logging.ERROR, 'message with\nnew lines\n'),
            (logging.WARNING, ''),
            (logging.CRITICAL, 'OHME OH MY\n\n\n'),
            (logging.WARNING, 'o' * 1000),
        ]
)
@pytest.mark.smoke()
def test_logger_setup(capsys, tmp_path, level: int, msg: str):
    log_file = tmp_path / 'test.log'
    logger = setup_logger(name='test_log', level=level, log_file=log_file)

    # call the logger with every log level
    messages = tuple(f'{s} {msg}'
                     for s in ('debug', 'info', 'warning', 'error', 'critical')
    )
    pairs = zip(
        (
            logger.debug,
            logger.info,
            logger.warning,
            logger.error,
            logger.critical
        ),
        messages,
        strict=True,
    )
    for f, m in pairs:
        f(m)

    # Only the levels >= the set log level should print anything to stdout
    should_log = [i >= level for i in range(logging.DEBUG, logging.CRITICAL + 1, 10)]

    captured = capsys.readouterr()
    expected_output = ''.join([messages[i] + '\n' for i in range(5) if should_log[i]])
    assert_equal(captured.err, expected_output)

    # Make sure everything was written to the log file (since it has level debug)
    msg_lines = msg.splitlines(keepends=True)
    msg_size = max(1, len(msg_lines))
    if msg and msg[-1] == '\n':
        msg_size += 1
    log_lines = log_file.read_text().splitlines(keepends=True)

    assert_equal(len(log_lines), 5 * msg_size)
    i = 0
    while i < len(log_lines):
        record = ''.join(log_lines[i:i+msg_size])
        assert re.match(LOG_REGEX, record)
        i += msg_size


@pytest.mark.parametrize(
        'x, y',
        [
            (0, 1),
            (-5, 0.0134753),
            (-3.25, 0.077432),
            (-1.5, 0.425096),
            (0.25, 0.969544),
            (2, 0.265802),
            (3.75, 0.0470095),
            (
                np.linspace(-1, 1, 11),
                [0.648054, 0.7477, 0.843551, 0.925007, 0.980328, 1., 0.980328, \
                 0.925007, 0.843551, 0.7477, 0.648054],
            ),
        ]
)
@pytest.mark.smoke()
def test_sech(x: float, y: float):
    assert_close(sech(x), y)


@pytest.mark.parametrize(
        'fname, exception', [
            ('not_a_path_obj.yaml', False),
            ('bad_non_path_obj.html', True),
            (Path('file.yml'), False),
            (Path('file.yaml'), False),
            (Path('bad_extension.json'), True),
            (Path('bad_extension.yl'), True),
        ]
)
@pytest.mark.smoke()
@pytest.mark.usefixtures('_mock_example_config')
def test_yaml_loader(
    fname: str | Path,
    exception: bool,
):
    if exception:
        with pytest.raises(
            NotImplementedError,
            match=r'.* file loading not yet supported.',
        ):
            cfg = read_config_file(fname)
    else:
        cfg = read_config_file(fname)
        assert_equal(cfg['do_rejection'], False)

