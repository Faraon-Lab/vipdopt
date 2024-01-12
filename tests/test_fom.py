
import pytest

import numpy as np
import numpy.typing as npt

from vipdopt.optimization import FoM, FOM_ZERO

from testing.utils import assert_close, assert_equal


def fom_func(n) -> npt.ArrayLike:
    return np.square(n)


def gradient_func(n) -> npt.ArrayLike:
    return 2 * n


INPUT_ARRAY = np.reshape(range(0, 9), (3,3))
SQUARED_ARRAY = np.array([[0, 1, 4], [9, 16, 25], [36, 49, 64]])
GRADIENT_ARRAY = np.array([[0, 2, 4], [6, 8, 10], [12, 14, 16]])
BASE_FOM = FoM([], [], fom_func, gradient_func, '', [])
DIV_ARRAY = np.ones((3, 3))


@pytest.mark.smoke()
def test_compute_fom():
    output = BASE_FOM.compute(INPUT_ARRAY)
    assert_equal(output, SQUARED_ARRAY)

    output = BASE_FOM.gradient(INPUT_ARRAY)
    assert_equal(output, GRADIENT_ARRAY)


@pytest.mark.smoke()
@pytest.mark.parametrize(
        'fom, exp_fom, exp_grad',
        [
            (BASE_FOM + BASE_FOM, 2 * SQUARED_ARRAY, 2 * GRADIENT_ARRAY),
            (BASE_FOM - BASE_FOM, 0 * SQUARED_ARRAY, 0 * GRADIENT_ARRAY),
            (BASE_FOM * BASE_FOM, SQUARED_ARRAY ** 2, GRADIENT_ARRAY ** 2),
            (2 * BASE_FOM, 2 * SQUARED_ARRAY, 2 * GRADIENT_ARRAY),
            (BASE_FOM * 2, 2 * SQUARED_ARRAY, 2 * GRADIENT_ARRAY),
            (1 - BASE_FOM, 1 - SQUARED_ARRAY, 1 - GRADIENT_ARRAY),
            (BASE_FOM - 1, SQUARED_ARRAY - 1, GRADIENT_ARRAY - 1),
            (BASE_FOM / 2, SQUARED_ARRAY / 2, GRADIENT_ARRAY / 2),
            (0.5 * BASE_FOM, SQUARED_ARRAY / 2, GRADIENT_ARRAY / 2),
            # Adding one to prevent divide by zero
            ((BASE_FOM + 1) / (BASE_FOM + 1), DIV_ARRAY, DIV_ARRAY),
            # Distributive property / linearity
            (BASE_FOM * (2 + 3), 5 * SQUARED_ARRAY, 5 * GRADIENT_ARRAY),
            ((2 + 3) * BASE_FOM, 5 * SQUARED_ARRAY, 5 * GRADIENT_ARRAY),
            (2 * (BASE_FOM + BASE_FOM), 4 * SQUARED_ARRAY, 4 * GRADIENT_ARRAY),
            ((BASE_FOM + BASE_FOM) * 2, 4 * SQUARED_ARRAY, 4 * GRADIENT_ARRAY),
            (
                2 * (BASE_FOM + 2 * BASE_FOM) + 3 * (BASE_FOM * BASE_FOM),
                6 * SQUARED_ARRAY + 3 * SQUARED_ARRAY ** 2,
                6 * GRADIENT_ARRAY + 3 * GRADIENT_ARRAY ** 2,
            ),
        ]
)
def test_arithmetic(fom: FoM, exp_fom: npt.ArrayLike, exp_grad: npt.ArrayLike):
    act_fom = fom.compute(INPUT_ARRAY)
    assert_equal(act_fom, exp_fom)

    act_grad = fom.gradient(INPUT_ARRAY)
    assert_equal(act_grad, exp_grad)


@pytest.mark.smoke()
def test_sum():
    n = 10
    combined_fom = sum([BASE_FOM] * n)
    output = combined_fom.compute(INPUT_ARRAY)
    assert_equal(output, 10 * SQUARED_ARRAY)
    output = combined_fom.gradient(INPUT_ARRAY)
    assert_equal(output, 10 * GRADIENT_ARRAY)


@pytest.mark.smoke()
def test_weights():
    n = 10
    weights = np.linspace(0.0, 1.0, num=n)
    foms = [BASE_FOM] * n
    combined_fom = sum(np.multiply(weights, foms), FOM_ZERO)

    assert isinstance(combined_fom, FoM)
    assert not isinstance(combined_fom, int)

    output = combined_fom.compute(INPUT_ARRAY)
    assert_close(output, sum(weights) * SQUARED_ARRAY)
    output = combined_fom.gradient(INPUT_ARRAY)
    assert_close(output, sum(weights) * GRADIENT_ARRAY)
