
from copy import copy

import numpy as np
import numpy.typing as npt
import pytest

from testing.utils import assert_close, assert_equal
from vipdopt.optimization import FoM


def fom_func(n) -> npt.ArrayLike:
    return np.square(n)


def gradient_func(n) -> npt.ArrayLike:
    return 2 * n


INPUT_ARRAY = np.reshape(range(9), (3,3))
SQUARED_ARRAY = np.array([[0, 1, 4], [9, 16, 25], [36, 49, 64]])
GRADIENT_ARRAY = np.array([[0, 2, 4], [6, 8, 10], [12, 14, 16]])
BASE_FOM = FoM([], [], fom_func, gradient_func, '', [])
ONES_ARRAY = np.ones((3, 3))
FOM_ZERO = FoM.zero(BASE_FOM)


@pytest.mark.smoke()
def test_compute_fom():
    output = BASE_FOM.compute_fom(INPUT_ARRAY)
    assert_equal(output, SQUARED_ARRAY)

    output = BASE_FOM.compute_gradient(INPUT_ARRAY)
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
            (1 + BASE_FOM, 1 + SQUARED_ARRAY, 1 + GRADIENT_ARRAY),
            (1 - BASE_FOM, 1 - SQUARED_ARRAY, 1 - GRADIENT_ARRAY),
            (BASE_FOM - 1, SQUARED_ARRAY - 1, GRADIENT_ARRAY - 1),
            (BASE_FOM / 2, SQUARED_ARRAY / 2, GRADIENT_ARRAY / 2),
            (0.5 * BASE_FOM, SQUARED_ARRAY / 2, GRADIENT_ARRAY / 2),
            # Adding one to prevent divide by zero
            ((BASE_FOM + 1) / (BASE_FOM + 1), ONES_ARRAY, ONES_ARRAY),
            (1 / (BASE_FOM + 1), 1 / (SQUARED_ARRAY + 1), 1 / (GRADIENT_ARRAY + 1)),
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
    act_fom = fom.compute_fom(INPUT_ARRAY)
    assert_equal(act_fom, exp_fom)

    act_grad = fom.compute_gradient(INPUT_ARRAY)
    assert_equal(act_grad, exp_grad)


@pytest.mark.smoke()
def test_sum():
    n = 10
    combined_fom = sum([BASE_FOM] * n, FOM_ZERO)
    output = combined_fom.compute_fom(INPUT_ARRAY)
    assert_equal(output, 10 * SQUARED_ARRAY)
    output = combined_fom.compute_gradient(INPUT_ARRAY)
    assert_equal(output, 10 * GRADIENT_ARRAY)


@pytest.mark.smoke()
@pytest.mark.xfail()
def test_weights():
    n = 10
    weights = np.linspace(0.0, 1.0, num=n)
    foms = [BASE_FOM] * n
    combined_fom = sum(np.multiply(weights, foms), FOM_ZERO)

    assert isinstance(combined_fom, FoM)
    assert not isinstance(combined_fom, int)

    output = combined_fom.compute_fom(INPUT_ARRAY)
    assert_close(output, sum(weights) * SQUARED_ARRAY)
    output = combined_fom.compute_gradient(INPUT_ARRAY)
    assert_close(output, sum(weights) * GRADIENT_ARRAY)


@pytest.mark.smoke()
def test_copy():
    org_vars = vars(BASE_FOM)
    cpy = copy(BASE_FOM)

    assert_equal(vars(cpy), org_vars)

    # Modifying the copy should not affect the original
    cpy = cpy + 1
    assert_equal(vars(BASE_FOM), org_vars)
    assert_equal(cpy.compute_fom(INPUT_ARRAY), SQUARED_ARRAY + 1)


@pytest.mark.smoke()
def test_iadd():
    fom = copy(BASE_FOM)
    fom += 1
    output = fom.compute_fom(INPUT_ARRAY)
    assert_equal(output, SQUARED_ARRAY + 1)

    output = fom.compute_gradient(INPUT_ARRAY)
    assert_equal(output, GRADIENT_ARRAY + 1)


@pytest.mark.smoke()
def test_isub():
    fom = copy(BASE_FOM)
    fom -= 1
    output = fom.compute_fom(INPUT_ARRAY)
    assert_equal(output, SQUARED_ARRAY - 1)

    output = fom.compute_gradient(INPUT_ARRAY)
    assert_equal(output, GRADIENT_ARRAY - 1)


@pytest.mark.smoke()
def test_imul():
    fom = copy(BASE_FOM)
    fom *= 2
    output = fom.compute_fom(INPUT_ARRAY)
    assert_equal(output, SQUARED_ARRAY * 2)

    output = fom.compute_gradient(INPUT_ARRAY)
    assert_equal(output, GRADIENT_ARRAY * 2)


@pytest.mark.smoke()
def test_itruediv():
    fom = copy(BASE_FOM)
    fom /= 2
    output = fom.compute_fom(INPUT_ARRAY)
    assert_equal(output, SQUARED_ARRAY / 2)

    output = fom.compute_gradient(INPUT_ARRAY)
    assert_equal(output, GRADIENT_ARRAY / 2)


@pytest.mark.smoke()
def test_bad_op():
    with pytest.raises(TypeError, match=r'unsupported operand type\(s\) for \*\*'):
        BASE_FOM ** 2

    with pytest.raises(TypeError, match=r'unsupported operand type\(s\) for //'):
        BASE_FOM // 2

    with pytest.raises(TypeError, match=r'unsupported operand type\(s\) for %'):
        BASE_FOM % 2

    fom1 = copy(BASE_FOM)
    fom2 = copy(BASE_FOM)

    fom2.polarization = 'Hello World'
    with pytest.raises(
        TypeError,
        match=r'unsupported operand type\(s\) for \+: \'FoM\' and \'FoM\' .* equal*',
    ):
        fom1 + fom2

    fom2 = copy(BASE_FOM)
    fom2.opt_ids = list(range(9))
    with pytest.raises(
        TypeError,
        match=r'unsupported operand type\(s\) for \+: \'FoM\' and \'FoM\' .* equal*',
    ):
        fom1 + fom2

    fom2 = copy(BASE_FOM)
    fom2.freq = list(range(9))
    with pytest.raises(
        TypeError,
        match=r'unsupported operand type\(s\) for \+: \'FoM\' and \'FoM\' .* equal*',
    ):
        fom1 + fom2

def test_dict():
    fom = copy(BASE_FOM)
    fdict = fom.as_dict()

    fom2 = FoM.from_dict(f'{fom.name}', fdict, {})

    assert_equal(fom2, fom)
