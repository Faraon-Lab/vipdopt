import sys
from copy import copy

import numpy as np
import numpy.typing as npt
import pytest
from functools import partial

from testing.utils import assert_close, assert_equal
from vipdopt.configuration import Config
from vipdopt.optimization import FoM
from vipdopt.optimization.fom import (TestFoM, 
        _test_fom_func, _test_gradient_func, 
        _test_unit_func, _test_unit_gradient_func, 
        unique_fwd_sim_map
)
from vipdopt.simulation import Source
from vipdopt.utils import flatten

# def _test_fom_func(n) -> npt.ArrayLike:
#     return np.sum(np.square(n))
# def _test_gradient_func(n) -> npt.ArrayLike:
#     return 2 * n

INPUT_ARRAY = np.reshape(range(9), (3, 3))  # X
SQUARED_ARRAY = np.array([[0, 1, 4], [9, 16, 25], [36, 49, 64]])  # X^2
SQUARE_SUM = SQUARED_ARRAY.sum()
GRADIENT_ARRAY = np.array([[0, 2, 4], [6, 8, 10], [12, 14, 16]])  # 2X
ONES_ARRAY = np.ones((3, 3))

# BASE_FOM = FoM('TE', [], [], [], [], fom_func, gradient_func, [0, 1, 2], [], [0, 1, 2])
BASE_FOM = FoM(_test_fom_func, _test_gradient_func, [], pos_max_freqs=[0,1,2], all_freqs=[0,1,2])
UNIT_FOM = FoM(_test_unit_func, _test_unit_gradient_func, [], pos_max_freqs=[0,1,2], all_freqs=[0,1,2])

@pytest.mark.smoke()
def test_compute_fom():
    output = BASE_FOM.compute_fom(n=INPUT_ARRAY)
    assert_equal(output, SQUARE_SUM)

    output = BASE_FOM.compute_grad(n=INPUT_ARRAY)
    assert_equal(output, GRADIENT_ARRAY)

# project.fom = FoM(None, None, [(FoM_1,), (FoM_2,)], (1.0,1.0))
@pytest.mark.smoke()
@pytest.mark.parametrize(
    'fom, exp_fom, exp_grad',
    [
        # BASE_FOM + BASE_FOM
        (FoM(None, None, [(BASE_FOM,), (BASE_FOM,)]), 2*SQUARE_SUM, 2*GRADIENT_ARRAY),
        # BASE_FOM - BASE_FOM
        (FoM(None, None, [(BASE_FOM,), (BASE_FOM,)], (1, -1)), 0*SQUARE_SUM, 0*GRADIENT_ARRAY),
        # 2 * BASE_FOM
        (FoM(None, None, [(BASE_FOM,)], (2,)), 2*SQUARE_SUM, 2*GRADIENT_ARRAY),
        (FoM(None, None, [(BASE_FOM, UNIT_FOM)], (2,)), 2*SQUARE_SUM, 2*GRADIENT_ARRAY),
        # 0.5 * BASE_FOM
        (FoM(None, None, [(BASE_FOM,)], (0.5,)), SQUARE_SUM/2, GRADIENT_ARRAY/2),
        (FoM(None, None, [(BASE_FOM, UNIT_FOM)], (1/2,)), 0.5*SQUARE_SUM, 0.5*GRADIENT_ARRAY),
        # 1 / 2 * BASE_FOM
        (FoM(None, None, [(BASE_FOM,)], (1/2,)), SQUARE_SUM/2, GRADIENT_ARRAY/2),
        # Distributive property / linearity
        (FoM(None, None, [(BASE_FOM,)], ((2+3),)), 5*SQUARE_SUM, 5*GRADIENT_ARRAY),
        (FoM(None, None, [(UNIT_FOM, BASE_FOM)], ((2+3),)), 5*SQUARE_SUM, 5*GRADIENT_ARRAY),
        (FoM(None, None, [(BASE_FOM,), (BASE_FOM,)], (2, 2)), 4*SQUARE_SUM, 4*GRADIENT_ARRAY),
        # 1 + BASE_FOM
        (FoM(None, None, [(BASE_FOM,), (UNIT_FOM,)]), 1 + SQUARE_SUM, GRADIENT_ARRAY),
        (FoM(None, None, [(UNIT_FOM,), (BASE_FOM,)]), SQUARE_SUM + 1, GRADIENT_ARRAY),
        # POWERS OF BASE_FOM
        (
            FoM(None, None, [(BASE_FOM, BASE_FOM)], (3,)),
            3 * SQUARE_SUM**2,
            6 * SQUARE_SUM * GRADIENT_ARRAY,
        ),


    ],
)
def test_arithmetic(fom: FoM, exp_fom: npt.ArrayLike, exp_grad: npt.ArrayLike):
    act_fom = fom.compute_fom(n=INPUT_ARRAY)
    assert_equal(act_fom, exp_fom)

    act_grad = fom.compute_grad(n=INPUT_ARRAY)
    assert_equal(act_grad, exp_grad)

# # Deprecated ==========================
# @pytest.mark.smoke()
# def test_sum():
#     n = 10
#     combined_fom = sum([BASE_FOM] * (n - 1), BASE_FOM)
#     output = combined_fom.compute_fom(INPUT_ARRAY)
#     assert_equal(output, 10 * SQUARE_SUM)
#     output = combined_fom.compute_grad(INPUT_ARRAY)
#     assert_equal(output, 10 * GRADIENT_ARRAY)

@pytest.mark.smoke()
def test_weights():
    n = 10
    weights = np.linspace(0.0, 1.0, num=n)
    foms = [(BASE_FOM,)] * n
    combined_fom = FoM(None, None, foms, tuple(weights))

    output = combined_fom.compute_fom(n=INPUT_ARRAY)
    assert_close(output, sum(weights) * SQUARE_SUM)
    output = combined_fom.compute_grad(n=INPUT_ARRAY)
    assert_close(output, sum(weights) * GRADIENT_ARRAY)

@pytest.mark.smoke()
def test_copy():
    org_vars = vars(BASE_FOM)
    cpy = copy(BASE_FOM)

    assert_equal(cpy, BASE_FOM)

    # # Modifying the copy should not affect the original
    # cpy = cpy + BASE_FOM
    cpy = FoM(None,None, [(cpy,),(BASE_FOM,)])
    assert_equal(vars(BASE_FOM), org_vars)
    assert_equal(cpy.compute_fom(n=INPUT_ARRAY), 2 * SQUARE_SUM)


# # DEPRECATED ============================================
# @pytest.mark.smoke()
# def test_iadd():
#     fom = copy(BASE_FOM)
#     fom += BASE_FOM
#     output = fom.compute_fom(INPUT_ARRAY)
#     assert_equal(output, 2 * SQUARE_SUM)

#     output = fom.compute_grad(INPUT_ARRAY)
#     assert_equal(output, 2 * GRADIENT_ARRAY)


# @pytest.mark.smoke()
# def test_isub():
#     fom = copy(BASE_FOM)
#     fom -= BASE_FOM
#     output = fom.compute_fom(INPUT_ARRAY)
#     assert_equal(output, 0 * SQUARE_SUM)

#     output = fom.compute_grad(INPUT_ARRAY)
#     assert_equal(output, 0 * GRADIENT_ARRAY)


# @pytest.mark.smoke()
# def test_imul():
#     fom = copy(BASE_FOM)
#     fom *= 2
#     output = fom.compute_fom(INPUT_ARRAY)
#     assert_equal(output, SQUARE_SUM * 2)

#     output = fom.compute_grad(INPUT_ARRAY)
#     assert_equal(output, GRADIENT_ARRAY * 2)


# @pytest.mark.smoke()
# def test_itruediv():
#     fom = copy(BASE_FOM)
#     fom /= 2
#     output = fom.compute_fom(INPUT_ARRAY)
#     assert_equal(output, SQUARE_SUM / 2)

#     output = fom.compute_grad(INPUT_ARRAY)
#     assert_equal(output, GRADIENT_ARRAY / 2)


# @pytest.mark.smoke()
# def test_repeated_mult():
#     fom1 = BASE_FOM * BASE_FOM

#     combined_fom = fom1 * BASE_FOM
#     output = combined_fom.compute_fom(INPUT_ARRAY)
#     assert_close(output, SQUARE_SUM**3)
#     output = combined_fom.compute_grad(INPUT_ARRAY)
#     assert_close(output, 3 * (SQUARED_ARRAY**2 * GRADIENT_ARRAY))

# @pytest.mark.smoke()
# def test_bad_op():
#     with pytest.raises(TypeError, match=r'unsupported operand type\(s\) for \*\*'):
#         BASE_FOM**2

#     with pytest.raises(TypeError, match=r'unsupported operand type\(s\) for //'):
#         BASE_FOM // 2

#     with pytest.raises(TypeError, match=r'unsupported operand type\(s\) for %'):
#         BASE_FOM % 2

#     with pytest.raises(TypeError, match=r'unsupported operand type\(s\) for /'):
#         BASE_FOM / BASE_FOM

@pytest.mark.smoke()
def test_dict():
    fom = copy(BASE_FOM)
    fdict = fom.as_dict()

    fom2 = FoM.from_dict(fdict)

    assert_equal(fom2, fom)


@pytest.mark.smoke()
def test_unique_source_map():
    srcs = [Source(f'src{i}', 'gaussian') for i in range(10)]
    # 0:1, 1:3, 2:5, 3:7, 4:9
    combos = [srcs[i:j] for (i, j) in zip(range(5), range(1, 10, 2), strict=False)]
    foms = [(
        FoM(_test_fom_func, _test_gradient_func,[],
            fwd_srcs = src_combo,
            pos_max_freqs=[0,1,2],
            all_freqs=[0,1,2]),
    )
    for src_combo in combos]

    fom = FoM(None, None, foms, weights=np.ones(10))

    sim_map = unique_fwd_sim_map(flatten(fom.foms))
    assert_equal(len(sim_map), 5)

    src_map_names = [{src.name for src in srcs} for srcs in sim_map]
    expected_names = [
        {'src0'},
        {'src1', 'src2'},
        {'src2', 'src3', 'src4'},
        {'src3', 'src4', 'src5', 'src6'},
        {'src4', 'src5', 'src6', 'src7', 'src8'},
    ]
    for name_set in expected_names:
        assert name_set in src_map_names

    for i, combo in enumerate(combos):
        assert_equal(sim_map[frozenset(combo)], [foms[i][0]])


@pytest.mark.smoke()
def test_save_load(tmpdir):
    fom1 = BASE_FOM
    fom1_vars = copy(vars(BASE_FOM))
    fom1_vars.pop('fom_func')
    fom1_vars.pop('grad_func')
    fom2 = TestFoM(**fom1_vars)
    p = tmpdir / 'fom.yml'
    fdict_1 = fom1.as_dict()
    fom2.save(p)

    
    fom4 = FoM.from_dict(fdict_1)
    fom3 = TestFoM._load_from_config(Config.from_file(p))

    assert_equal(fdict_1, fom4.as_dict())
    # NOTE: The functions are not the same because FoM base class does not have default fom_func or grad_func
    # for attr in vars(fom1):
    #     assert_close(fom4.__getattribute__(attr), fom1.__getattribute__(attr))
    for attr in vars(fom2):
        assert_close(fom3.__getattribute__(attr), fom2.__getattribute__(attr))


@pytest.mark.smoke()
def test_nested_save_load(tmpdir):
    fom1 = FoM(None, None, 
            [(BASE_FOM,UNIT_FOM), (UNIT_FOM,BASE_FOM), (BASE_FOM,BASE_FOM)], 
            (2, 3, 4.5)
        )
    p = tmpdir / 'fom.yml'
    fdict_1 = fom1.as_dict()
    fom1.save(p)
    
    fom1_vars = copy(vars(fom1))
    fom2 = FoM(**fom1_vars)
    
    assert_equal(fdict_1, fom2.as_dict())
    for attr in vars(fom1):
        assert_close(fom2.__getattribute__(attr), fom1.__getattribute__(attr))
    

    fom4 = FoM.from_dict(fdict_1)
    assert_equal(fdict_1, fom4.as_dict())
    for attr in vars(fom1):
        print(attr)
        assert_close(fom4.__getattribute__(attr), fom1.__getattribute__(attr))
    
    fom3 = FoM._load_from_config(Config.from_file(p))
    # NOTE: The functions are not the same because FoM base class does not have default fom_func or grad_func
    # for attr in vars(fom1):
    #     assert_close(fom4.__getattribute__(attr), fom1.__getattribute__(attr))
    for attr in vars(fom2):
        print(attr)
        assert_close(fom3.__getattribute__(attr), fom2.__getattribute__(attr))
