"""Dark-column boundary condition and native-call/seed preservation."""
import numpy as np
import pytest
from climlab_stardust_extension.radiation.rrtm import rrtmg_sw as sw

def arguments(coszen, col_by_col=1):
    n = len(coszen)
    args = [0] * 53
    for k in list(range(8, 18)) + list(range(22, 32)) + [33, 34] + list(range(41, 45)) + list(range(49, 53)):
        args[k] = np.arange(n, dtype=float)[:, None]
    for k in range(18, 22):
        args[k] = np.zeros((14, n, 2))
    args[2:6] = [2, col_by_col, 1, 1]
    args[33] = np.asarray(coszen)
    return args

def outputs(n, value=1.):
    shapes = [(n,3),(n,3),(n,2),(n,3),(n,3),(n,2)] + [(n,3,14)]*4
    return tuple(np.full(s,value) for s in shapes)

def test_all_dark_never_calls_native(monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail('Native SW must not be called for all-dark columns')
    monkeypatch.setattr(sw._rrtmg_sw,'climlab_rrtmg_sw_ensemble',forbidden)
    monkeypatch.setattr(sw._rrtmg_sw,'climlab_rrtmg_sw_single_col',forbidden)
    result=sw._shortwave_without_dark_columns(2,2,*arguments([0.,-0.1]))
    assert all(np.all(x==0) for x in result)

def test_positive_illumination_has_no_threshold(monkeypatch):
    a=arguments([np.nextafter(0.,1.),0.5]);expected=outputs(2)
    def native(n,l,*received):
        assert n==2 and received[33] is a[33]
        return expected
    monkeypatch.setattr(sw._rrtmg_sw,'climlab_rrtmg_sw_ensemble',native)
    assert sw._shortwave_without_dark_columns(2,2,*a) is expected

def test_column_independent_seeds_compact_only_light(monkeypatch):
    a=arguments([0.,0.4,0.,0.7])
    def native(n,l,*received):
        assert n==2
        np.testing.assert_array_equal(received[8][:,0],[1.,3.])
        assert received[18].shape==(14,2,2)
        return outputs(n)
    monkeypatch.setattr(sw._rrtmg_sw,'climlab_rrtmg_sw_ensemble',native)
    result=sw._shortwave_without_dark_columns(4,2,*a)
    for x in result:
        assert np.all(x[[0,2]]==0) and np.all(x[[1,3]]==1)

def test_column_dependent_seeds_keep_original_indices(monkeypatch):
    calls=[];a=arguments([0.,0.4,0.,0.7],col_by_col=0)
    def native(col,member,*received,ncol,nlay):
        assert ncol==4 and nlay==2
        np.testing.assert_array_equal(received[8],a[8])
        calls.append((col,member));return outputs(1,float(member))
    monkeypatch.setattr(sw._rrtmg_sw,'climlab_rrtmg_sw_single_col',native)
    result=sw._shortwave_without_dark_columns(4,2,*a)
    assert calls==[(2,1),(2,2),(4,1),(4,2)]
    for x in result:
        assert np.all(x[[0,2]]==0) and np.all(x[[1,3]]==1.5)

def test_nonfinite_illumination_rejected():
    with pytest.raises(ValueError,match='Nonfinite'):
        sw._shortwave_without_dark_columns(2,2,*arguments([0.,np.nan]))
