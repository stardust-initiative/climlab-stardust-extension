"""Native event and evolution number budgets for fixed-mass sectional bins."""
import numpy as np
import pytest
from climlab.domain.domain import zonal_mean_column
from climlab.domain.field import Field
from climlab_stardust_extension.microphysics import Coagulation


def process(cores, df=1.6):
    _, atm = zonal_mean_column(num_lat=2, num_lev=2)
    state = {f'bin_{i}': Field(np.full((2, 2), 1e-9), domain=atm)
             for i in range(len(cores))}
    return Coagulation(state=state, cores=dict(zip(state, cores)),
                       d0=500e-9, rho_p=2200., Df=df, kf=1.,
                       temperature=np.full((2, 2), 220.), timestep=1.)


@pytest.mark.parametrize('cores', [[1, 2, 4], [1, 8, 64],
    [1, 1.9999, 4], [1, 2.0001, 4], [1],
    (2**np.arange(7)).tolist(), (2**np.arange(8)).tolist()])
@pytest.mark.parametrize('df', [1.6, 2.5, 3.])
def test_each_native_collision(cores, df):
    p = process(cores, df)
    names = list(p.state)
    masses = p.masses_vec[:, None, None]
    kernel = p.Kcoag.copy()
    for i in range(len(cores)):
        for j in range(i, len(cores)):
            p.Kcoag[:] = 0.
            p.Kcoag[i, j] = kernel[i, j]
            p.Kcoag[j, i] = kernel[j, i]
            rate = (0.5 if i == j else 1.) * kernel[i, j] * 1e-18 / (masses[i]*masses[j])
            tendency = p._compute()
            net = np.array([tendency[n] for n in names]) / rate
            products = net.copy()
            products[i] += masses[i]
            products[j] += masses[j]
            product_mass = masses[i] + masses[j]
            np.testing.assert_allclose(net.sum(axis=0)/product_mass, 0., atol=3e-15)
            np.testing.assert_allclose(products.sum(axis=0)/product_mass, 1., rtol=3e-15)
            # Overflow explicitly retains the old last-bin mass closure.
            expected = max(1., float(product_mass[0, 0]/masses[-1, 0, 0]))
            np.testing.assert_allclose((products/masses).sum(axis=0), expected, rtol=3e-14)
            np.testing.assert_allclose((net/masses).sum(axis=0), expected-2., atol=3e-14)
            if cores == [1, 2, 4] and (i, j) == (0, 1):
                np.testing.assert_allclose(net[:, 0, 0]/masses[0, 0, 0], [-1., -1., 2.], atol=3e-15)


def test_native_evolution_number_budget():
    p = process((2**np.arange(8)).tolist())
    names = list(p.state)
    m = p.masses_vec[:, None, None]
    initial_mass = sum(p.state.values()).copy()
    for _ in range(30):
        before = np.array([p.state[n].copy() for n in names])
        dq = p._compute()
        dq = np.array([dq[n] for n in names])
        expected = np.zeros((2, 2))
        for i in range(len(names)):
            for j in range(i, len(names)):
                rate = (0.5 if i == j else 1.)*p.Kcoag[i, j]*before[i]*before[j]/(m[i]*m[j])
                products = max(1., float((m[i, 0, 0]+m[j, 0, 0])/m[-1, 0, 0]))
                expected += rate*(products-2.)
        dt = .001/np.max(np.maximum(-dq, 0.)/before)
        p.set_timestep(timestep=dt)
        p.step_forward()
        after = np.array([p.state[n] for n in names])
        assert np.isfinite(after).all() and np.all(after >= 0.)
        np.testing.assert_allclose(after.sum(axis=0), initial_mass, rtol=2e-14)
        np.testing.assert_allclose(((after-before)/m).sum(axis=0)/dt, expected, rtol=2e-10)
