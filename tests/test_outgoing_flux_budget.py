import numpy as np
from climlab_stardust_extension.dynamics.outgoing_flux_budget import limit_flux

def test_randomized_budget_conservation_and_positivity():
    rng=np.random.default_rng(20261111)
    # Nonuniform grids, both directions, empty cells, large simultaneous outflows.
    for trial in range(500):
     q=10.**rng.uniform(-40,2,(8,30));q[rng.random(q.shape)<.2]=0
     dx=10.**rng.uniform(-2,3,30);dt=10.**rng.uniform(-2,3)
     f=rng.normal(size=(8,31))*10.**rng.uniform(-35,4,(8,31));f[:,0]=f[:,-1]=0
     g,active=limit_flux(f,q,dx,dt);after=q+dt*(g[:,:-1]-g[:,1:])/dx
     assert np.min(after)>=-1e-12*max(q.max(),1e-30)
     np.testing.assert_allclose((after*dx).sum(1),(q*dx).sum(1),rtol=2e-14,atol=1e-35)
     if not active.any():np.testing.assert_array_equal(g,f)
    q=np.ones((2,5));dx=np.ones(5);f=np.zeros((2,6));f[:,2]=.1
    np.testing.assert_array_equal(limit_flux(f,q,dx,1)[0],f)
    # One cell drained simultaneously through its two faces.
    q=np.array([[0.,1.,0.]]);f=np.array([[0.,-.8,.8,0.]])
    g,_=limit_flux(f,q,np.ones(3),1);a=q+g[:,:-1]-g[:,1:]
    assert a.min()>=0;np.testing.assert_allclose(a.sum(),1,atol=2e-15)
