"""Physical covariance, live updates and support of opt-in diffusion tensors."""
from datetime import datetime
import numpy as np
import pytest
import xarray as xr
import climlab
from climlab_stardust_extension.dynamics import TwoDimensionalAdvectionDiffusion,AtmosphericData
from climlab_stardust_extension.dynamics.tensor_diffusion import bound_staggered_cross

@pytest.mark.parametrize('convention,sign',[('legacy',-1),('physical',1)])
@pytest.mark.parametrize('cross',[-50.,0.,50.])
def test_covariance_measures_physical_cross(convention,sign,cross):
    lat=np.arange(-89.5,90,1.);lev=np.arange(50.,151.)
    q=climlab.column_state(lat=lat,lev=lev)['Tatm']
    y=climlab.constants.a*np.sin(np.deg2rad(q.domain.axes['lat'].points));p=q.domain.axes['lev'].points*100
    q[:]=np.exp(-.5*(y[:,None]/333000)**2-.5*((p[None,:]-10000)/500)**2)+1e-12
    c=np.full((len(lat)+1,len(lev)+1),cross)
    model=TwoDimensionalAdvectionDiffusion(state={'q':q},timestep=3600,
        U=np.zeros((len(lat)+1,len(lev))),W=np.zeros((len(lat),len(lev)+1)),
        W_sedimentation=np.zeros((len(lat),len(lev)+1)),
        Kyy=np.full((len(lat)+1,len(lev)),1e5),Kzz=np.full((len(lat),len(lev)+1),.1),
        Kyz=c,cross_diffusion_convention=convention)
    def covariance():
        w=q*model.M_air;w=w/w.sum();ym=(w*y[:,None]).sum();pm=(w*p[None,:]).sum()
        return float((w*(y[:,None]-ym)*(p[None,:]-pm)).sum())
    initial=covariance();mass=float((q*model.M_air).sum());model.step_forward()
    inferred=(covariance()-initial)/(2*3600)
    if cross:assert inferred==pytest.approx(sign*cross,rel=.15)
    else:assert abs(inferred)<1e-8
    assert float((q*model.M_air).sum())==pytest.approx(mass,rel=1e-12)
    assert q.min()>=-1e-14
    c[:]=cross+1
    np.testing.assert_allclose(model.Kyz,(-sign)*c*np.cos(model._phibounds[:,None]))


def test_cross_bounds_do_not_change_diagonals_or_depend_on_sign_cancellation():
    rng=np.random.default_rng(23);latb=np.linspace(-90,90,13);latc=(latb[:-1]+latb[1:])/2
    pb=np.geomspace(1,1000,11);pc=(pb[:-1]+pb[1:])/2
    yy=rng.uniform(0,3,(13,10));pp=rng.uniform(0,3,(12,11));cross=rng.uniform(-10,10,(13,11))
    yy0=yy.copy();pp0=pp.copy();original=cross.copy()
    result,stats=bound_staggered_cross(yy,pp,cross,latc,latb,pc,pb)
    np.testing.assert_array_equal(yy,yy0);np.testing.assert_array_equal(pp,pp0);np.testing.assert_array_equal(cross,original)
    assert all(stats[k]<=1+1e-12 for k in ['max_corner_ratio','max_y_face_ratio','max_p_face_ratio'])
    assert np.all(abs(result)<=abs(original))
    zero,_=bound_staggered_cross(yy*0,pp,cross,latc,latb,pc,pb)
    np.testing.assert_array_equal(zero,0)


def test_tensor_interpolation_requires_full_support_and_no_extrapolation():
    q=climlab.column_state(lat=np.arange(-85,90,10),lev=[5,20,50,100,200,500])['Tatm']
    coords=dict(month=np.arange(1,13),level=[10.,100.,200.],latitude=[-60.,0.,60.])
    valid=np.ones((12,3,3));valid[:,1,1]=0
    fields={}
    for name,grid in [('kyy','bounds*centers'),('kzz','centers*bounds'),('kyz','bounds*bounds')]:
        a=xr.DataArray(np.ones((12,3,3)),dims=['month','level','latitude'],coords=coords,attrs={'diffusion_tensor_component':name})
        a=a.assign_coords(tensor_valid=(('month','level','latitude'),valid))
        fields[name]=dict(data=a,grid_type=grid)
    atm=AtmosphericData(param_configs=fields,state={'q':q},t_0=datetime(2024,1,1),symmetric_diffusion=True)
    for field in atm._current_data.values():
        assert np.isfinite(field).all()
        outside=(abs(field.lat)>60)|(field.level<10)|(field.level>200)
        np.testing.assert_array_equal(field.where(outside,0),0)
    yy=atm._param_configs['kyy']['data']
    assert float(yy.sel(month=1).interp(lat=0,level=100))==0
