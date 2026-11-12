"""Conservative outgoing-flux budget for nonnegative tracer transport."""
import numpy as np

def limit_flux(flux, q, dx, dt):
    """Flux is positive toward increasing coordinate, including pressure.

    Close outer faces; cap combined outgoing transfer by donor inventory.
    Each adjusted face is shared by both cells. Existing non-negative fluxes
    within budget are unchanged. Roundoff reserve only applies to capped donors.
    No concentration clipping is performed.
    """
    f=flux.copy(); f[...,0]=f[...,-1]=0.
    available=np.maximum(q,0)*dx/dt
    outgoing=np.maximum(f[...,1:],0)+np.maximum(-f[...,:-1],0)
    active=outgoing>available
    if not np.any(active):return f,active
    alpha=np.ones_like(q)
    np.divide(available*(1-32*np.finfo(float).eps),outgoing,out=alpha,where=active)
    left=np.concatenate([np.ones_like(alpha[...,:1]),alpha],axis=-1)
    right=np.concatenate([alpha,np.ones_like(alpha[...,:1])],axis=-1)
    f*=np.where(f>0,left,right)
    return f,active
