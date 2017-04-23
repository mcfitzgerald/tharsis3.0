import numpy as np

def dilser(low=0.001, limit=100.0, dilfactor = 2.0):
    """Returns a list containing a dilution series that ranges from
    "low" to "limit" by "dilfactor".
    """
    a = [low]
    while a[-1] <= limit:
        a.append(a[len(a)-1]*dilfactor)
    
    return np.array(a)

def klotz1(k,free_ligand):
    bound_fraction = (k*free_ligand)/(1 + k*free_ligand)
    return bound_fraction

def klotz1_obj(k,lig,dat,eps=None):
    bfrac = (k*lig)/(1 + k*lig)
    if eps is None:
        if lig.ndim > 1:
            return np.concatenate((bfrac - dat))
        else:
            return (bfrac - dat)
    else:
        if lig.ndim > 1:
            return np.concatenate((bfrac - dat)/eps)
        else:
            return (bfrac - dat)/eps
    
def noiser(dat,sd):
    return np.random.normal(1,sd,len(dat))*dat