
__all__ = ["dilser"]

import numpy as np
from scipy import optimize
from . import models

def dilser(low=0.001, limit=100., dilfactor=2.):
    '''returns a numpy array dilution series from low to limit'''
    #replace this with a generator/iterator someday
    a = [low]
    while a[-1] <= limit:
        a.append(a[len(a)-1]*dilfactor)
    return np.array(a)


def objfunc(parm,model,data,*modelargs):
    return (model(parm,*modelargs) - np.concatenate(data))


def objfunc_wt(parm,model,data,err,*modelargs):
    return ((model(parm,*modelargs) - np.concatenate(data))/np.concatenate(err))


def fitter(self,err=None,**kwargs):
    '''Wraps scipy optimize routine and uses data/parameters from model object.
    If weights are included, a weighted fit is performed'''
    if err is None:
        return [optimize.least_squares(objfunc,self.guess,bounds=self.bounds, \
                args=(self.model.modfunc,self.model.meanset[i],self.model.ligs,self.model.rtot),**kwargs) \
                for i in range(self.model.meanset.shape[0])]
    else:
        return [optimize.least_squares(objfunc_wt,self.guess,bounds=self.bounds, \
                args=(self.model.modfunc,self.model.meanset[i],err[i],self.model.ligs,self.model.rtot),**kwargs) \
                for i in range(self.model.meanset.shape[0])]


def fitwrap(self,weight=0,**kwargs):
    if weight == 1:
        return fitter(self,err=(self.model.stdset),**kwargs)
    elif weight == 2:
        return fitter(self,err=(self.model.meanset),**kwargs)
    else:
        return fitter(self,**kwargs)
