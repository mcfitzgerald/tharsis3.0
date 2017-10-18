# -*- coding: utf-8 -*-
__all__ = ["WymSim","WymSimSubset","WymSimHardlig","WymFit"]

from . import helpers
from . import models
import numpy as np

class WymSim:

    #default init values
    __dparms = np.array([3.,2.,0.1,100.])
    __drtot = np.array([0.001,0.0025,0.005,0.01,0.025,0.05])
    __dlrange = np.array([0.001,100.,2.])
    __drp = 3
    __dst = 3333
    __dns = 0.05
    __dmodfunc = models.wymfunc

    def __init__(self,modfunc=__dmodfunc,parms=__dparms,rtot=__drtot,ligrange=__dlrange,reps=__drp,sets=__dst,noise=__dns):
        '''Returns a Wyman model object'''
        self.sets = sets
        self.reps = reps
        self.modfunc = modfunc
        self.parms = parms
        self.rtot = rtot
        self.noise = noise
        self.ligs = np.array([helpers.dilser(ligrange[0],ligrange[1],ligrange[2]) for i in range(len(self.rtot))])
        self.bfrac = np.array([self.modfunc(self.parms,self.ligs[i],self.rtot[i]) for i in range(len(self.rtot))])
        self.noised = np.array([[np.random.normal(self.bfrac,(self.noise*self.bfrac)) for i in range(self.reps)] for i in range(self.sets)])
        self.meanset = np.array([self.noised[i].mean(axis=0) for i in range(len(self.noised))])
        self.stdset = np.array([self.noised[i].std(axis=0) for i in range(len(self.noised))])


class WymSimSubset:

    #default init values
    __dparms = np.array([3.,2.,0.1,100.])
    __drtot = np.array([0.001,0.0025,0.005,0.01,0.025,0.05])
    __dlrange = np.array([0.001,100.,2.])
    __drp = 3
    __dst = 3333
    __dns = 0.05
    __dmodfunc = models.wymfunc

    def __init__(self,modfunc=__dmodfunc,parms=__dparms,rtot=__drtot,ligrange=__dlrange,reps=__drp,sets=__dst,noise=__dns):
        '''Returns a Wyman model object'''
        self.sets = sets
        self.reps = reps
        self.modfunc = modfunc
        self.parms = parms
        self.rtot = rtot
        self.noise = noise
        self.ligs = np.array([helpers.dilser(ligrange[0],ligrange[1],ligrange[2]) for i in range(len(self.rtot))])
        self.bfrac = np.array([self.modfunc(self.parms,self.ligs[i],self.rtot[i]) for i in range(len(self.rtot))])
        self.noised = np.array([[np.random.normal(self.bfrac,(self.noise*self.bfrac)) for i in range(self.reps)] for i in range(self.sets)])
        self.meanset = np.array([self.noised[i].mean(axis=0) for i in range(len(self.noised))])

    def subset(self,lo,hi):
        self.meanset = np.array([[j[lo:hi] for j in i] for i in self.meanset])
        self.ligs = np.array([j[lo:hi] for j in self.ligs])
        self.bfrac = np.array([j[lo:hi] for j in self.ligs])


class WymSimHardlig:

    #default init values
    __dparms = np.array([3.,2.,0.1,100.])
    __drtot = np.array([0.001,0.0025,0.005,0.01,0.025,0.05])
    __drp = 3
    __dst = 3333
    __dns = 0.05
    __dmodfunc = models.wymfunc

    def __init__(self,ligs,modfunc=__dmodfunc,parms=__dparms,rtot=__drtot,reps=__drp,sets=__dst,noise=__dns):
        '''Returns a Wyman model object'''
        self.sets = sets
        self.reps = reps
        self.modfunc = modfunc
        self.parms = parms
        self.rtot = rtot
        self.noise = noise
        self.ligs = ligs
        self.bfrac = np.array([self.modfunc(self.parms,self.ligs[i],self.rtot[i]) for i in range(len(self.rtot))])
        self.noised = np.array([[np.random.normal(self.bfrac,(self.noise*self.bfrac)) for i in range(self.reps)] for i in range(self.sets)])
        self.meanset = np.array([self.noised[i].mean(axis=0) for i in range(len(self.noised))])
        self.stdset = np.array([self.noised[i].std(axis=0) for i in range(len(self.noised))])

class WymFit:

    __dguess = np.array([10.,10.,10.,100.])
    __dbounds = ((0.,0.,0.,0.,),(100.,100.,100.,10000.))

    def __init__(self, model, guess=__dguess, bounds=__dbounds, weight=0, **kwargs):
        '''Takes Wyman model object and performs nls fitting with optional weighting (1 = weight by s.d, 2 = weight by Y)'''
        self.model = model
        self.guess = guess
        self.bounds = bounds
        self.fits = helpers.fitwrap(self,weight=weight,**kwargs)
        self.ests = np.array([self.fits[i].x for i in range(len(self.fits))])
        self.k11 = self.ests[:,0]
        self.k21 = self.ests[:,1]
        self.k22 = self.ests[:,2]
        self.l20 = self.ests[:,3]
