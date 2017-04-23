import numpy as np
#import matplotlib.pyplot as plt
from scipy import optimize

#helper functions
def dilser(low=0.001, limit=100., dilfactor=2.):
    '''returns a numy array dilution series from low to limit'''
    #replace this with a generator/iterator someday
    a = [low]
    while a[-1] <= limit:
        a.append(a[len(a)-1]*dilfactor)       
    return np.array(a)
    
class WymanSim(object):
  
    def genfunc(parm,lig,rtot):
        '''Generates ligand binding data according to Wyman model.
        Takes numpy arrays containing parameters k11,k21,k22,l20; 
        free ligand concentrations, and total receptor concentrations.
        Must be iterated against ligand data, i.e., for loop required when
        used for more than one receptor concentration. In fitting functions,
        the datasets are concentatenated for faster operation'''
    
        k11 = parm[0]
        k21 = parm[1]
        k22 = parm[2]
        l20 = parm[3]
    
        rfree = (((-1 - k11*lig)) + \
        (np.sqrt((np.square((1 + k11*lig)) + \
        8.*l20*rtot*(1 + k21*lig + k21*k22*(np.square(lig))))))) \
        / (4*l20*(1 + k21*lig + k21*k22*(np.square(lig))))    
       
        bfrac = (k11*lig + l20*k21*rfree*lig + \
        2*l20*k21*k22*rfree*(np.square(lig))) \
        / (1 + 2*l20*rfree + k11*lig + \
        2*l20*k21*rfree*lig + 2*l20*k21*k22*rfree*(np.square(lig)))
    
        return bfrac
    
    #default init values -- pulled out here for a more concise __init__ definition
    dparms = [3.,2.,0.1,100.]
    drtot = [0.001,0.0025,0.005,0.01,0.025,0.05]
    dlrange = [0.001,100.,2.]
    drp = 3
    dst = 3333
    dns = 0.05
    
    def __init__(self,parms=dparms,rtot=drtot,ligrange=dlrange,reps=drp,sets=dst,noise=dns):
        '''Returns a Wyman model object'''
        self.sets = sets
        self.reps = reps
        self.parms = np.array(parms)
        self.rtot = np.array(rtot)
        self.ligs = np.array([dilser(ligrange[0],ligrange[1],ligrange[2]) for i in range(len(self.rtot))])
        self.bfrac = np.array([WymanSim.genfunc(self.parms,self.ligs[i],self.rtot[i]) for i in range(len(self.rtot))])
        self.noise = noise
        self.noised = np.array([[np.random.normal(self.bfrac,(self.noise*self.bfrac)) for i in range(self.reps)] for i in range(self.sets)])
        self.meanset = np.array([self.noised[i].mean(axis=0) for i in range(len(self.noised))])
        self.stdset = np.array([self.noised[i].std(axis=0) for i in range(len(self.noised))])

        
class WymanSimFit(object):
    
    #MODIFIED FUNCTION FOR USE WITH 'PERFECT' RECTANGULAR DATA ARRAYS
    #SUCH AS THOSE FROM SIMULATION
    def fitfunc(parm,lig,data,rtot,eps=None):
        k11 = parm[0]
        k21 = parm[1]
        k22 = parm[2]
        l20 = parm[3]
    
        ligc = np.concatenate(lig)
        datac = np.concatenate(data)
        rtot_tr = rtot[:,None]
    
        rfree = (((-1 - k11*ligc)) + \
        (np.sqrt(np.concatenate((np.square((1 + k11*lig)) + \
        8.*l20*rtot_tr*(1 + k21*lig + k21*k22*(np.square(lig)))))))) \
        / (4*l20*(1 + k21*ligc + k21*k22*(np.square(ligc))))    
       
        bfrac = (k11*ligc + l20*k21*rfree*ligc + \
        2*l20*k21*k22*rfree*(np.square(ligc))) \
        / (1 + 2*l20*rfree + k11*ligc + \
        2*l20*k21*rfree*ligc + 2*l20*k21*k22*rfree*(np.square(ligc)))
    
        residual = (bfrac - datac)

        if eps is None:
            return residual
        else:
            weights = 1/(np.concatenate(eps))
            return (residual*weights)
    
    def optwrap(self,err=None,**kwargs):
        if err is None:
            return [optimize.least_squares(WymanSimFit.fitfunc,self.guess,bounds=self.bounds, \
                    args=(self.model.ligs,self.model.meanset[i],self.model.rtot),**kwargs) \
                    for i in range(self.model.meanset.shape[0])]
        else:
            return [optimize.least_squares(WymanSimFit.fitfunc,self.guess,bounds=self.bounds, \
                    args=(self.model.ligs,self.model.meanset[i],self.model.rtot,err[i]),**kwargs) \
                    for i in range(self.model.meanset.shape[0])]
    
    def fitwrap(self,weight,**kwargs):
        if weight == 1:
            return WymanSimFit.optwrap(self,err=self.model.stdset,**kwargs)
        elif weight == 2: 
            return WymanSimFit.optwrap(self,err=(self.model.noise*self.model.meanset),**kwargs)
        elif weight == 3:
            return WymanSimFit.optwrap(self,err=(self.model.meanset),**kwargs)
        else:
            return WymanSimFit.optwrap(self,**kwargs)
    
    
    #default values
    dguess = [10.,10.,10.,100.]
    dbounds = ((0.,0.,0.,0.,),(100.,100.,100.,10000.))
    
    def __init__(self, model, guess=dguess, bounds=dbounds, weight=0, **kwargs):
        '''Takes Wyman Model Object'''
        self.model = model
        self.guess = np.array(guess)
        self.bounds = bounds
        self.fits = WymanSimFit.fitwrap(self,weight,**kwargs)
        self.ests = np.array([self.fits[i].x for i in range(len(self.fits))])
        self.k11 = self.ests[:,0]
        self.k21 = self.ests[:,1]
        self.k22 = self.ests[:,2]
        self.l20 = self.ests[:,3]
         