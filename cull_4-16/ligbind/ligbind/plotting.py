__all__ = ["semlog","viewf","resplot","semlogerr"]

import numpy as np
import matplotlib.pyplot as plt

def semlog(lig,sat,symb,labs=None):
    colors =['b','g','r','c','m','y','k','dimgrey','lightgrey','darkgrey','silver','gainsboro','salmon','teal','sage','wheat','violet','plum','thistle']
    if labs is None:
        for i in range(len(sat)):
            plt.semilogx(lig[i],sat[i],symb,color=colors[i])
    else:
        for i in range(len(sat)):
            plt.semilogx(lig[i],sat[i],symb, label=labs[i],color=colors[i])
            plt.legend(loc=4,prop={'size':6},numpoints=1)

def viewf(model, parms, ligs, sats, rtots, symbol='--', index=None):
    """parms is the least_squares result.x"""
    colors = ['b','g','r','c','m','y','k','dimgrey','lightgrey','darkgrey','silver','gainsboro','salmon','teal','sage','wheat','violet','plum','thistle']
    if index is None:
        for i in range(len(ligs)):
            plt.semilogx(ligs[i], model(parms,ligs[i],rtots[i]),symbol, color=colors[i])
            plt.semilogx(ligs[i], sats[i],'.',color=colors[i])
    else:
        plt.semilogx(ligs[index], model(parms,ligs[index],rtots[index]),symbol, color=colors[index])
        plt.semilogx(ligs[index], sats[index],'.',color=colors[index])

def resplot(model,result, ligs, rtots):
    """takes result=output from scipy.optimize.least_squares, ligs=ligand values from data,
    rtots=total receptor concentration from data and returns a plot of residuals vs fitted values"""
    parm = result.x
    resid = result.fun
    fitted = np.concatenate([model(parm,ligs[i],rtots[i]) for i in range(len(ligs))])
    plt.plot(fitted,resid,'.')


def semlogerr(ligset,datset,errors):
    '''numpy arrays ligset, datset, and errors must have same shape'''
    colors = ['b','g','r','c','m','y','k','dimgrey','lightgrey','darkgrey','silver','gainsboro','salmon','teal','sage','wheat','violet','plum','thistle']
    for i in range(ligset.shape[0]):
        plt.xscale('log')
        plt.errorbar(ligset[i], datset[i], errors[i], color=colors[i], linestyle='none',fmt='.')
