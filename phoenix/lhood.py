def lnlik4p(parms,rtot,data,ligs):
    datac = np.concatenate(data)
    modparms = parms[0:-1]
    rtots = rtot
    f = parms[-1]
    model = lb.models.wymfunc(modparms,ligs,rtots)
    invsig2 = 1.0/np.square(f*datac) #do I need to change this to DATAC?
    return -0.5*(np.sum((datac-model)**2*invsig2 - np.log(invsig2)))
