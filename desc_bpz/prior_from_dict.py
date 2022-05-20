from desc_bpz.bpz_tools_py3 import *
import numpy as np

def function(z, m, paramdict, dummy):
    """Generic prior that reads in parameters from a
    dictionary and applies them, rather than hard
    coded    
    """

    fo_arr = paramdict['fo_arr']
    kt_arr = paramdict['kt_arr']
    alpha_arr = paramdict['a_arr']
    zo_arr = paramdict['zo_arr']
    km_arr = paramdict['km_arr']
    m0 = paramdict['mo']
    numpertype = paramdict['nt_array']
    # to keep consistent with old form, just copy nt here
    # get rid of use as argument passed in, instead it is
    # read from dictionary, but we'll leave as a dummy
    # argument for now to have same argument list as
    # the old prior form.
    nt = numpertype
    
    global zt_at_a
    nz=len(z)
    momin_hdf=m0
    if m>32.: m=32.
    if m<m0: m=m0    

    ## nt Templates = nell Elliptical + nsp Spiral + nSB starburst
    #try:  # nt is a list of 3 values
    #    nell, nsp, nsb = nt
    #except:  # nt is a single value
    #    nell = 1  # 1 Elliptical in default template set
    #    nsp = 2   # 2 Spirals in default template set
    #    nsb = nt - nell - nsp  # rest Irr/SB
    #nn = nell, nsp, nsb
    
    # See Table 1 of Benitez00
    #a  = 2.465,  1.806,  0.906
    #zo = 0.431,  0.390,  0.0626
    #km = 0.0913, 0.0636, 0.123
    #k_t= 0.450,  0.147
    
    a  = np.repeat(alpha_arr, numpertype)
    zo = np.repeat(zo_arr, numpertype)
    km = np.repeat(km_arr, numpertype)
    # there are one less kt and f0 values, as the final one
    # is defined as 1 - sum(others)
    k_t = np.repeat(kt_arr, numpertype[:-1])
    # but, we have to divide the total probability by the
    # number of templates in each broad type
    fot_pertemp = fo_arr / numpertype[:-1]
    fo_t = np.reapeat(fot_pertemp, numpertype[:-1])
    
    dm=m - momin_hdf
    zmt=clip(zo + km * dm, 0.01, 15.)
    zmt_at_a=zmt**(a)
    #We define z**a as global to keep it 
    #between function calls. That way it is 
    # estimated only once
    try:
        zt_at_a.shape
    except NameError:
        zt_at_a=power.outer(z,a)
	
    #Morphological fractions
    # need to add the fractions for the final type
    # that is 1 - others
    n_most = np.sum(numpertype[:-1])
    f_t=np.zeros((len(a),),float)
    f_t[:n_most]=fo_t*exp(-k_t*dm)
    f_t[n_most:]=(1.-add.reduce(f_t[:n_most]))/float(numpertype[-1])
    #Formula:
    #zm=zo+km*(m_m_min)
    #p(z|T,m)=(z**a)*exp(-(z/zm)**a)
    p_i=zt_at_a[:nz,:nt]*exp(-clip(zt_at_a[:nz,:nt]/zmt_at_a[:nt],0.,700.))
    #This eliminates the very low level tails of the priors
    norm=add.reduce(p_i[:nz,:nt],0)
    p_i[:nz,:nt]=where(less(p_i[:nz,:nt]/norm[:nt],1e-2/float(nz)),
		      0.,p_i[:nz,:nt]/norm[:nt])
    norm=add.reduce(p_i[:nz,:nt],0)
    p_i[:nz,:nt]=p_i[:nz,:nt]/norm[:nt]*f_t[:nt]
    return p_i
