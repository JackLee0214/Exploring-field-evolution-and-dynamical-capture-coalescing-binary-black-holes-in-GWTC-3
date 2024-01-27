import bilby
import numpy as np
import pickle
from pandas.core.frame import DataFrame
import h5py
from astropy.cosmology import Planck15
from bilby.hyper.likelihood import HyperparameterLikelihood
from bilby.core.prior import Interped, Uniform, LogUniform
from bilby.core.prior import TruncatedGaussian as TG
from bilby.core.prior import Beta as Bt
from bilby.core.prior import PowerLaw 
from scipy.interpolate import RegularGridInterpolator, interp1d
import astropy.units as u
import sys
from scipy.integrate import quad,cumtrapz
from scipy.special._ufuncs import xlogy, erf
import json
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.stats import gaussian_kde
import matplotlib.gridspec as gridspec
from tqdm import tqdm

####
#ploting
from Spin_Ori_model_libs import PLS_G1,PLS_G1_un, PLS_mass, PL_un, Two_pop, PLS_dyn_mass_un,PLS_dyn_G1_m,PLS_G2_m,PLS_dyn_m, PLS_G1_un, PPC_selected, hyper_Two_pop, llh_z


def DF_ct1(ct1,sigma_t,zeta,zmin=-1):
    align=TG(1,sigma_t,zmin,1)
    iso=Uniform(-1,1)
    return align.prob(ct1)*zeta+iso.prob(ct1)*(1-zeta)

def Two_pop_plot_mass_dist(post,colors,filename,Nsample,Three_pop=False):
    label1='Aligned spin'
    label2='Isotropic Spin'
    if Three_pop:
        label1='Field'
        label2='Dynamical'

    fig=plt.figure(figsize=(10,8))
    
    gs = gridspec.GridSpec(3, 1, figure=fig)
    ax1 = fig.add_subplot(gs[:2])
    ax2 = fig.add_axes([0.50,0.70,0.40,0.22])
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2:])
    ax4 = fig.add_subplot(gs1[:,:1])
    ax3 = fig.add_subplot(gs1[:,1:])
    
    ######mass
    parameters={}
    #low_idx=np.where(np.array(post['r2'])<0.25)
    indx=np.random.choice(np.arange(len(post['log_likelihood'])),size=Nsample)
    keys=['mmin1','mmax1','alpha1','delta1','beta1','alpha2','r2','mmin2','mmax2','delta2','beta2','sigma_t1','zmin1','mu_a','sigma_a','lgR0']
    keys1=['alpha1','mmin1','mmax1','delta1']
    keys2=['alpha2','mmin2','mmax2','delta2']
    if Three_pop:
        keys=keys+['sigma_t3','zeta3','k','r3']
    for i in np.arange(10):
        keys.append('N'+str(i+1))
        keys1.append('N'+str(i+1))
    for i in np.arange(10):
        keys.append('q'+str(i+1))
        keys2.append('q'+str(i+1))
    keys1.append('beta1')
    keys2.append('beta2')

    for key in keys:
        #parameters.update({key:np.array(post[key])[low_idx]})
        parameters.update({key:np.array(post[key])[indx]})
    if 'mu_t1' in post.keys():
        parameters.update({'mu_t1':np.array(post['mu_t1'])[indx]})
    else:
        parameters.update({'mu_t1':np.ones(Nsample)})
    

    m1pdfs=[]
    field=[]
    dyn=[]
    f_ct=[]
    d_ct=[]
    f_q=[]
    d_q=[]
    spin_a=[]
    a_sam=np.linspace(0,1,100)
    ct_sam=np.linspace(-1,1,100)
    m1_sam=np.linspace(2,60,500)
    m2_sam=np.linspace(2,60,499)
    q_sam=np.linspace(0.01,1,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    xx=np.linspace(2,60,500)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    
    j,k=np.meshgrid(m1_sam,q_sam)
    dj = m1_sam[1]-m1_sam[0]
    for i in tqdm(np.arange(Nsample)):
        para1=[parameters[key][i] for key in keys1]
        para2=[parameters[key][i] for key in keys2]
        R0=10**parameters['lgR0'][i]
        field.append(np.sum(PLS_G1(x,y,*para1)*dy,axis=0)*(1-parameters['r2'][i])*R0)
        dyn.append(np.sum(PLS_G1(x,y,*para2)*dy,axis=0)*parameters['r2'][i]*R0)
        #field.append(PLS_mass(m1_sam,*para1)*(1-parameters['r2'][i]))
        #dyn.append(PLS_mass(m1_sam,*para2)*parameters['r2'][i])
        f_ct.append(TG(parameters['mu_t1'][i],parameters['sigma_t1'][i],parameters['zmin1'][i],1).prob(ct_sam))
        if Three_pop:
            d_ct.append(DF_ct1(ct_sam,parameters['sigma_t3'][i],parameters['zeta3'][i]*parameters['k'][i]*parameters['r3'][i]))
        f_q.append(np.sum(PLS_G1(j,k*j,*para1)*j*dj,axis=1))
        d_q.append(np.sum(PLS_G1(j,k*j,*para2)*j*dj,axis=1))
        spin_a.append(TG(parameters['mu_a'][i],parameters['sigma_a'][i],0,1).prob(a_sam))
    
    spin_a=np.array(spin_a)
    a_pup=np.percentile(spin_a,95,axis=0)
    a_plow=np.percentile(spin_a,5,axis=0)
    a_pmid=np.percentile(spin_a,50,axis=0)


    ax4.xaxis.grid(True,which='major',ls=':',color='grey')
    ax4.yaxis.grid(True,which='major',ls=':',color='grey')
    ax4.fill_between(a_sam,a_plow,a_pup,color='orange',alpha=0.3)
    ax4.plot(a_sam,a_plow,color='orange',alpha=0.8,lw=0.5)
    ax4.plot(a_sam,a_pup,color='orange',alpha=0.8,lw=0.5)
    ax4.plot(a_sam,a_pmid,color='orange',alpha=0.8)

    ax4.set_xlim(0,1)
    ax4.set_xlabel(r'$a_{1,2}$')
    ax4.set_ylabel(r'$p(a)$')
    
    field=np.array(field)
    field_pup=np.percentile(field,95,axis=0)
    field_plow=np.percentile(field,5,axis=0)
    field_pmid=np.percentile(field,50,axis=0)
    field_pmean=np.mean(field,axis=0)
    dyn=np.array(dyn)
    dyn_pup=np.percentile(dyn,95,axis=0)
    dyn_plow=np.percentile(dyn,5,axis=0)
    dyn_pmid=np.percentile(dyn,50,axis=0)
    dyn_pmean=np.mean(dyn,axis=0)

    m1pdfs=field+dyn
    pup=np.percentile(m1pdfs,95,axis=0)
    plow=np.percentile(m1pdfs,5,axis=0)
    pmean=np.mean(m1pdfs,axis=0)
    print('ploting')
    ax1.xaxis.grid(True,which='major',ls=':',color='grey')
    ax1.yaxis.grid(True,which='major',ls=':',color='grey')
    ax1.fill_between(m1_sam,field_plow,field_pup,color=colors[0],alpha=0.3,label=label1)
    ax1.plot(m1_sam,field_plow,color=colors[0],alpha=0.8,lw=0.5)
    ax1.plot(m1_sam,field_pup,color=colors[0],alpha=0.8,lw=0.5)
    ax1.plot(m1_sam,field_pmid,color=colors[0],alpha=0.9)
    ax1.fill_between(m1_sam,dyn_plow,dyn_pup,color=colors[1],alpha=0.3,label=label2)
    ax1.plot(m1_sam,dyn_plow,color=colors[1],alpha=0.8,lw=0.5)
    ax1.plot(m1_sam,dyn_pup,color=colors[1],alpha=0.8,lw=0.5)
    ax1.plot(m1_sam,dyn_pmid,color=colors[1],alpha=0.9)
    
    ax1.set_yscale('log')
    ax1.set_ylim(1e-3,1e2)
    ax1.set_xlim(0,60)
    ax1.set_xlabel(r'$m_{1}/M_{\odot}$')
    ax1.set_ylabel(r'$\frac{{\rm d}\mathcal{R}(z=0)}{{\rm d}m_{1}}~[{\rm Gpc}^{-3}~{\rm yr}^{-1}~M_{\odot}^{-1}]$')
    #ax1.set_title('The Stellar-formed BBHs')
    ax1.legend(loc=2)
    
    ###cos tilt
    f_ct=np.array(f_ct)
    f_ct_pup=np.percentile(f_ct,95,axis=0)
    f_ct_plow=np.percentile(f_ct,5,axis=0)
    f_ct_pmid=np.percentile(f_ct,50,axis=0)
    f_ct_pmean=np.mean(f_ct,axis=0)
    d_ct_pmean=Uniform(-1,1).prob(ct_sam)
    if Three_pop:
        d_ct=np.array(d_ct)
        d_ct_pup=np.percentile(d_ct,95,axis=0)
        d_ct_plow=np.percentile(d_ct,5,axis=0)
        d_ct_pmid=np.percentile(d_ct,50,axis=0)
        d_ct_pmean=np.mean(d_ct,axis=0)
    
    ax3.xaxis.grid(True,which='major',ls=':',color='grey')
    ax3.yaxis.grid(True,which='major',ls=':',color='grey')
    print('ploting')

    ax3.fill_between(ct_sam,f_ct_plow,f_ct_pup,color=colors[0],alpha=0.3)
    ax3.plot(ct_sam,f_ct_pmid,color=colors[0],alpha=0.8)
    ax3.plot(ct_sam,d_ct_pmean,color=colors[1],alpha=0.8)
    if Three_pop:
        ax3.fill_between(ct_sam,d_ct_plow,d_ct_pup,color=colors[1],alpha=0.3)

    ax3.set_xlabel(r'$\cos\theta_{1,2}$')
    #ax3.set_xlabel(r'$\cos\theta_{1}$')
    ax3.set_ylabel(r'$p(\cos\theta)$')
    ax3.set_xlim(-1,1)
    
    ######mass ratio
    f_q=np.array(f_q)
    f_q_pup=np.percentile(f_q,95,axis=0)
    f_q_plow=np.percentile(f_q,5,axis=0)
    f_q_pmid=np.percentile(f_q,50,axis=0)
    d_q=np.array(d_q)
    d_q_pup=np.percentile(d_q,95,axis=0)
    d_q_plow=np.percentile(d_q,5,axis=0)
    d_q_pmid=np.percentile(d_q,50,axis=0)
    
    
    ax2.xaxis.grid(True,which='major',ls=':',color='grey')
    ax2.yaxis.grid(True,which='major',ls=':',color='grey')
    ax2.fill_between(q_sam,f_q_plow,f_q_pup,color=colors[0],alpha=0.3,label=label1)
    ax2.plot(q_sam,f_q_plow,color=colors[0],alpha=0.8,lw=0.5)
    ax2.plot(q_sam,f_q_pup,color=colors[0],alpha=0.8,lw=0.5)
    ax2.plot(q_sam,f_q_pmid,color=colors[0],alpha=0.9)
    ax2.fill_between(q_sam,d_q_plow,d_q_pup,color=colors[1],alpha=0.3,label=label2)
    ax2.plot(q_sam,d_q_plow,color=colors[1],alpha=0.8,lw=0.5)
    ax2.plot(q_sam,d_q_pup,color=colors[1],alpha=0.8,lw=0.5)
    ax2.plot(q_sam,d_q_pmid,color=colors[1],alpha=0.9)
    ax2.set_yscale('log')
    ax2.set_ylim(5e-3,1e2)
    ax2.set_xlabel(r'$q$')
    ax2.set_ylabel(r'$p(q)$')
    ax2.set_xlim(0,1)
    
    
    plt.tight_layout()
    plt.savefig(filename)
    
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.lines import Line2D
import matplotlib.patches
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import cm
import glob
from scipy.stats import gaussian_kde
import sys
import pandas as pd

def plot_2d(ax,post,paras,xlim,ylim):
    beta_1=np.array(post[paras[0]])
    beta_2=np.array(post[paras[1]])
    idx=np.where(beta_1>beta_2)
    CL=float(len(beta_1[idx]))/float(len(beta_1))
    print('C.L.:',CL)

    # Plotting bounds
    x_min = xlim[0]
    x_max = xlim[1]
    y_min = ylim[0]
    y_max = ylim[1]
    
    kde = gaussian_kde([beta_1,beta_2])
    x_gridpoints = np.linspace(x_min,x_max,60)
    y_gridpoints = np.linspace(y_min,y_max,59)
    x_grid,y_grid = np.meshgrid(x_gridpoints,y_gridpoints)
    z_grid = kde([x_grid.reshape(-1),y_grid.reshape(-1)]).reshape(y_gridpoints.size,x_gridpoints.size)

    # Sort the resulting z-values to get estimates of where to place 50% and 90% contours
    sortedVals = np.sort(z_grid.flatten())[::-1]
    cdfVals = np.cumsum(sortedVals)/np.sum(sortedVals)
    i50 = np.argmin(np.abs(cdfVals - 0.50))
    i90 = np.argmin(np.abs(cdfVals - 0.90))
    val50 = sortedVals[i50]
    val90 = sortedVals[i90]

    # Draw contours at these locations
    CS = ax.contour(x_gridpoints,y_gridpoints,z_grid,levels=(val90,val50),linestyles=('dashed','solid'),colors='orange',linewidths=1,label='Without HM')
    #ax.legend(loc='lower right')
    return ax

def plot_2d_beta(post,filename,paras=['beta1','beta2'],labels=[r'$\beta_{\rm F}(\beta_{\rm A})$',r'$\beta_{\rm D}(\beta_{\rm I})$'],xlim=[-8,8],ylim=[-8,8],color='#CC6677',plot_double=False,post2=None):
    beta_1=np.array(post[paras[0]])
    beta_2=np.array(post[paras[1]])
    idx=np.where(beta_1>beta_2)
    CL=float(len(beta_1[idx]))/float(len(beta_1))
    print('C.L.:',CL)

    # Plotting bounds
    x_min = xlim[0]
    x_max = xlim[1]
    y_min = ylim[0]
    y_max = ylim[1]

    # Create a linear colormap between white and the "Broken PL" model color
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white",color])

    # Plot 2D alpha1 vs. alpha2 posterior
    fig,ax = plt.subplots(figsize=(4,3))
    ax.hexbin(beta_1,beta_2,cmap=cmap,gridsize=25,mincnt=1,extent=(x_min,x_max,y_min,y_max),linewidths=(0.,))

    # The next chunk of code creates contours
    # First construct a KDE and evaluate it on a grid across alpha1 vs. alpha2 space
    kde = gaussian_kde([beta_1,beta_2])
    x_gridpoints = np.linspace(x_min,x_max,60)
    y_gridpoints = np.linspace(y_min,y_max,59)
    x_grid,y_grid = np.meshgrid(x_gridpoints,y_gridpoints)
    z_grid = kde([x_grid.reshape(-1),y_grid.reshape(-1)]).reshape(y_gridpoints.size,x_gridpoints.size)

    # Sort the resulting z-values to get estimates of where to place 50% and 90% contours
    sortedVals = np.sort(z_grid.flatten())[::-1]
    cdfVals = np.cumsum(sortedVals)/np.sum(sortedVals)
    i50 = np.argmin(np.abs(cdfVals - 0.50))
    i90 = np.argmin(np.abs(cdfVals - 0.90))
    val50 = sortedVals[i50]
    val90 = sortedVals[i90]

    # Draw contours at these locations
    CS = ax.contour(x_gridpoints,y_gridpoints,z_grid,levels=(val90,val50),linestyles=('dashed','solid'),colors='k',linewidths=1)

    # Draw a diagonal line for illustration purposes
    ax.plot(np.arange(-8,12),np.arange(-8,12),lw=1,ls='--',color='black',alpha=0.75)
    if plot_double:
        ax=plot_2d(ax,post2,paras,xlim,ylim,)
    # Misc formatting
    ax.grid(True,dashes=(1,3))
    ax.set_xlim(x_min,x_max)
    ax.set_ylim(y_min,y_max)
    ax.set_xlabel(labels[0],fontsize=14)
    ax.set_ylabel(labels[1],fontsize=14)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(filename,bbox_inches='tight')

def Field_m(m1,m2,mmin1,mmax1,alpha1,delta1,beta1,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,sigma_t1,\
        mmin2,mmax2,alpha2,delta2,beta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,sigma_t2,r2,mu_a,sigma_a,zmin1):
    pm12=PLS_mass(m1,alpha1,mmin1,mmax1,delta1,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10)*PLS_mass(m2,alpha1,mmin1,mmax1,delta1,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10)*(m2/m1)**beta1
    return pm12

def Dyn_m_shift(m,mmin1,mmax1,alpha1,delta1,beta1,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,sigma_t1,\
        mmin2,mmax2,alpha2,delta2,beta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,sigma_t2,r2,mu_a,sigma_a,zmin1):
    return PLS_mass(m,alpha2+2,mmin2,mmax2,delta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10)

def Dyn_m(m,mmin1,mmax1,alpha1,delta1,beta1,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,sigma_t1,\
        mmin2,mmax2,alpha2,delta2,beta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,sigma_t2,r2,mu_a,sigma_a,zmin1):
    return PLS_mass(m,alpha2,mmin2,mmax2,delta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10)


def XRB_PPC(post,colors,n_catalogs,N,filename,channel):
    if 'sigma_t2' not in post.keys():
        post['sigma_t2']=np.ones(len(post['mmin1']))*100
    x=Uniform(2,100).sample(size=200000)
    y=Uniform(2,100).sample(size=200000)
    indx=np.random.choice(np.arange(len(post['log_likelihood'])),size=n_catalogs)
    ppd_m1 = np.zeros((N,n_catalogs))
    ppd_m1_field = np.zeros((N,n_catalogs))
    ppd_m_Dyn = np.zeros((N,n_catalogs))
    ppd_m_Dyn_shifted = np.zeros((N,n_catalogs))
    ppd_m1_field_obs = np.zeros((N,n_catalogs))
    need_keys=['mu_a', 'sigma_a', 'beta1', 'mmin1', 'mmax1', 'alpha1', 'delta1', 'sigma_t1', 'beta2', 'mmin2', 'mmax2', 'alpha2', 'delta2', 'r2','N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10','q1', 'q2','q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9','q10','zmin1','sigma_t2']
    parameters={key:np.array(post[key])[indx] for key in need_keys}

    
    fig = plt.figure(figsize=(5,3))
    ax = fig.add_subplot()
    ax.xaxis.grid(True,which='major',ls=':',color='grey')
    ax.yaxis.grid(True,which='major',ls=':',color='grey')
    ######the HMXBs
    ax.errorbar([10.90,15.65,21.20],(np.arange(3)+np.arange(3)+1)/6.,[[1./6.,1./6.,1./6.],[1./6.,1./6.,1./6.]],[[1.4,1.45,2.2],[1.4,1.45,2.2]],fmt=' ',elinewidth=0.2,capsize=2,label='HMXB BHs',color="#e31a1c")
    
    #ax.errorbar([10.2,10.90,15.65,21.20],(np.arange(4)+np.arange(4)+1)/8.,[[1./8.,1./8.,1./8.,1/8.],[1./8.,1./8.,1./8.,1/8.]],[[2.0,1.4,1.45,2.2],[2.0,1.4,1.45,2.2]],fmt=' ',elinewidth=0.2,capsize=2,label='HMXB BHs',color="#e31a1c")
    
    if channel=='field':
        for i in tqdm(range(n_catalogs)):
            para={key:parameters[key][i] for key in parameters.keys()}
            ps_field = Field_m(x,y,**para)
            ps_field /= np.sum(ps_field)
            chosenInd = np.random.choice(np.arange(len(ps_field)),p=ps_field,size=N)
            ppd_m1_field[:,i] = x[chosenInd]
            #####source frame cdf
    
        ax.fill_betweenx(y = np.linspace(0,1,len(ppd_m1_field[:,0])),
                            x1 = np.quantile(np.sort(ppd_m1_field,axis=0),0.05,axis=1),
                            x2 = np.quantile(np.sort(ppd_m1_field,axis=0),0.95,axis=1),
                            color = colors[1], alpha=0.2, label=r'$m_1$ distribution of Field BBHs'
                            )
        
        ax.fill_betweenx(y = [0,1./3.,1./3.,2./3.,2./3.,1.],
                            x1 = np.sort(np.append(np.quantile(np.sort(ppd_m1_field[:3],axis=0),0.05,axis=1),np.quantile(np.sort(ppd_m1_field[:3],axis=0),0.05,axis=1))),
                            x2 = np.sort(np.append(np.quantile(np.sort(ppd_m1_field[:3],axis=0),0.95,axis=1),np.quantile(np.sort(ppd_m1_field[:3],axis=0),0.95,axis=1))),
                            color = colors[2], alpha=0.2
                            )
        
        ax.plot(np.sort(np.append(np.quantile(np.sort(ppd_m1_field[:3],axis=0),0.5,axis=1),np.quantile(np.sort(ppd_m1_field[:3],axis=0),0.5,axis=1))),[0,1./3.,1./3.,2./3.,2./3.,1.], color = colors[2], alpha=1, label='3 prediction from Field BBHs')
        ax.set_xlabel(r"$m_1~[M_{\odot}]$")
        

    else:
        for i in tqdm(range(n_catalogs)):
            para={key:parameters[key][i] for key in parameters.keys()}
            ps_Dyn = Dyn_m_shift(x,**para)
            ps_Dyn /= np.sum(ps_Dyn)
            chosenInd = np.random.choice(np.arange(len(ps_Dyn)),p=ps_Dyn,size=N)
            ppd_m_Dyn_shifted[:,i] = x[chosenInd]
                    
            ps_Dyn = Dyn_m(x,**para)
            ps_Dyn /= np.sum(ps_Dyn)
            chosenInd = np.random.choice(np.arange(len(ps_Dyn)),p=ps_Dyn,size=N)
            ppd_m_Dyn[:,i] = x[chosenInd]
        
        xx=np.linspace(5,100,1000)
        yy=PL_un(xx,7,40,2.35,5)
        yy_cdf=np.cumsum(yy)/np.sum(yy)
        #ax.plot(xx,yy_cdf,colors[9],label=r'$\alpha=2.35$,$m_{\rm min}=7$,$m_{\rm max}=40$,$\delta_{\rm m}=5$')
        
        #####dynamical cdf
        ax.fill_betweenx(y = np.linspace(0,1,len(ppd_m_Dyn[:,0])),
                            x1 = np.quantile(np.sort(ppd_m_Dyn,axis=0),0.05,axis=1),
                            x2 = np.quantile(np.sort(ppd_m_Dyn,axis=0),0.95,axis=1),
                            color = colors[5], alpha=0.2, label='Dynamical channel mass function'
                            )
        ax.fill_betweenx(y = np.linspace(0,1,len(ppd_m_Dyn_shifted[:,0])),
                            x1 = np.quantile(np.sort(ppd_m_Dyn_shifted,axis=0),0.05,axis=1),
                            x2 = np.quantile(np.sort(ppd_m_Dyn_shifted,axis=0),0.95,axis=1),
                            color = colors[1], alpha=0.2, label=r'initial mass function ($\alpha=\alpha_{\rm D}+2$)'
                            )
                                
        ax.set_xlabel(r"$m~[M_{\odot}]$")
    
    #######
    ax.set_xlim(0,60)
    ax.set_ylim(0,1)
    ax.set_ylabel('CDF')
    
    plt.legend(loc='lower right',fontsize='small')
    plt.tight_layout()
    plt.savefig(filename,bbox_inches='tight')


def Field_mact(m1,m2,a1,a2,ct1,ct2,z,mmin1,mmax1,alpha1,delta1,beta1,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,sigma_t1,\
        mmin2,mmax2,alpha2,delta2,beta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,sigma_t2,r2,mu_a,sigma_a,gamma=2.7,zmin1=None):
    pm12=PLS_G1(m1,m2,alpha1,mmin1,mmax1,delta1,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,beta1)
    pact12=TG(mu_a,sigma_a,0,1).prob(a1)*TG(mu_a,sigma_a,0,1).prob(a2)**TG(1,sigma_t1,-1,1).prob(ct1)*TG(1,sigma_t1,-1,1).prob(ct2)
    return pm12*pact12*llh_z(z,gamma)*(1-r2)

def Dyn_mact(m1,m2,a1,a2,ct1,ct2,z,mmin1,mmax1,alpha1,delta1,beta1,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,sigma_t1,\
        mmin2,mmax2,alpha2,delta2,beta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,sigma_t2,r2,mu_a,sigma_a,gamma=2.7,zmin1=None):
    pm12=PLS_G1(m1,m2,alpha2,mmin2,mmax2,delta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,beta2)
    pact12=TG(1,sigma_t2,-1,1).prob(ct1)*TG(1,sigma_t2,-1,1).prob(ct2)*TG(mu_a,sigma_a,0,1).prob(a1)*TG(mu_a,sigma_a,0,1).prob(a2)
    return pm12*pact12*llh_z(z,gamma)*r2
   
def Calculate_Prob(post,n_catalogs):
    #samples, evidences, events, hp_likelihood, postdata = read_data(snr_cut=10,ifar_cut=1,rule_out_G2=1)
    events=['GW150914_095045', 'GW151012_095443', 'GW151226_033853', 'GW170104_101158', 'GW170608_020116', 'GW170809_082821', 'GW170814_103043', 'GW170818_022509', 'GW170823_131358', 'GW190408_181802', 'GW190412_053044', 'GW190413_134308', 'GW190421_213856', 'GW190503_185404', 'GW190512_180714', 'GW190513_205428', 'GW190521_074359', 'GW190527_092055', 'GW190630_185205', 'GW190707_093326', 'GW190708_232457', 'GW190720_000836', 'GW190727_060333', 'GW190728_064510', 'GW190803_022701', 'GW190828_063405', 'GW190828_065509', 'GW190910_112807', 'GW190915_235702', 'GW190924_021846', 'GW190925_232845', 'GW190930_133541', 'GW190413_052954', 'GW190719_215514', 'GW190725_174728', 'GW190731_140936', 'GW191105_143521', 'GW191127_050227', 'GW191129_134029', 'GW191204_171526', 'GW191215_223052', 'GW191216_213338', 'GW191222_033537', 'GW200112_155838', 'GW200128_022011', 'GW200129_065458', 'GW200202_154313', 'GW200208_130117', 'GW200209_085452', 'GW200219_094415', 'GW200224_222234', 'GW200225_060421', 'GW200302_015811', 'GW200311_115853', 'GW200316_215756', 'GW191103_012549', 'GW200216_220804']
    with open('./data/GWTC3_BBH_Mixed5000_ruled_2G.pickle', 'rb') as fp:
        samples, ln_evidences = pickle.load(fp)
    hp_likelihood = HyperparameterLikelihood(posteriors=samples, hyper_prior=hyper_Two_pop, log_evidences=ln_evidences, max_samples=2000)

    postdata={'m1':hp_likelihood.data['m1'],'m2':hp_likelihood.data['m2'],'a1':hp_likelihood.data['a1'],'a2':hp_likelihood.data['a2'],'ct1':hp_likelihood.data['cos_tilt_1'],'ct2':hp_likelihood.data['cos_tilt_2'],'z':hp_likelihood.data['z']}
    indx=np.random.choice(np.arange(len(post['log_likelihood'])),size=n_catalogs)
    parameters={key:np.array(post[key])[indx] for key in post.keys()}
    parameters.pop('log_likelihood')
    parameters.pop('log_prior')
    parameters.pop('lgR0')
    Z=[]
    for i in tqdm(range(n_catalogs)):
        # Select a random population sample
        para={key:parameters[key][i] for key in parameters.keys()}
        F=np.average(Field_mact(**postdata,**para),axis=1)
        D=np.average(Dyn_mact(**postdata,**para),axis=1)
        FD=F+D
        Z.append(np.array([F/FD,D/FD]))
    
    Z=np.array(Z).T

    Z05=np.percentile(Z,5,axis=2)
    Z50=np.percentile(Z,50,axis=2)
    Z95=np.percentile(Z,95,axis=2)
    Zmean=np.average(Z,axis=2)
    zeroindex=np.where(Z95>1e-2)
    print('events','Aligned','Isotropic')
    for n,i in enumerate(events):
        print('{}\\{}&{}&{} \\\\'.format(i[:8],i[8:],round(Zmean[n][0],3),round(Zmean[n][1],3)))
    return Zmean[:,0]


def Field_m(m1,m2,mmin1,mmax1,alpha1,delta1,beta1,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,sigma_t1,\
        mmin2,mmax2,alpha2,delta2,beta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,sigma_t2,r2,mu_a,sigma_a,zmin1):
    pm12=PLS_G1(m1,m2,alpha1,mmin1,mmax1,delta1,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,beta1)
    return pm12

####################################################################
#plot corners
####################################################################
import corner
from matplotlib import lines as mpllines
def plot_corner(post,params,show_keys,color,filename,smooth=1.):
    print('ploting')
    data2=np.array([np.array(post[key]) for key in params])
    levels = (1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.))
    c1=color
    ranges=[[min(data2[i]),max(data2[i])] for i in np.arange(len(data2))]
    percentiles=[[np.percentile(data2[i],5),np.percentile(data2[i],50),np.percentile(data2[i],95)] for i in np.arange(len(data2))]
    titles=[r"${:.2f}_{{-{:.2f}}}^{{+{:.2f}}}$".format(percentiles[i][1],percentiles[i][1]-percentiles[i][0], percentiles[i][2]-percentiles[i][1]) for i in np.arange(len(data2)) ]
    kwargs = dict(title_kwargs=dict(fontsize=15), labels=show_keys, smooth=smooth, bins=25,  quantiles=[0.05,0.5,0.95], range=ranges,\
    levels=levels, show_titles=True, titles=None, plot_density=False, plot_datapoints=True, fill_contours=True, title_qs=[0.05,0.95],\
    label_kwargs=dict(fontsize=20), max_n_ticks=1, alpha=0.5, hist_kwargs=dict(color=c1))
    groupdata=[data2]
    plt.cla()
    fig = corner.corner(groupdata[0].T, color=c1, **kwargs)
    lines = [mpllines.Line2D([0], [0], color=c1)]
    axes = fig.get_axes()
    ndim = int(np.sqrt(len(axes)))
    #axes[ndim - 1].legend(lines, labels, fontsize=14)
    plt.savefig(filename)
    
GWTC3=['GW150914_095045', 'GW151012_095443', 'GW151226_033853', 'GW170104_101158', 'GW170608_020116', 'GW170729_185629', 'GW170809_082821', 'GW170814_103043', 'GW170818_022509', 'GW170823_131358', 'GW190408_181802', 'GW190412_053044', 'GW190413_134308', 'GW190421_213856', 'GW190503_185404', 'GW190512_180714', 'GW190513_205428', 'GW190517_055101', 'GW190519_153544', 'GW190521_030229', 'GW190521_074359', 'GW190527_092055', 'GW190602_175927', 'GW190620_030421', 'GW190630_185205', 'GW190701_203306', 'GW190706_222641', 'GW190707_093326', 'GW190708_232457', 'GW190720_000836', 'GW190727_060333', 'GW190728_064510', 'GW190803_022701', 'GW190828_063405', 'GW190828_065509', 'GW190910_112807', 'GW190915_235702', 'GW190924_021846', 'GW190925_232845', 'GW190929_012149', 'GW190930_133541', 'GW190413_052954', 'GW190719_215514', 'GW190725_174728', 'GW190731_140936', 'GW190805_211137', 'GW191105_143521', 'GW191109_010717', 'GW191127_050227', 'GW191129_134029', 'GW191204_171526', 'GW191215_223052', 'GW191216_213338', 'GW191222_033537', 'GW191230_180458', 'GW200112_155838', 'GW200128_022011', 'GW200129_065458', 'GW200202_154313', 'GW200208_130117', 'GW200209_085452', 'GW200219_094415', 'GW200224_222234', 'GW200225_060421', 'GW200302_015811', 'GW200311_115853', 'GW200316_215756', 'GW191103_012549', 'GW200216_220804']

G2_events=['GW170729_185629','GW190517_055101','GW190519_153544','GW190521_030229',
'GW190602_175927','GW190620_030421','GW190701_203306','GW190706_222641',
'GW190929_012149','GW190805_211137','GW191109_010717','GW191230_180458']
 
def PLS_dyn_G1G2(m1,m2,alpha3,mmin3,mmax3,o1,o2,o3,o4,o5,o6,o7,r3,\
        mmin2,mmax2,alpha2,delta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,beta2):
    m1_sam = np.linspace(mmin2,mmax3,500)
    m2_sam = np.linspace(mmin2,mmax3,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = PLS_dyn_mass_un(x,y,alpha3,mmin3,mmax3,o1,o2,o3,o4,o5,o6,o7,r3,\
        mmin2,mmax2,alpha2,delta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,beta2)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    p1 = PLS_dyn_G1_m(m1,alpha2,mmin2,mmax2,delta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10)*(1-r3)*PLS_dyn_m(m2,alpha3,mmin3,mmax3,o1,o2,o3,o4,o5,o6,o7,r3,\
        mmin2,mmax2,alpha2,delta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10)*(m2/m1)**beta2*(mmin2<m2)*(m2<m1)*(m1<mmax3)/AMP1
    p2 =PLS_G2_m(m1,alpha3,mmin3,mmax3,o1,o2,o3,o4,o5,o6,o7)*r3*PLS_dyn_m(m2,alpha3,mmin3,mmax3,o1,o2,o3,o4,o5,o6,o7,r3,\
        mmin2,mmax2,alpha2,delta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10)*(m2/m1)**beta2*(mmin2<m2)*(m2<m1)*(m1<mmax3)/AMP1
    return p1,p2

def PLS_dyn_G1G2_m(m1,alpha3,mmin3,mmax3,o1,o2,o3,o4,o5,o6,o7,r3,\
        mmin2,mmax2,alpha2,delta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10):
    p1 = PLS_dyn_G1_m(m1,alpha2,mmin2,mmax2,delta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10)*(1-r3)
    p2 = PLS_G2_m(m1,alpha3,mmin3,mmax3,o1,o2,o3,o4,o5,o6,o7)*r3
    return p1,p2

def Dyn_plot_mass_dist(post,colors,filename,Nsample):

    fig=plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot()
    ax4 = fig.add_axes([0.60,0.60,0.30,0.30])
    
    ######mass
    parameters={}
    #low_idx=np.where(np.array(post['r2'])<0.25)
    indx=np.random.choice(np.arange(len(post['log_likelihood'])),size=Nsample)
    keys=['alpha2','mmin2','mmax2','delta2','mu_a','sigma_a','mu_a3','sigma_a3','r3','alpha3','mmin3','mmax3','r3']
    keys2=['alpha3','mmin3','mmax3','alpha2','mmin2','mmax2','delta2','r3']
    for i in np.arange(10):
        keys.append('q'+str(i+1))
        keys2.append('q'+str(i+1))
    for i in np.arange(7):
        keys.append('o'+str(i+1))
        keys2.append('o'+str(i+1))

    for key in keys:
        parameters.update({key:np.array(post[key])[indx]})
    if 'mu_t1' in post.keys():
        parameters.update({'mu_t1':np.array(post['mu_t1'])[indx]})
    else:
        parameters.update({'mu_t1':np.ones(Nsample)})
    
    dyn=[]
    dyn_2G=[]
    spin_a=[]
    spin_a_2G=[]
    a_sam=np.linspace(0,1,100)
    ct_sam=np.linspace(-1,1,100)
    m1_sam=np.linspace(2,100,500)
    m2_sam=np.linspace(2,100,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    xx=np.linspace(2,100,500)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    
    for i in tqdm(np.arange(Nsample)):
        para2={key:parameters[key][i] for key in keys2}
        p1,p2 = PLS_dyn_G1G2_m(m1_sam,**para2)
        dyn.append(p1)
        dyn_2G.append(p2)
        spin_a.append(TG(parameters['mu_a'][i],parameters['sigma_a'][i],0,1).prob(a_sam))
        spin_a_2G.append(TG(parameters['mu_a3'][i],parameters['sigma_a3'][i],0,1).prob(a_sam))

    spin_a_2G=np.array(spin_a_2G)
    a_2G_pup=np.percentile(spin_a_2G,95,axis=0)
    a_2G_plow=np.percentile(spin_a_2G,5,axis=0)
    a_2G_pmid=np.percentile(spin_a_2G,50,axis=0)
    
    spin_a=np.array(spin_a)
    a_pup=np.percentile(spin_a,95,axis=0)
    a_plow=np.percentile(spin_a,5,axis=0)
    a_pmid=np.percentile(spin_a,50,axis=0)

    ax4.xaxis.grid(True,which='major',ls=':',color='grey')
    ax4.yaxis.grid(True,which='major',ls=':',color='grey')
    ax4.fill_between(a_sam,a_plow,a_pup,color=colors[1],alpha=0.3,label='first generation')
    ax4.plot(a_sam,a_plow,color=colors[1],alpha=0.8,lw=0.5)
    ax4.plot(a_sam,a_pup,color=colors[1],alpha=0.8,lw=0.5)
    ax4.plot(a_sam,a_pmid,color=colors[1],alpha=0.8)
    ax4.fill_between(a_sam,a_2G_plow,a_2G_pup,color=colors[3],alpha=0.3,label='higher generation')
    ax4.plot(a_sam,a_2G_plow,color=colors[3],alpha=0.8,lw=0.5)
    ax4.plot(a_sam,a_2G_pup,color=colors[3],alpha=0.8,lw=0.5)
    ax4.plot(a_sam,a_2G_pmid,color=colors[3],alpha=0.8)

    ax4.set_xlim(0,1)
    ax4.set_xlabel(r'$a$')
    ax4.set_ylabel(r'$p(a)$')
    #ax4.legend(loc='upper right')
    
    dyn=np.array(dyn)
    dyn_pup=np.percentile(dyn,95,axis=0)
    dyn_plow=np.percentile(dyn,5,axis=0)
    dyn_pmid=np.percentile(dyn,50,axis=0)
    dyn_pmean=np.mean(dyn,axis=0)
    dyn_2G=np.array(dyn_2G)
    dyn_2G_pup=np.percentile(dyn_2G,95,axis=0)
    dyn_2G_plow=np.percentile(dyn_2G,5,axis=0)
    dyn_2G_pmid=np.percentile(dyn_2G,50,axis=0)
    dyn_2G_pmean=np.mean(dyn_2G,axis=0)

    print('ploting')
    ax1.xaxis.grid(True,which='major',ls=':',color='grey')
    ax1.yaxis.grid(True,which='major',ls=':',color='grey')
    ax1.fill_between(m1_sam,dyn_2G_plow,dyn_2G_pup,color=colors[3],alpha=0.3,label='higher generation')
    ax1.plot(m1_sam,dyn_2G_plow,color=colors[3],alpha=0.8,lw=0.5)
    ax1.plot(m1_sam,dyn_2G_pup,color=colors[3],alpha=0.8,lw=0.5)
    ax1.plot(m1_sam,dyn_2G_pmid,color=colors[3],alpha=0.9)
    ax1.fill_between(m1_sam,dyn_plow,dyn_pup,color=colors[1],alpha=0.3,label='first generation')
    ax1.plot(m1_sam,dyn_plow,color=colors[1],alpha=0.8,lw=0.5)
    ax1.plot(m1_sam,dyn_pup,color=colors[1],alpha=0.8,lw=0.5)
    ax1.plot(m1_sam,dyn_pmid,color=colors[1],alpha=0.9)
    
    ax1.set_yscale('log')
    ax1.set_ylim(1e-4,1e0)
    ax1.set_xlim(0,100)
    ax1.set_xlabel(r'$m/M_{\odot}$')
    ax1.set_ylabel(r'$p_{\rm Dyn}(m)$')
    ax1.set_title('The dynamical BHs')
    ax1.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(filename)


def cal_BBH_quantiles(post,values,Nsample=None):
    quants=[]
    keys=['alpha1','mmin1','mmax1','delta1']
    for i in np.arange(10):
        keys.append('N'+str(i+1))
    keys.append('beta1')
    m1_sam=np.linspace(2,100,400)
    m2_sam=np.linspace(2,100,299)
    x,y = np.meshgrid(m1_sam,m2_sam)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    if Nsample==None:
        Nsample=len(post['mmin1'])
    for i in tqdm(np.arange(Nsample)):
        para=[post[key][i] for key in keys]
        pdf=np.sum(PLS_G1_un(x,y,*para),axis=0)
        cdf=np.cumsum(pdf)/np.sum(pdf)
        f=interp1d(cdf,m1_sam)
        quants.append(f(values))
    quants=np.array(quants).T
    return quants
