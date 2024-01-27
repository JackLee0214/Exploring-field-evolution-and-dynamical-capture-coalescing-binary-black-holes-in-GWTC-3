import bilby
from bilby.core.sampler import run_sampler
import numpy as np
import pickle
from pandas.core.frame import DataFrame
import h5py
from astropy.cosmology import Planck15
from bilby.hyper.likelihood import HyperparameterLikelihood
from bilby.core.result import read_in_result as rr
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


##########################
#un_normed mass model bank
#########################
def smooth(m,mmin,delta):
    A = (m-mmin == 0.)*1e-10 + (m-mmin)
    B = (m-mmin-delta == 0.)*1e-10 + abs(m-mmin-delta)
    f_m_delta = delta/A - delta/B
    return (np.exp(f_m_delta) + 1.)**(-1.)*(m<=(mmin+delta))*(m>mmin)+1.*(m>(mmin+delta))

def PL_un(m1,mmin,mmax,alpha,delta):
    norm=(mmax**(1-alpha)-mmin**(1-alpha))/(1-alpha)
    pdf = m1**(-alpha)/norm*smooth(m1,mmin,delta)
    return np.where((mmin<m1) & (m1<mmax), pdf , 1e-100)

def PL(m1,mmin2,mmax2,alpha2,delta2):
    xx=np.linspace(2,100,1000)
    yy=PL_un(xx,mmin2,mmax2,alpha2,delta2)
    norm=np.sum(yy)*98./1000.
    return PL_un(m1,mmin2,mmax2,alpha2,delta2)/norm
    
##########################
#mass
##########################

def PLS_mass(m1,alpha,mmin,mmax,delta,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10):
    xi=np.exp(np.linspace(np.log(6),np.log(60),10))
    yi=np.array([N1,N2,N3,N4,N5,N6,N7,N8,N9,N10])
    cs = CubicSpline(xi,yi,bc_type='natural')
    m1_sam = np.linspace(mmin,mmax,500)
    dx = m1_sam[1]-m1_sam[0]
    norm = np.sum(np.exp(cs(m1_sam)*(m1_sam>6)*(m1_sam<60))*PL_un(m1_sam,mmin,mmax,alpha,delta))*dx
    pm1 = np.exp(cs(m1)*(m1>6)*(m1<60))*PL_un(m1,mmin,mmax,alpha,delta)/norm
    return np.where((mmin<m1) & (m1<mmax), pm1 , 1e-100)

def PLS_G1_un(m1,m2,alpha,mmin,mmax,delta,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,beta):
    xi=np.exp(np.linspace(np.log(6),np.log(60),10))
    yi=np.array([N1,N2,N3,N4,N5,N6,N7,N8,N9,N10])
    cs = CubicSpline(xi,yi,bc_type='natural')
    pm1 = np.exp(cs(m1)*(m1>6)*(m1<60))*PL_un(m1,mmin,mmax,alpha,delta)
    pm2 = np.exp(cs(m2)*(m2>6)*(m2<60))*PL_un(m2,mmin,mmax,alpha,delta)
    pq = (m2/m1)**beta
    return np.where((mmin<m2) & (m2<m1) & (m1<mmax), pm1*pm2*pq , 1e-100)

def PLS_G1(m1,m2,alpha,mmin,mmax,delta,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,beta):
    m1_sam = np.linspace(mmin,mmax,500)
    m2_sam = np.linspace(mmin,mmax,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = PLS_G1_un(x,y,alpha,mmin,mmax,delta,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,beta)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = PLS_G1_un(m1,m2,alpha,mmin,mmax,delta,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,beta)/AMP1
    return np.where((mmin<m2) & (m2<m1) & (m1<mmax), pdf , 1e-100)

##########################
#spin
##########################

def spin_a(a1,mu_a,sigma_a):
    return TG(mu_a,sigma_a,0,1).prob(a1)
   
def DF_ct12(ct1,ct2,sigma_t,zeta,zmin=-1):
    align=TG(1,sigma_t,zmin,1)
    iso=Uniform(-1,1)
    return align.prob(ct1)*align.prob(ct2)*zeta+iso.prob(ct1)*iso.prob(ct2)*(1-zeta)

def Aligned_ct12(ct1,ct2,sigma_t,zmin=-1,mu_t=1):
    align=TG(mu_t,sigma_t,zmin,1)
    return align.prob(ct1)*align.prob(ct2)

##########################
#mass vs spin
##########################
#PLS default
    
def PLS_default(m1,m2,a1,a2,ct1,ct2,alpha,mmin,mmax,delta,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,beta,mu_a,sigma_a,sigma_t,zeta,zmin):
    pact = DF_ct12(ct1,ct2,sigma_t,zeta,zmin)*spin_a(a1,mu_a,sigma_a)*spin_a(a2,mu_a,sigma_a)
    hp = PLS_G1(m1,m2,alpha,mmin,mmax,delta,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,beta)*pact
    return hp
    
def PLS_default_nospin(m1,m2,a1,a2,ct1,ct2,alpha,mmin,mmax,delta,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,beta,mu_a,sigma_a,sigma_t,zeta,zmin):
    pact = 1./4.
    hp = PLS_G1(m1,m2,alpha,mmin,mmax,delta,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,beta)*pact
    return hp
    
#two pop
def Two_pop(m1,m2,a1,a2,ct1,ct2,mmin1,mmax1,alpha1,delta1,beta1,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,sigma_t1,mmin2,mmax2,alpha2,delta2,beta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,r2,mu_a,sigma_a,zmin1):
    p1=PLS_G1(m1,m2,alpha1,mmin1,mmax1,delta1,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,beta1)*Aligned_ct12(ct1,ct2,sigma_t1,zmin=zmin1)*(1-r2)
    p2=PLS_G1(m1,m2,alpha2,mmin2,mmax2,delta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,beta2)*Uniform(-1,1).prob(ct1)*Uniform(-1,1).prob(ct2)*r2
    p=p1+p2
    pa=spin_a(a1,mu_a,sigma_a)*spin_a(a2,mu_a,sigma_a)
    pdf = p*pa
    return pdf

def Two_pop_nospin(m1,m2,a1,a2,ct1,ct2,mmin1,mmax1,alpha1,delta1,beta1,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,sigma_t1,mmin2,mmax2,alpha2,delta2,beta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,r2,mu_a,sigma_a,zmin1):
    p1=PLS_G1(m1,m2,alpha1,mmin1,mmax1,delta1,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,beta1)*(1-r2)
    p2=PLS_G1(m1,m2,alpha2,mmin2,mmax2,delta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,beta2)*r2
    p=p1+p2
    pact = 1./4.
    pdf = p*pact
    return pdf


##########################
#with 2G BBHs
##########################

def PLS_G2_m_un(m1,alpha3,mmin3,mmax3,n1,n2,n3,n4,n5,n6,n7):
    xi=np.exp(np.linspace(np.log(20),np.log(80),7))
    yi=np.array([n1,n2,n3,n4,n5,n6,n7])
    cs = CubicSpline(xi,yi,bc_type='natural')
    pm1 = np.exp(cs(m1)*(m1>20)*(m1<80))*PL_un(m1,mmin3,mmax3,alpha3,0)
    return np.where((mmin3<m1) & (m1<mmax3), pm1 , 1e-100)

def PLS_G2_m(m1,alpha3,mmin3,mmax3,n1,n2,n3,n4,n5,n6,n7):
    m1_sam = np.linspace(mmin3,mmax3,500)
    pgrid1 = PLS_G2_m_un(m1_sam,alpha3,mmin3,mmax3,n1,n2,n3,n4,n5,n6,n7)
    dx = m1_sam[1]-m1_sam[0]
    AMP1 = np.sum(pgrid1*dx)
    pdf = PLS_G2_m_un(m1,alpha3,mmin3,mmax3,n1,n2,n3,n4,n5,n6,n7)/AMP1
    return np.where((mmin3<m1) & (m1<mmax3), pdf , 1e-100)

def PLS_dyn_G1_m_un(m1,alpha,mmin,mmax,delta,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10):
    xi=np.exp(np.linspace(np.log(6),np.log(60),10))
    yi=np.array([N1,N2,N3,N4,N5,N6,N7,N8,N9,N10])
    cs = CubicSpline(xi,yi,bc_type='natural')
    pm1 = np.exp(cs(m1)*(m1>6)*(m1<60))*PL_un(m1,mmin,mmax,alpha,delta)
    return np.where((mmin<m1) & (m1<mmax), pm1 , 1e-100)

def PLS_dyn_G1_m(m1,alpha,mmin,mmax,delta,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10):
    m1_sam = np.linspace(mmin,mmax,500)
    pgrid1 = PLS_dyn_G1_m_un(m1_sam,alpha,mmin,mmax,delta,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10)
    dx = m1_sam[1]-m1_sam[0]
    AMP1 = np.sum(pgrid1*dx)
    pdf = PLS_dyn_G1_m_un(m1,alpha,mmin,mmax,delta,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10)/AMP1
    return np.where((mmin<m1) & (m1<mmax), pdf , 1e-100)

def PLS_dyn_m(m1,alpha3,mmin3,mmax3,o1,o2,o3,o4,o5,o6,o7,r3,\
        mmin2,mmax2,alpha2,delta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10):
    p1=PLS_dyn_G1_m(m1,alpha2,mmin2,mmax2,delta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10)*(1-r3)
    p2=PLS_G2_m(m1,alpha3,mmin3,mmax3,o1,o2,o3,o4,o5,o6,o7)*r3
    return p1+p2

def PLS_dyn_ma(m1,a1,mmin2,mmax2,alpha2,delta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,mu_a,sigma_a,zmin1,alpha3,mmin3,mmax3,o1,o2,o3,o4,o5,o6,o7,mu_a3,sigma_a3,r3):
    p1=PLS_dyn_G1_m(m1,alpha2,mmin2,mmax2,delta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10)*(1-r3)*TG(mu_a,sigma_a,0,1).prob(a1)
    p2=PLS_G2_m(m1,alpha3,mmin3,mmax3,o1,o2,o3,o4,o5,o6,o7)*r3*TG(mu_a3,sigma_a3,0,1).prob(a1)
    return p1+p2
    
def PLS_dyn_mass_un(m1,m2,alpha3,mmin3,mmax3,o1,o2,o3,o4,o5,o6,o7,r3,\
        mmin2,mmax2,alpha2,delta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,beta2):
    pm1=PLS_dyn_m(m1,alpha3,mmin3,mmax3,o1,o2,o3,o4,o5,o6,o7,r3,\
        mmin2,mmax2,alpha2,delta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10)
    pm2=PLS_dyn_m(m2,alpha3,mmin3,mmax3,o1,o2,o3,o4,o5,o6,o7,r3,\
        mmin2,mmax2,alpha2,delta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10)
    pq=(m2/m1)**beta2
    return np.where((mmin2<m2) & (m2<m1) & (m1<mmax3), pm1*pm2*pq , 1e-100)

def PLS_dyn_mass(m1,m2,alpha3,mmin3,mmax3,o1,o2,o3,o4,o5,o6,o7,r3,\
        mmin2,mmax2,alpha2,delta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,beta2):
    m1_sam = np.linspace(mmin2,mmax3,500)
    m2_sam = np.linspace(mmin2,mmax3,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = PLS_dyn_mass_un(x,y,alpha3,mmin3,mmax3,o1,o2,o3,o4,o5,o6,o7,r3,\
        mmin2,mmax2,alpha2,delta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,beta2)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = PLS_dyn_mass_un(m1,m2,alpha3,mmin3,mmax3,o1,o2,o3,o4,o5,o6,o7,r3,\
        mmin2,mmax2,alpha2,delta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,beta2)/AMP1
    return np.where((mmin2<m2) & (m2<m1) & (m1<mmax3), pdf , 1e-100)
    
def Three_pop_mass(m1,m2,mmin1,mmax1,alpha1,delta1,beta1,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,sigma_t1,mmin2,mmax2,alpha2,delta2,beta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,r2,mu_a,sigma_a,zmin1,alpha3,mmin3,mmax3,o1,o2,o3,o4,o5,o6,o7,mu_a3,sigma_a3,sigma_t3,zeta3,r3):
    p1=PLS_G1(m1,m2,alpha1,mmin1,mmax1,delta1,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,beta1)*(1-r2)
    p2=PLS_dyn_mass(m1,m2,alpha3,mmin3,mmax3,o1,o2,o3,o4,o5,o6,o7,r3,\
            mmin2,mmax2,alpha2,delta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,beta2)*r2
    pdf=p2+p1
    return pdf

def PLS_dyn_mact(m1,a1,ct1,mmin2,mmax2,alpha2,delta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,mu_a,sigma_a,zmin1,alpha3,mmin3,mmax3,o1,o2,o3,o4,o5,o6,o7,mu_a3,sigma_a3,sigma_t3,zeta3,r3,k):
    p1=PLS_dyn_G1_m(m1,alpha2,mmin2,mmax2,delta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10)*(1-r3)*TG(mu_a,sigma_a,0,1).prob(a1)*(Uniform(-1,1).prob(ct1)*(1-zeta3*r3*k)+TG(1,sigma_t3,-1,1).prob(ct1)*zeta3*r3*k)
    p2=PLS_G2_m(m1,alpha3,mmin3,mmax3,o1,o2,o3,o4,o5,o6,o7)*r3*TG(mu_a3,sigma_a3,0,1).prob(a1)*(Uniform(-1,1).prob(ct1)*(1-zeta3)+TG(1,sigma_t3,-1,1).prob(ct1)*zeta3)
    return p1+p2
    
def PLS_dyn(m1,m2,a1,a2,ct1,ct2,mmin2,mmax2,alpha2,delta2,beta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,mu_a,sigma_a,zmin1,alpha3,mmin3,mmax3,o1,o2,o3,o4,o5,o6,o7,mu_a3,sigma_a3,sigma_t3,zeta3,r3,k):
    pmact1=PLS_dyn_mact(m1,a1,ct1,mmin2,mmax2,alpha2,delta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,mu_a,sigma_a,zmin1,alpha3,mmin3,mmax3,o1,o2,o3,o4,o5,o6,o7,mu_a3,sigma_a3,sigma_t3,zeta3,r3,k)
    pmact2=PLS_dyn_mact(m2,a2,ct2,mmin2,mmax2,alpha2,delta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,mu_a,sigma_a,zmin1,alpha3,mmin3,mmax3,o1,o2,o3,o4,o5,o6,o7,mu_a3,sigma_a3,sigma_t3,zeta3,r3,k)
    pq=(m2/m1)**beta2
    m1_sam = np.linspace(mmin2,mmax3,500)
    m2_sam = np.linspace(mmin2,mmax3,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = PLS_dyn_mass_un(x,y,alpha3,mmin3,mmax3,o1,o2,o3,o4,o5,o6,o7,r3,\
        mmin2,mmax2,alpha2,delta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,beta2)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    return np.where((mmin2<m2) & (m2<m1) & (m1<mmax3), pmact1*pmact2*pq/AMP1 , 1e-100)

def Three_pop(m1,m2,a1,a2,ct1,ct2,mmin1,mmax1,alpha1,delta1,beta1,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,sigma_t1,mmin2,mmax2,alpha2,delta2,beta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,r2,mu_a,sigma_a,zmin1,alpha3,mmin3,mmax3,o1,o2,o3,o4,o5,o6,o7,mu_a3,sigma_a3,sigma_t3,zeta3,r3,k):
    p1=PLS_G1(m1,m2,alpha1,mmin1,mmax1,delta1,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,beta1)*\
        TG(mu_a,sigma_a,0,1).prob(a1)*TG(mu_a,sigma_a,0,1).prob(a2)*TG(1,sigma_t1,zmin1,1).prob(ct1)*TG(1,sigma_t1,zmin1,1).prob(ct2)*(1-r2)
    p2=PLS_dyn(m1,m2,a1,a2,ct1,ct2,mmin2,mmax2,alpha2,delta2,beta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,\
        mu_a,sigma_a,zmin1,alpha3,mmin3,mmax3,o1,o2,o3,o4,o5,o6,o7,mu_a3,sigma_a3,sigma_t3,zeta3,r3,k)*r2
    pdf=p2+p1
    return pdf
      
def Three_pop_nospin(m1,m2,a1,a2,ct1,ct2,mmin1,mmax1,alpha1,delta1,beta1,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,sigma_t1,mmin2,mmax2,alpha2,delta2,beta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,r2,mu_a,sigma_a,zmin1,alpha3,mmin3,mmax3,o1,o2,o3,o4,o5,o6,o7,mu_a3,sigma_a3,sigma_t3,zeta3,r3,k):
    pdf=Three_pop_mass(m1,m2,mmin1,mmax1,alpha1,delta1,beta1,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,sigma_t1,mmin2,mmax2,alpha2,delta2,beta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,r2,mu_a,sigma_a,zmin1,alpha3,mmin3,mmax3,o1,o2,o3,o4,o5,o6,o7,mu_a3,sigma_a3,sigma_t3,zeta3,r3)*1./4.
    return pdf


################################################################################################
#redshift
#################################################################################################
fdVcdz=interp1d(np.linspace(0,5,10000),4*np.pi*Planck15.differential_comoving_volume(np.linspace(0,5,10000)).to(u.Gpc**3/u.sr).value)

zs=np.linspace(0,1.9,1000)
dVdzs=fdVcdz(zs)
logdVdzs=np.log(dVdzs)
def llh_z(z,gamma):
    norm=np.sum((1+zs)**(gamma-1)*dVdzs)*1.9/1000.
    norm0=np.sum((1+zs)**(-1)*dVdzs)*1.9/1000.
    return np.where((z>0) & (z<1.9), (1+z)**gamma/norm*norm0 , 1e-100)

def p_z(z,gamma):
    norm=np.sum((1+zs)**(gamma-1)*dVdzs)*1.9/1000.
    p = (1+z)**(gamma-1)*fdVcdz(z)/norm
    return np.where((z>0) & (z<1.9), p , 1e-100)

def log_N(T,lgR0,gamma):
    return np.log(T) + lgR0/np.log10(np.e) + np.logaddexp.reduce((gamma-1)*np.log(zs+1) + logdVdzs) + np.log(1.9/1000)


#################################################################################################
#hyper prior
#################################################################################################
  
def hyper_Two_pop(dataset,mmin1,mmax1,alpha1,delta1,beta1,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,\
        sigma_t1,mmin2,mmax2,alpha2,delta2,beta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,r2,mu_a,sigma_a,zmin1,gamma):
    z = dataset['z']
    a1,a2,ct1,ct2 = dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1,m2 = dataset['m1'], dataset['m2']
    hp = Two_pop(m1,m2,a1,a2,ct1,ct2,mmin1,mmax1,alpha1,delta1,beta1,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,\
        sigma_t1,mmin2,mmax2,alpha2,delta2,beta2,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,r2,mu_a,sigma_a,zmin1)*llh_z(z,gamma)
    return hp
    
def hyper_PLS_default(dataset,alpha,mmin,mmax,delta,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,beta,mu_a,sigma_a,sigma_t,zeta,zmin,gamma):
    z = dataset['z']
    a1,a2,ct1,ct2 = dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1,m2 = dataset['m1'], dataset['m2']
    pact = DF_ct12(ct1,ct2,sigma_t,zeta,zmin)*spin_a(a1,mu_a,sigma_a)*spin_a(a2,mu_a,sigma_a)
    hp = PLS_G1(m1,m2,alpha,mmin,mmax,delta,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,beta)*llh_z(z,gamma)*pact
    return hp

 
def hyper_Three_pop(dataset,mmin1,mmax1,alpha1,delta1,beta1,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,sigma_t1,mmin2,mmax2,alpha2,delta2,beta2,\
        q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,r2,mu_a,sigma_a,zmin1,alpha3,mmin3,mmax3,o1,o2,o3,o4,o5,o6,o7,mu_a3,sigma_a3,sigma_t3,zeta3,r3,k,gamma):
    z = dataset['z']
    a1,a2,ct1,ct2 = dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1,m2 = dataset['m1'], dataset['m2']
    hp = Three_pop(m1,m2,a1,a2,ct1,ct2,mmin1,mmax1,alpha1,delta1,beta1,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,sigma_t1,mmin2,mmax2,alpha2,delta2,beta2,\
        q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,r2,mu_a,sigma_a,zmin1,alpha3,mmin3,mmax3,o1,o2,o3,o4,o5,o6,o7,mu_a3,sigma_a3,sigma_t3,zeta3,r3,k)*llh_z(z,gamma)
    return hp
########################
#priors
########################

def Two_pop_priors():
    priors=bilby.prior.PriorDict()
    priors.update(dict(lgR0 = Uniform(0,3),
            gamma=2.7,
            mu_a = Uniform(0., 1., 'mu_a', '$\\mu_{\\rm a}$'),
            sigma_a = Uniform(0.05, 0.5, 'sigma_a', '$\\sigma_{\\rm a}$'),
            beta1 = Uniform(-8,8., 'beta_1', '$\\beta_{\\rm A}$'),
            mmin1 = Uniform(2., 10., 'mmin1', '$m_{\\rm min,A}$'),
            mmax1 = Uniform(20., 100, 'mmax1', '$m_{\\rm max,A}$'),
            alpha1 = Uniform(-8, 8., 'alpha1', '$\\alpha_{\\rm A}$'),
            delta1 = Uniform(0., 10., 'delta1', '$\\delta_{\\rm m,A}$'),
            sigma_t1 = Uniform(0.1,4, 'sigma_t1', '$\\sigma_{\\rm t,A}$'),
            beta2 = Uniform(-8.,8., 'beta_2', '$\\beta_{\\rm I}$'),
            mmin2 = Uniform(2., 10., 'mmin2', '$m_{\\rm min,I}$'),
            mmax2 = Uniform(20., 100, 'mmax2', '$m_{\\rm max,I}$'),
            alpha2 = Uniform(-8, 8., 'alpha2', '$\\alpha_{\\rm I}$'),
            delta2 = Uniform(0., 10., 'delta2', '$\\delta_{\\rm m,I}$'),
            r2 = Uniform(0,1,'r2','$r_{\\rm I}$'),
            zmin1=-1
            ))
    priors.update({'N'+str(i+1): TG(0,1,-100,100,name='N'+str(i+1))  for i in np.arange(10)})
    priors.update({'N1':0,'N'+str(10): 0})
    priors.update({'q'+str(i+1): TG(0,1,-100,100,name='q'+str(i+1))  for i in np.arange(10)})
    priors.update({'q1':0,'q'+str(10): 0})
    return priors


def Three_pop_priors():
    priors=bilby.prior.PriorDict()
    two_pop=Two_pop_priors()
    priors.update(two_pop)
    priors.update(dict(mu_a3 = Uniform(0.4, 1., 'mu_a3', '$\\mu_{\\rm a,3}$'),
                    sigma_a3 = Uniform(0.05, 0.5, 'sigma_a3', '$\\sigma_{\\rm a,3}$'),
                    sigma_t3 = Uniform(0.1,1),
                    zeta3 = Uniform(0,1),
                    mmin3 = Uniform(20., 50., 'mmin3', '$m_{\\rm min,3}$'),
                    mmax3 = Uniform(60., 100, 'mmax3', '$m_{\\rm max,3}$'),
                    alpha3 = Uniform(-8, 8., 'alpha3', '$\\alpha_3$'),
                    r3 = Uniform(0,1)
                    ))
    priors.update({'o'+str(i+1): TG(0,1,-100,100,name='o'+str(i+1))  for i in np.arange(7)})
    priors.update({'o1':0,'o'+str(7): 0})
    return priors
  

def Compare_priors():
    priors=bilby.prior.PriorDict()
    priors.update(dict(lgR0 = Uniform(0,3),
                        gamma=2.7,
                        beta = Uniform(-8,8.),
                        mmin = Uniform(2., 10., 'mmin', '$m_{\\rm min}$'),
                        mmax = Uniform(30., 100, 'mmax', '$m_{\\rm max}$'),
                        alpha = Uniform(-8., 8., 'alpha', '$\\alpha$'),
                        delta = Uniform(0., 10., 'delta', '$\\delta_{\\rm m}$'),
                        mu_a = Uniform(0., 1., 'mu_a', '$\\mu_{\\rm a}$'),
                        sigma_a = Uniform(0.05, 0.5, 'sigma_a', '$\\sigma_{\\rm a}$'),
                        sigma_t = Uniform(0.1, 4., 'sigma_t', '$\\sigma_{\\rm t}$'),
                        zeta = Uniform(0,1,'zeta','$\\zeta$'),
                        zmin = -1))
    priors.update({'N'+str(i+1): TG(0,1,-100,100,name='N'+str(i+1))  for i in np.arange(10)})
    priors.update({'N1':0,'N'+str(10): 0})
    return priors

def Three_pop_constraint(params):
    params['constraint']=np.sign(1-params['zeta3']*params['r3']*params['k']/(1-params['r3']))-1
    return params

def Three_pop_priors():
    priors=bilby.prior.PriorDict(conversion_function=Three_pop_constraint)
    two_pop=Two_pop_priors()
    priors.update(two_pop)
    priors.update(dict(mu_a3 = Uniform(0.4, 1., 'mu_a3', '$\\mu_{\\rm a,3}$'),
                    sigma_a3 = Uniform(0.05, 0.5, 'sigma_a3', '$\\sigma_{\\rm a,3}$'),
                    sigma_t3 = Uniform(0.1,1),
                    zeta3 = Uniform(0,1),
                    mmin3 = Uniform(20., 50., 'mmin3', '$m_{\\rm min,3}$'),
                    mmax3 = Uniform(60., 100, 'mmax3', '$m_{\\rm max,3}$'),
                    alpha3 = Uniform(-8, 8., 'alpha3', '$\\alpha_3$'),
                    r3 = Uniform(0,1),
                    k = Uniform(2.99,3),
                    constraint = bilby.prior.Constraint(minimum=-0.1, maximum=0.1)))
    priors.update({'o'+str(i+1): TG(0,1,-100,100,name='o'+str(i+1))  for i in np.arange(7)})
    priors.update({'o1':0,'o'+str(7): 0})
    return priors
    
#####################
#injection campaign

def a_from_xyz(x,y,z):
    return (x**2+y**2+z**2)**0.5
def ct_from_xyz(x,y,z):
    return z/(x**2+y**2+z**2)**0.5
def act_Uniform(a1,a2,ct1,ct2):
    pa=Uniform(0,1).prob(a1)*Uniform(0,1).prob(a1)
    pct=Uniform(-1,1).prob(ct1)*Uniform(-1,1).prob(ct2)
    return pa*pct

inject_dir='./data/'
with h5py.File(inject_dir+'o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5', 'r') as f:
    Tobs = f.attrs['analysis_time_s']/(365.25*24*3600) # years
    Ndraw = f.attrs['total_generated']
    
    m1_inj = np.array(f['injections/mass1_source'])
    m2_inj = np.array(f['injections/mass2_source'])
    z_inj = np.array(f['injections/redshift'])
    log1pz_inj = np.log1p(z_inj)
    logdVdz_inj = np.log(4*np.pi) + np.log(Planck15.differential_comoving_volume(z_inj).to(u.Gpc**3/u.sr).value)
   
    s1z_inj = np.array(f['injections/spin1z'])
    s2z_inj = np.array(f['injections/spin2z'])
    s1x_inj = np.array(f['injections/spin1x'])
    s2x_inj = np.array(f['injections/spin2x'])
    s1y_inj = np.array(f['injections/spin1y'])
    s2y_inj = np.array(f['injections/spin2y'])
    a1_inj = a_from_xyz(s1x_inj,s1y_inj,s1z_inj)
    ct1_inj = ct_from_xyz(s1x_inj,s1y_inj,s1z_inj)
    a2_inj = a_from_xyz(s2x_inj,s2y_inj,s2z_inj)
    ct2_inj = ct_from_xyz(s2x_inj,s2y_inj,s2z_inj)
    
    p_draw = np.array(f['injections/sampling_pdf'])
    
    logpdraw = np.log(p_draw)

    gstlal_ifar = np.array(f['injections/ifar_gstlal'])
    pycbc_ifar = np.array(f['injections/ifar_pycbc_hyperbank'])
    pycbc_bbh_ifar = np.array(f['injections/ifar_pycbc_bbh'])
    opt_snr = np.array(f['injections/optimal_snr_net'])
    name = np.array(f['injections/name'])

snr_thr = 10.
ifar_thr = 1.
detection_selector_O3 = (gstlal_ifar > ifar_thr) | (pycbc_ifar > ifar_thr) | (pycbc_bbh_ifar > ifar_thr)
detection_selector_O12 = (opt_snr > snr_thr)
detection_selector = np.where(name == b'o3', detection_selector_O3,detection_selector_O12)

#####################################################################
#spin pdf on xyz, which is injected
def log_SpinF():
    r1=s1x_inj**2+s1y_inj**2+s1z_inj**2
    r2=s2x_inj**2+s2y_inj**2+s2z_inj**2
    return -2*np.log(4*np.pi)-np.log(r1)-np.log(r2)
logpspin_inj=log_SpinF()
#####################################################################
#spin pdf on magnitude and cosine tilt angle, whcih is needed
logpact_inj=np.log(act_Uniform(a1_inj,a2_inj,ct1_inj,ct2_inj))
#reweight the draw pdf
logpdraw=logpdraw-logpspin_inj+logpact_inj

#This is selection effect
def Rate_selection_function_with_uncertainty(Nobs,mass_spin_model,lgR0,gamma,**kwargs):
    log_dNdz = lgR0/np.log10(np.e) + (gamma-1)*log1pz_inj + logdVdz_inj
    log_dNdmds = np.log(mass_spin_model(m1_inj,m2_inj,a1_inj,a2_inj,ct1_inj,ct2_inj,**kwargs))
    log_dNdzdmds = np.where(detection_selector, log_dNdz+log_dNdmds, np.NINF)
    log_Nexp = np.log(Tobs) + np.logaddexp.reduce(log_dNdzdmds - logpdraw) - np.log(Ndraw)
    term1 = Nobs*log_N(Tobs,lgR0,gamma)
    term2 = -np.exp(log_Nexp)
    selection=term1 + term2
    logmu=log_Nexp-log_N(Tobs,lgR0,gamma)
    varsel= np.sum(np.exp(2*(np.log(Tobs)+log_dNdzdmds - logpdraw-log_N(Tobs,lgR0,gamma)- np.log(Ndraw))))-np.exp(2*logmu)/Ndraw
    total_vars=Nobs**2 * varsel / np.exp(2*logmu)
    Neff=np.exp(2*logmu)/varsel
    return selection, total_vars, Neff

def PPC_selected(mass_spin_model,lgR0=1.2,gamma=2.7,**kwargs):
    log_dNdz = lgR0/np.log10(np.e) + (gamma-1)*log1pz_inj + logdVdz_inj
    log_dNdmds = np.log(mass_spin_model(m1_inj,m2_inj,a1_inj,a2_inj,ct1_inj,ct2_inj,**kwargs))
    log_dNdzdmds = np.where(detection_selector, log_dNdz+log_dNdmds, np.NINF)
    logpdf = log_dNdzdmds - logpdraw
    return m1_inj,m2_inj,a1_inj,a2_inj,ct1_inj,ct2_inj,np.exp(logpdf)


############
# all GWTC_1 events
GWTC1_events = ['GW150914_095045', 'GW151012_095443', 'GW151226_033853', 'GW170104_101158', 'GW170608_020116',
                'GW170729_185629',  'GW170809_082821', 'GW170814_103043', 'GW170817', 'GW170818_022509', 'GW170823_131358']


# All O3a events (FAR < 0.25/yr)
O3a_highsig = ['GW190408_181802', 'GW190412_053044', 'GW190413_134308',
                'GW190421_213856', 'GW190425',
                'GW190503_185404', 'GW190512_180714', 'GW190513_205428',
                'GW190517_055101', 'GW190519_153544', 'GW190521_030229', 'GW190521_074359',
                'GW190527_092055', 'GW190602_175927', 'GW190620_030421', 'GW190630_185205',
                'GW190701_203306', 'GW190706_222641', 'GW190707_093326', 'GW190708_232457',
                'GW190720_000836', 'GW190727_060333',
                'GW190728_064510', 'GW190803_022701', 'GW190814_211039',
                'GW190828_063405', 'GW190828_065509', 'GW190910_112807', 'GW190915_235702',
                'GW190924_021846', 'GW190925_232845', 'GW190929_012149', 'GW190930_133541']
                
# Add O3a events (0.25/yr < FAR < 1/yr)
O3a_add = ['GW190413_052954', 'GW190426_152155', 'GW190719_215514', 'GW190725_174728',
            'GW190731_140936', 'GW190805_211137', 'GW190917_114630']
# All O3b events (FAR < 0.25/yr)
O3b_highsig = ['GW191105_143521', 'GW191109_010717', 'GW191127_050227', 'GW191129_134029',
                'GW191204_171526', 'GW191215_223052', 'GW191216_213338', 'GW191222_033537',
                'GW191230_180458',  'GW200105_162426', 'GW200112_155838', 'GW200115_042309',
                'GW200128_022011', 'GW200129_065458', 'GW200202_154313', 'GW200208_130117',
                'GW200209_085452', 'GW200219_094415', 'GW200224_222234', 'GW200225_060421',
                'GW200302_015811', 'GW200311_115853', 'GW200316_215756']
               
# Add O3b events (0.25/yr < FAR < 1/yr)
O3b_add = ['GW191103_012549', 'GW200216_220804']

# special events (BNS, NSBH, BBH outlayer)
special_events = ['GW170817', 'GW190425', 'GW190426_152155', 'GW190814_211039', 'GW190917_114630', 'GW200105_162426', 'GW200115_042309']

#summary

#####GWTC2p1
           
GWTC2p1=GWTC1_events+O3a_highsig+O3a_add

####BBH_documents
doc_1 = [BBH for BBH in GWTC2p1 if BBH not in special_events]
doc_2 = [BBH for BBH in O3b_highsig+O3b_add if BBH not in special_events]

GWTC3 = doc_1+doc_2

G2_events=['GW170729_185629','GW190517_055101','GW190519_153544','GW190521_030229',
            'GW190602_175927','GW190620_030421','GW190701_203306','GW190706_222641',
            'GW190929_012149','GW190805_211137','GW191109_010717','GW191230_180458']
