import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import lines as mpllines
import h5py as h5
from plot_utils import Two_pop_plot_mass_dist,plot_2d_beta,plot_corner,Calculate_Prob,XRB_PPC,Dyn_plot_mass_dist,cal_BBH_quantiles
import pickle
colors=['lightcoral','sandybrown','darkorange','goldenrod','olive','palegreen','lightseagreen','darkcyan','skyblue','navy','indigo','crimson']

outdir='results'
fig_dir='figures'

###### read results from the main model
with open('./{}/Three_pop_result.json'.format(outdir)) as a:
    data=json.load(a)
post=data['posterior']['content']

###### read results from Extend Default Model
with open('./{}/Two_pop_result.json'.format(outdir)) as a:
    data=json.load(a)
post2=data['posterior']['content']

##########################
#For main model
##########################
"""

print('ploting figure1!')

Two_pop_plot_mass_dist(post,colors=["#4d9221","#08306b","#e31a1c"],filename='./{}/Three_pop_two_pop_mqact.pdf'.format(fig_dir),Nsample=2000,Three_pop=True)


print('ploting figure4!')

Dyn_plot_mass_dist(post,colors=["#4d9221","#08306b","#e31a1c",'pink'],filename='./{}/Dyn_mqact.pdf'.format(fig_dir),Nsample=5000)


print('ploting figure5!')

params=['mu_a', 'sigma_a', 'beta1',  'mmax1', 'alpha1', 'sigma_t1', 'beta2', 'mmax2', 'alpha2', 'r2', 'mu_a3', 'sigma_a3', 'mmin3', 'mmax3', 'alpha3', 'r3','zeta3','sigma_t3']
show_keys=[r'$\mu_{{\rm a,1}}$',r'$\sigma_{{\rm a,1}}$',r'$\beta_{\rm F}$',r'$m_{{\rm max,F}}[M_{\odot}]$',r'$\alpha_{\rm F}$',r'$\sigma_{{\rm t,F}}$',r'$\beta_{\rm D}$',r'$m_{{\rm max,D,1}}[M_{\odot}]$',r'$\alpha_{\rm D}$',r'$r_{\rm D}$',r'$\mu_{{\rm a,2}}$',r'$\sigma_{{\rm a,2}}$',r'$m_{{\rm min,D,2}} [M_{\odot}]$',r'$m_{{\rm max,D,2}} [M_{\odot}]$',r'$\alpha_{\rm D,2}$',r'$r_{\rm D,2}$',r'$\zeta_{\rm AGN,2}$',r'$\sigma_{\rm t,AGN}$']
plot_corner(post,params=params,show_keys=show_keys,color='skyblue',filename='./{}/Three_pop_corner.pdf'.format(fig_dir),)


print('ploting figure3!')

XRB_PPC(post,colors=["#e31a1c","#4d9221","#08306b"],n_catalogs=1000,N=500,filename='./{}/Three_pop_HMXB_Field.pdf'.format(fig_dir),channel='field')
XRB_PPC(post,colors=colors,n_catalogs=1000,N=500,filename='./{}/Three_pop_HMXB_Dyn.pdf'.format(fig_dir),channel='dyn')

print('ploting figure2!')

plot_2d_beta(post,paras=['beta1','beta2'],labels=[r'$\beta_{\rm F}(\beta_{\rm A})$',r'$\beta_{\rm D}(\beta_{\rm I})$'],xlim=[-8,8],ylim=[-8,8],filename='./{}/Compare_beta.pdf'.format(fig_dir),color='#CC6677',plot_double=True,post2=post2)
plot_2d_beta(post,paras=['alpha1','alpha2'],labels=[r'$\alpha_{\rm F}$',r'$\alpha_{\rm D}$'],xlim=[-8,8],ylim=[-8,8],filename='./{}/Three_pop_alpha.pdf'.format(fig_dir),color="#4d9221")


print('ploting figure6!')
values=np.array([0.99])
size=None
quants=cal_BBH_quantiles(post,values,Nsample=size)
quant_dict={str(values[i]):quants[i] for i in np.arange(1)}
quant_dict.update({key:np.array(post[key])[:size] for key in ['mmax1']})
params=['0.99','mmax1']
show_keys=[r'$m_{\rm 99\%}[M_{\odot}]$',r'$m_{\rm max,1}[M_{\odot}]$']
plot_corner(quant_dict,params=params,show_keys=show_keys,color='crimson',filename='./{}/Field_BBH_quantiles_corner.pdf'.format(fig_dir))


##########################
#For extend default model
##########################

print('ploting figure8!')
params=[ 'mu_a', 'sigma_a', 'beta1', 'mmin1','delta1', 'mmax1', 'alpha1', 'sigma_t1', 'beta2', 'mmin2','delta2', 'mmax2',  'alpha2', 'r2']
show_keys=[r'$\mu_{{\rm a}}$',r'$\sigma_{{\rm a}}$',r'$\beta_{\rm A}$',r'$m_{{\rm min,A}}[M_{\odot}]$',r'$\delta_{\rm m,A}$',r'$m_{{\rm max,A}}[M_{\odot}]$',r'$\alpha_{\rm A}$',r'$\sigma_{{\rm t}}$',r'$\beta_{\rm I}$',r'$m_{{\rm min,I}}[M_{\odot}]$',r'$\delta_{\rm m,I}$',r'$m_{{\rm max,I}}[M_{\odot}]$',r'$\alpha_{\rm I}$',r'$r_{\rm I}$']
plot_corner(post2,params=params,show_keys=show_keys,color='green',filename='./{}/Two_pop_corner.pdf'.format(fig_dir))

print('ploting figure7!')
Two_pop_plot_mass_dist(post2,colors=["#4d9221","#08306b","#e31a1c"],filename='./{}/Two_pop_mqact.pdf'.format(fig_dir),Nsample=2000)

######### ploting figure12

print('ploting figure12!')
with open('./{}/Two_pop_1_result.json'.format(outdir)) as a:
    data=json.load(a)
post3=data['posterior']['content']

params=[ 'mu_a', 'sigma_a', 'beta1', 'alpha1', 'sigma_t1', 'beta2', 'alpha2', 'r2']
show_keys=[r'$\mu_{{\rm a}}$',r'$\sigma_{{\rm a}}$',r'$\beta_{\rm A}$',r'$\alpha_{\rm A}$',r'$\sigma_{{\rm t}}$',r'$\beta_{\rm I}$',r'$\alpha_{\rm I}$',r'$r_{\rm I}$']
plot_corner(post3,params=params,show_keys=show_keys,color='orange',filename='./{}/Two_pop_1_corner.pdf'.format(fig_dir))

######### ploting figure10 figure11

print('ploting figure11!')
with open('./{}/Two_pop_varmut_result.json'.format(outdir)) as a:
    data=json.load(a)
post4=data['posterior']['content']
params=[ 'mu_t1', 'sigma_t1']
show_keys=[r'$\mu_{{\rm t}}$',r'$\sigma_{{\rm t}}$']
plot_corner(post4,params=params,show_keys=show_keys,color='purple',filename='./{}/mut_corner.pdf'.format(fig_dir))
print('ploting figure10!')
Two_pop_plot_mass_dist(post4,colors=["#4d9221","#08306b","#e31a1c"],filename='./{}/Two_pop_varmut_mqact.pdf'.format(fig_dir),Nsample=2000)
"""
######### ploting figure9

print('ploting figure9!')
with open('./{}/Two_pop_restric_05_result.json'.format(outdir)) as a:
    data=json.load(a)
post5=data['posterior']['content']
Two_pop_plot_mass_dist(post5,colors=["#4d9221","#08306b","#e31a1c"],filename='./{}/Two_pop_restrict_mqact.pdf'.format(fig_dir),Nsample=2000)
