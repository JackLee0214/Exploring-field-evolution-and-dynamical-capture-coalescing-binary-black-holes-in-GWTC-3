import bilby
from bilby.core.sampler import run_sampler
import numpy as np
import h5py
import pickle
from bilby.core.result import read_in_result as rr
import json
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import pickle
from bilby.hyper.likelihood import HyperparameterLikelihood

from bilby.core.prior import Interped, Uniform, LogUniform, DeltaFunction
import sys

from Spin_Ori_model_libs import Two_pop_priors, hyper_Two_pop, hyper_Three_pop, Three_pop_priors, Rate_selection_function_with_uncertainty
from Spin_Ori_model_libs import Two_pop_nospin, Three_pop_nospin
from Spin_Ori_model_libs import hyper_PLS_default, Compare_priors, PLS_default, PLS_default_nospin
run=1

outdir='results'
label=['Three_pop','Two_pop','Default','Aligned','Isotropic'][int(sys.argv[1])]
add_label=''
#add_label='_restric'
sampler='pymultinest'
npool=1

#read data
if label in ['Three_pop']:
    with open('./data/GWTC3_BBH_Mixed5000_full.pickle', 'rb') as fp:
        samples, ln_evidences = pickle.load(fp)
else:
    with open('./data/GWTC3_BBH_Mixed5000_ruled_2G.pickle', 'rb') as fp:
        samples, ln_evidences = pickle.load(fp)
Nobs=len(samples)
print(Nobs)

Neff_obs_thr=10

if label=='Two_pop':
    hyper_prior=hyper_Two_pop
    mass_spin_model=Two_pop_nospin
    priors=Two_pop_priors()
    if add_label=='_restric':
        priors['sigma_t1']=Uniform(0.1,0.5)
elif label=='Three_pop':
    hyper_prior=hyper_Three_pop
    mass_spin_model=Three_pop_nospin
    priors=Three_pop_priors()
    
else:
    hyper_prior=hyper_PLS_default
    mass_spin_model=PLS_default_nospin
    priors=Compare_priors()
    if label=='Default':
        pass
    elif label=='Aligned':
        priors['zeta']=1
        priors['sigma_t']=Uniform(0.1,4)
    elif label=='Isotropic':
        priors['zeta']=0
        priors['sigma_t']=100
        
    if add_label=='_restric':
        priors['sigma_t']=Uniform(0.1,0.5)


class Hyper_selection_with_var(HyperparameterLikelihood):

    def likelihood_ratio_obs_var(self):
        self.hyper_prior.parameters.update(self.parameters)
        weights = np.nan_to_num(self.hyper_prior.prob(self.data) / self.data['prior'])
        expectations = np.nan_to_num(np.mean(weights, axis=-1))
        square_expectations = np.mean(weights**2, axis=-1)
        variances = (square_expectations - expectations**2) / (
            self.samples_per_posterior * expectations**2
        )
        variance = np.sum(variances)
        Neffs = expectations**2/square_expectations*self.samples_per_posterior
        Neffmin = np.min(Neffs)
        return np.nan_to_num(variance), np.nan_to_num(Neffmin), np.nan_to_num(np.sum(np.log(expectations)))
        
    def log_likelihood(self):
        obs_vars, obs_Neff, llhr= self.likelihood_ratio_obs_var()
        if (obs_Neff>Neff_obs_thr):
            #print('obs_Neff:',obs_Neff)
            selection, sel_vars, sel_Neff = Rate_selection_function_with_uncertainty(self.n_posteriors, mass_spin_model, **self.parameters)
            if (sel_Neff>4*self.n_posteriors):
                #print('sel_Neff:',sel_Neff)
                return self.noise_log_likelihood() + llhr + selection
            else:
                return -1e100
        else:
            return -1e100
            
    def get_log_total_vars(self, mass_spin_model):
        selection, sel_vars, sel_Neff = Rate_selection_function_with_uncertainty(self.n_posteriors, mass_spin_model, **self.parameters)
        obs_vars, obs_Neff = self.likelihood_obs_var()
        total_vars=sel_vars+obs_vars
        return total_vars
 
hp_likelihood = Hyper_selection_with_var(posteriors=samples, hyper_prior=hyper_prior, log_evidences=ln_evidences, max_samples=1e+100)
#bilby.core.utils.setup_logger(outdir=outdir, label=label+add_label)
if run:
    result = run_sampler(likelihood=hp_likelihood, priors=priors, sampler=sampler, nlive=1000,npool=npool,
                use_ratio=False, outdir=outdir, label=label+add_label)
else:
    result=rr('./{}/{}_result.json'.format(outdir,label+add_label))

with open('./{}/{}_result.json'.format(outdir,label+add_label)) as a:
    data=json.load(a)
plot_paras=[key for key in result.search_parameter_keys if key not in ['N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9','o2','o3','o4','o5','o6','o7']]
result.plot_corner(quantiles=[0.05, 0.95],parameters=plot_paras,filename='./{}/{}_corner.pdf'.format(outdir,label+add_label),smooth=1.,color='green')
