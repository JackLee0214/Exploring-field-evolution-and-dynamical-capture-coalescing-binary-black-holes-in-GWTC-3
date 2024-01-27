
## `Exploring field-evolution and dynamical-capture coalescing binary black holes in GWTC-3`


These files include the codes and data to re-produce the results of the work  _Resolving the stellar-collapse and hierarchical-merger origins of the coalescing black holes_, arXiv: To be updated
, [Yin-Jie Li](https://inspirehep.net/authors/1838354) ,  [Yuan-Zhu Wang](https://inspirehep.net/authors/1664025), and  [Shao-Peng Tang](https://inspirehep.net/authors/1838355) 
#### Main requirements
- [BILBY](https://git.ligo.org/lscsoft/bilby)
- [PyMultiNest](https://johannesbuchner.github.io/PyMultiNest/install.html)

#### Data
The events posterior samples are adopted from the [Gravitational Wave Open Science Center](https://www.gw-openscience.org/eventapi/html/GWTC/), here `C01:Mixed` samples are used for analysis and stored in `data/GWTC3_BBH_Mixed_5000.pickle`, and `GWTC3_BBH_Mixed5000_ruled_2G.pickle`. In the later sample,G2_events: 'GW170729_185629','GW190517_055101','GW190519_153544','GW190521_030229','GW190602_175927','GW190620_030421','GW190701_203306','GW190706_222641','GW190929_012149','GW190805_211137','GW191109_010717', and 'GW191230_180458' are excluded.


The injection campaigns `data/o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5`
Note, one should first download the injection campaign
`o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5` from [Abbot et al.](https://doi.org/10.5281/zenodo.5546676), 
and set it to `data/`
  
#### Hierarchical Bayesian inference
- Inference with our main model: run the python script `inference.py` , and specify the *main model*, *Extend Default* model, *Default* model, *Single Aligned*, *Single Isotropic* by setting `label='Three_pop'`, `label='Two_pop'`, `'Default'`, `'Aligned'`, `'Isotropic'` in the script.

- Inference with the restriction that sigma_t smaller than 0.5 by setting `add_label='_restric'` in the script. in the script.

The inferred results `*.json` will be saved to `results`

#### Results
- `Three_pop_result.pickle` is the posterior samples inferred by the main model.
- `Two_pop_result.pickle` is the events' samples reweighed by the Extend Default model.

#### Generate figures
Run the python script `Figure_script.py`

The figures will be saved to `figures`
  
#### Acknowledgements
The  publicly available code [GWPopulation](https://github.com/ColmTalbot/gwpopulation) is referenced to calculate the variance of log-likelihood in the Monte Carlo integrals, and the [FigureScript](https://dcc.ligo.org/public/0171/P2000434/003/Produce-Figures.ipynb) from [LIGO Document P2000434](https://dcc.ligo.org/LIGO-P2000434/public) is referenced to produced figures in this work.


  


