import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from collections import Counter
from scipy.interpolate import griddata
import pymc3 as pm
import theano as tt
import statsmodels.api as sm
import scipy as sc

sns.set()

# 7.1
d = pd.read_csv('./data/rugged.csv', sep = ';')
d['log_gdp'] = np.log(d['rgdppc_2000'])
dd = d[~d['rgdppc_2000'].isnull()].reset_index(drop=True).copy()
da1 = dd[dd.cont_africa == 1].reset_index(drop=True).copy()
da0 = dd[dd.cont_africa == 0].reset_index(drop=True).copy()


# 7.2
with pm.Model() as m7_1:
    a1 = pm.Normal('a1', mu = 8, sigma = 100)
    b1 = pm.Normal('b1', sigma = 1)
    s1 = pm.Uniform('s1', upper = 10)
    mu1 = pm.Deterministic('mu1', a1 + b1*da1.rugged.values)
    log_gdp = pm.Normal('log_gdp',mu=mu1, sigma = s1, observed = da1.log_gdp.values)
    tracem71 = pm.sample(draws=1000, tune = 1000)

with pm.Model() as m7_2:
    a2 = pm.Normal('a2', mu = 8, sigma = 100)
    b2 = pm.Normal('b2', sigma = 1)
    s2 = pm.Uniform('s2', upper = 10)
    mu2 = pm.Deterministic('mu2', a2 + b2*da0.rugged.values)
    log_gdp = pm.Normal('log_gdp',mu=mu2, sigma = s2, observed = da0.log_gdp.values)
    tracem72 = pm.sample(draws=1000, tune = 1000)

pm.summary(tracem71)
pm.summary(tracem72)

def plot_poserterior_mean(trace_mu,x_val, y_val,credible_interval = .97):
    idx = np.argsort(x_val)
    mu_hpd = az.hpd(trace_mu, credible_interval=credible_interval)

    plt.plot(x_val, y_val, marker = 'o', linestyle = '')
    plt.plot(x_val[idx], trace_mu.mean(axis = 0)[idx], linestyle = '-')
    plt.fill_between(x_val[idx],mu_hpd[idx,0],mu_hpd[idx,1],color = 'grey', alpha =.3)
    plt.xlabel('rugged')
    plt.ylabel('log gdp')
    return plt

plot_poserterior_mean(tracem71['mu1'], da1.rugged, da1.log_gdp)
plt.show()
plot_poserterior_mean(tracem72['mu2'], da0.rugged, da0.log_gdp)
plt.show()

# 7.3
with pm.Model() as m7_3:
    alpha = pm.Normal('alpha', mu = 8, sigma = 100)
    beta = pm.Normal('beta', sigma = 1)
    sigma = pm.Uniform('sigma', upper = 10)
    mu = pm.Deterministic('mu', alpha + beta*dd.rugged.values)
    log_gdp = pm.Normal('log_gdp',mu=mu, sigma = sigma, observed = dd.log_gdp.values)
    tracem73 = pm.sample(draws=1000, tune = 1000)

plot_poserterior_mean(tracem73['mu'], dd.rugged, dd.log_gdp)

# 7.4
with pm.Model() as m7_4:
    alpha = pm.Normal('alpha', mu = 8, sigma = 100)
    beta = pm.Normal('beta', sigma = 1)
    beta2 = pm.Normal('beta2', sigma = 1)
    sigma = pm.Uniform('sigma', upper = 10)
    mu = pm.Deterministic('mu', alpha + beta*dd.rugged.values + beta2*dd.cont_africa.values)
    log_gdp = pm.Normal('log_gdp',mu=mu, sigma = sigma, observed = dd.log_gdp.values)
    tracem74 = pm.sample(draws=1000, tune = 1000)

# 7.5
m7_3.name = 'm73'
m7_4.name = 'm74'
pm.compare({m7_3:tracem73, m7_4:tracem74})

# 7.6
rugged_seq = np.arange(-1,8,.25)
mu_Af = np.zeros((len(rugged_seq),tracem74['mu'].shape[0]))
mu_noAf = np.zeros((len(rugged_seq),tracem74['mu'].shape[0]))

for row, seq in enumerate(rugged_seq):
    mu_Af[row,:] = tracem74['alpha'] + tracem74['beta']*rugged_seq[row] + tracem74['beta2']*1
    mu_noAf[row,:] = tracem74['alpha'] + tracem74['beta']*rugged_seq[row] + tracem74['beta2']*0

hpd_af = az.hpd(mu_Af.T,credible_interval=.97)
hpd_noaf = az.hpd(mu_noAf.T,credible_interval=.97)


plt.plot(da1.rugged, da1.log_gdp, marker = 'o', linestyle = '', color = 'blue')
plt.plot(rugged_seq, mu_Af.mean(1), color = 'blue')
plt.fill_between(rugged_seq, hpd_af[:,0], hpd_af[:,1], alpha = .4)
plt.plot(da0.rugged, da0.log_gdp, marker = 'o', linestyle = '', color = 'black')
plt.plot(rugged_seq, mu_noAf.mean(1), color = 'black')
plt.fill_between(rugged_seq, hpd_noaf[:,0], hpd_noaf[:,1], alpha = .2, color = 'black')
plt.xlabel('Terrain Ruggedness Index')
plt.ylabel('log GDP')

# 7.7
with pm.Model() as m7_5:
    alpha = pm.Normal('alpha', mu = 8, sigma = 100)
    betaR = pm.Normal('betaR', sigma = 1)
    betaA = pm.Normal('betaA', sigma = 1)
    betaAR = pm.Normal('betaAR', sigma = 1)
    sigma = pm.Uniform('sigma', upper = 10)
    gamma = betaR + betaAR*dd.cont_africa.values
    mu = alpha + gamma*dd.rugged.values + betaA*dd.cont_africa.values
    log_gdp = pm.Normal('log_gdp',mu=mu, sigma = sigma, observed = dd.log_gdp.values)
    tracem75 = pm.sample(draws=1000, tune = 1000)

pm.summary(tracem75)

# 7.8
m7_5.name = 'm75'
comp = pm.compare({m7_3:tracem73, m7_4:tracem74, m7_5: tracem75})
comp
pm.compareplot(comp)

# 7.9
with pm.Model() as m7_5b:
    alpha = pm.Normal('alpha', mu = 8, sigma = 100)
    betaR = pm.Normal('betaR', sigma = 1)
    betaA = pm.Normal('betaA', sigma = 1)
    betaAR = pm.Normal('betaAR', sigma = 1)
    sigma = pm.Uniform('sigma', upper = 10)
    mu = alpha + betaR*dd.rugged + betaAR*dd.rugged.values*dd.cont_africa.values + betaA*dd.cont_africa.values
    log_gdp = pm.Normal('log_gdp',mu=mu, sigma = sigma, observed = dd.log_gdp.values)
    tracem75b = pm.sample(draws=1000, tune = 1000)

m7_5b.name = 'm75b'
pm.summary(tracem75b)
comp_ = pm.compare({m7_5:tracem75, m7_5b:tracem75b})
plt.figure(figsize=(10,8))
pm.compareplot(comp_)

# 7.10
rugged_seq = np.arange(-1,8,.25)
mu_Af = np.zeros((len(rugged_seq),tracem75['alpha'].shape[0]))
mu_noAf = np.zeros((len(rugged_seq),tracem75['alpha'].shape[0]))

for row, seq in enumerate(rugged_seq):
    mu_Af[row,:] = tracem75['alpha'] + (tracem75['betaR'] + tracem75['betaAR']*1)*rugged_seq[row]  +  tracem75['betaA']*1
    mu_noAf[row,:] = tracem75['alpha'] + (tracem75['betaR'] + tracem75['betaAR']*0)*rugged_seq[row]  +  tracem75['betaA']*0

hpd_af = az.hpd(mu_Af.T,credible_interval=.97)
hpd_noaf = az.hpd(mu_noAf.T,credible_interval=.97)

# 7.11
plt.plot(da1.rugged, da1.log_gdp, marker = 'o', linestyle = '', color = 'blue')
plt.plot(rugged_seq, mu_Af.mean(1), color = 'blue')
plt.fill_between(rugged_seq, hpd_af[:,0], hpd_af[:,1], alpha = .4)
plt.xlabel('Terrain Ruggedness Index')
plt.ylabel('log GDP')
plt.show()
plt.plot(da0.rugged, da0.log_gdp, marker = 'o', linestyle = '', color = 'black')
plt.plot(rugged_seq, mu_noAf.mean(1), color = 'black')
plt.fill_between(rugged_seq, hpd_noaf[:,0], hpd_noaf[:,1], alpha = .2, color = 'black')
plt.xlabel('Terrain Ruggedness Index')
plt.ylabel('log GDP')
plt.show()

# 7.12
pm.summary(tracem75).round(2)

# 7.13
gamma_Africa = tracem75['betaR'] + tracem75['betaAR'] * 1
gamma_notAfrica = tracem75['betaR']

# 7.14
gamma_Africa.mean()
gamma_notAfrica.mean()

# 7.15
sns.kdeplot(gamma_Africa, color = 'blue',label = 'Africa')
sns.kdeplot(gamma_notAfrica, color = 'black', label = 'Not Africa')
plt.legend()