#%%
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

sns.set()

#%%
# Functions
def sort_vals(vals, ascending = True):
    """sorts valus from high to low
    returns:
    idx - values in ascending or descending order
    """
    if ascending:
        idx = np.argsort(-vals)
    else:
        idx = np.argsort(vals)
    return idx

def plot_fill_between(x_vals, y_vals, alpha_mean, beta_mean, mu_hpd, pred_hpd, xlabel, ylabel, title, figsize = (10,8)):
    sorted_x_vals = np.sort(x_vals, axis = 0)
    mu_pred_sort = -np.sort(-mu_hpd, axis = 0)

    plt.figure(figsize=figsize)
    plt.plot(x_vals,y_vals, color = 'orange', marker = '.', linestyle = '')
    plt.plot(sorted_x_vals, np.mean(alpha_mean) + np.mean(beta_mean)*sorted_x_vals, color = 'white', alpha = 1)
    plt.fill_between(sorted_x_vals, mu_pred_sort[:,0], mu_pred_sort[:,1], color='white', alpha=0.3)
    plt.fill_between(sorted_x_vals,pred_hpd[:,0], pred_hpd[:,1], color = 'grey' )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontsize = 18)
    
    return plt.show()

#%% [markdown]
# # Chapter 5

#%%
d = pd.read_csv('.\data\WaffleDivorce.csv', sep = ';')
d.columns = d.columns.str.lower()
d.head()

#%%
d['medianagemarriage_s'] = (d.medianagemarriage - d.medianagemarriage.mean())/ d.medianagemarriage.std()

#%% [markdown]
# ## Code 5.1
#%%
shared_x = tt.shared(d.medianagemarriage_s.values)
shared_y = tt.shared(d.divorce.values)
with pm.Model() as m51:
    alpha = pm.Normal('alpha', mu = 10, sigma = 10)
    beta = pm.Normal('beta', mu = 0, sigma = 1)
    mu = pm.Deterministic('mu', alpha + beta*shared_x)
    sigma = pm.Uniform('sigma', lower = 0, upper = 10)
    divorce = pm.Normal('divorce',mu = mu, sigma = sigma, observed = shared_y)
    trace51 = pm.sample(draws = 1000,tune = 1000)

#%%
varnames = ['alpha', 'beta','sigma']
pm.summary(trace51, varnames = varnames)

#%% [markdown]
# ## Code 5.2
#%%
new_x_values = np.linspace(-3,3.5,num = 30)
shared_x.set_value(new_x_values)
shared_y.set_value(np.repeat(0, repeats = len(new_x_values)))
with m51:
    post_pred = pm.sample_posterior_predictive(trace51,samples = 1000,model=m51)


#%%
mu_hpd = az.hpd(trace51['mu'], credible_interval=.89)
post_pred_hpd = az.hpd(post_pred['divorce'], credible_interval=.89)


#%%
idx = sort_vals(d.medianagemarriage_s)
sorted_x_vals = d.medianagemarriage_s[idx]

plt.figure(figsize=(10,8))
plt.plot(d.medianagemarriage_s.values,d.divorce.values, color = 'blue', marker = '.', linestyle = '')
plt.plot(sorted_x_vals, trace51['alpha'].mean() + np.mean(trace51['beta'])*sorted_x_vals, color = 'black', alpha = 1)
plt.fill_between(sorted_x_vals, mu_hpd[idx,0], mu_hpd[idx,1], color='black', alpha=0.3)
plt.xlabel('Median Age Marriage')
plt.ylabel('Divorce')

plt.show()

#%% [markdown]
# ## Code 5.3
#%%
d['marriage_s'] = (d.marriage - d.marriage.mean())/ d.marriage.std()

#%%
shared_x = tt.shared(d.marriage_s.values)
shared_y = tt.shared(d.divorce.values)
with pm.Model() as m53:
    alpha = pm.Normal('alpha', mu = 10, sigma = 10)
    beta = pm.Normal('MAM_beta', mu = 0, sigma = 1)
    mu = pm.Deterministic('mu', alpha + beta*shared_x)
    sigma = pm.Uniform('sigma', lower = 0, upper = 10)
    divorce = pm.Normal('divorce',mu = mu, sigma = sigma, observed = shared_y)
    trace53 = pm.sample(draws = 1000,tune = 1000)

#%%
varnames_53 = ['alpha', 'MAM_beta','sigma']
pm.summary(trace53, varnames = varnames_53)

#%%
new_x_values = np.linspace(-3,3.5,num = 30)
shared_x.set_value(new_x_values)
shared_y.set_value(np.repeat(0, repeats = len(new_x_values)))
with m53:
    post_pred = pm.sample_posterior_predictive(trace53,samples = 1000,model=m53)

#%%
mu_hpd = az.hpd(trace53['mu'], credible_interval=.89)
post_pred_hpd = az.hpd(post_pred['divorce'], credible_interval=.89)

#%%
idx = sort_vals(d.marriage_s)
sorted_x_vals = d.marriage_s[idx]

plt.figure(figsize=(10,8))
plt.plot(d.marriage_s.values,d.divorce.values, color = 'blue', marker = '.', linestyle = '')
plt.plot(sorted_x_vals, trace53['alpha'].mean() + np.mean(trace53['beta'])*sorted_x_vals, color = 'black', alpha = 1)
plt.fill_between(sorted_x_vals, mu_hpd[idx,0], mu_hpd[idx,1], color='black', alpha=0.3)
plt.xlabel('Median Age Marriage', fontsize = 14)
plt.ylabel('Divorce', fontsize = 14)
plt.title('Divorce ~ Marriage', fontsize = 16)

plt.show()


#%% [markdown]
# ## Code 5.4
#%%
shared_x = tt.shared(d[['marriage_s','medianagemarriage_s']].values)
shared_y = tt.shared(d.divorce.values)

with pm.Model() as m54:
    alpha = pm.Normal('alpha', mu = 10, sigma = 10)
    beta = pm.Normal('MARR_beta', mu = 0, sigma = 1)
    beta2 = pm.Normal('MAM_beta', mu = 0, sigma = 1)
    mu = pm.Deterministic('mu', alpha + beta*shared_x.get_value()[:,0] + beta2*shared_x.get_value()[:,1])
    sigma = pm.Uniform('sigma', lower = 0, upper = 10)
    divorce = pm.Normal('divorce',mu = mu, sigma = sigma, observed = shared_y)
    trace54 = pm.sample(draws = 1000,tune = 1000)

#%%
varnames = ['alpha', 'MARR_beta','MAM_beta','sigma']
pm.summary(trace54, varnames = varnames, alpha = .11).round(3)
#%%
#notice how after adding im the marriage rate of the state our signs flip from positive to negative.
# this is classic example of mulitcollinearity
pm.summary(trace53, varnames = varnames_53, alpha = .11).round(3)

#%%[markdow]
# ## Code 5.5
#%%
# interpretaion from the book, "Once we know median age of marraiage for a state there is little or no additional 
# predictive power in also knowing the rate of marriage in that state"
az.plot_forest(trace54,var_names=varnames)
plt.vlines(x = 0, ymin = 0, ymax = 5)



#%% [markdown]
# ## Code 5.6
#%%
shared_x = tt.shared(d.medianagemarriage_s.values)
shared_y = tt.shared(d.divorce.values)
with pm.Model() as m56:
    alpha = pm.Normal('alpha', mu = 10, sigma = 10)
    beta = pm.Normal('MAM_beta', mu = 0, sigma = 1)
    mu = pm.Deterministic('mu', alpha + beta*shared_x)
    sigma = pm.Uniform('sigma', lower = 0, upper = 10)
    divorce = pm.Normal('divorce',mu = mu, sigma = sigma, observed = shared_y)
    trace56 = pm.sample(draws = 1000,tune = 1000)

#%%
varnames_56 = ['alpha', 'MAM_beta','sigma']
pm.summary(trace56, varnames = varnames)

#%% [markdown]
# ## Code 5.7
#%%
# computer expected value at MAP, for each State
mu = trace56['alpha'].mean() + trace56['MAM_beta'].mean()*d.medianagemarriage_s
# compute residual for each state
m_resid = d.medianagemarriage_s - mu

#%%
idx = np.argsort(d.medianagemarriage_s)
plt.plot('medianagemarriage_s', 'marriage_s', data = d, marker = '.', linestyle = '')
plt.plot(d.loc[idx,'medianagemarriage_s'], mu[idx], linestyle = '-',color = 'black')



#%%
