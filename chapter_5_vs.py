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

#%%
mu_hpd = az.hpd(trace51['mu'], credible_interval=.89)
post_pred_hpd = az.hpd(post_pred['divorce'], credible_interval=.89)
post_pred_hpd = -np.sort(-post_pred_hpd, axis = 0)


#%%
plot_fill_between(d.medianagemarriage_s.values, 
                d.divorce.values, trace51['alpha'], trace51['beta'], mu_hpd, post_pred_hpd,'median_age','divorce rate','yeah' )

#%%


#%%
