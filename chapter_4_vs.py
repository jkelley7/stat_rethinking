
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
# ### Code 4.1a

#%%
np.random.seed(42)
for i in range(1,18,4):
    pos = np.sum(np.random.uniform(low = -1, high = 1,size = (i,1000)), axis= 0)
    sns.distplot(pos)
    plt.xlabel('position')
    plt.ylabel('Density')
    plt.title(f'{i} steps')
    plt.show()

#%% [markdown]
# ### Code 4.2 & 4.3
# 
# small deviates multipled together form a normal distribution. This is due to multiplying small numbers together are approximately the same as addition

#%%
np.random.seed(42)
growth = np.prod(1 + np.random.uniform(0,.1, size = (12,10000)), axis = 0)
sns.distplot(growth,norm_hist=True)
plt.xlabel('product')
plt.ylabel('Density')
plt.show()

#%% [markdown]
# ### Code 4.4

#%%
big = np.prod(1 + np.random.uniform(0,.5, size = (12,10000)), axis = 0)
small = np.prod(1 + np.random.uniform(0,.01, size = (12,10000)), axis = 0)
plt.figure()
sns.distplot(big,norm_hist=False)
plt.xlabel('product')
plt.ylabel('Density')
plt.title('big')
plt.figure()
sns.distplot(small,norm_hist=False)
plt.xlabel('product')
plt.ylabel('Density')
plt.title('small')
plt.show()

#%% [markdown]
# ### Code 4.5
# 
# Large deviates which are multipled together tend to produce gaussians on the log scale
# Why? because adding logs is equivalent to multiplying original numbers

#%%
np.random.seed(42)
log_big = np.log(np.prod(1 + np.random.uniform(0,.5, size = (12,10000)), axis = 0))
sns.distplot(log_big,norm_hist=False)
plt.xlabel('product')
plt.ylabel('Density')
plt.title('log big')
plt.show()

#%% [markdown]
# ### Code 4.6

#%%
w = 6
n = 9
p_grid = np.linspace(0,1,100)
posterior = stats.binom.pmf(k = w,n = n, p= p_grid)*stats.uniform.pdf(p_grid, 0 ,1) #likelihood * prior
posterior = posterior/np.sum(posterior)


#%%
plt.plot(p_grid, posterior)
plt.xlabel('probability')
plt.ylabel('density')
plt.show()

#%% [markdown]
# ### Code 4.7 - 4.10

#%%
# 4.7
d = pd.read_csv('.\data\Howell1.csv', sep = ";")
d.head()

# 4.8
print(d.info())

# 4.9
d['height'].head()

#4.10
d2 = d.query('age >= 18')
print(f'the length of d2 is {len(d2)}')


#%%
# plot the height of people over the age of 18
# it looks rather gaussian in shape. This maybe because height is the sum of man small growth factors
# as we said previously, distribution of sums tends to converge to gaussian dist
sns.distplot(d2['height'])

#%% [markdown]
# ### Code 4.11

#%%
mu = np.linspace(100,250, 150)
plt.plot(mu, stats.norm.pdf(mu, 178, 20))
plt.xlabel('heights (cm) mu')
plt.ylabel('density')
plt.show()

#%% [markdown]
# ### Code 4.12

#%%
std = np.linspace(-10,60, 150)
plt.plot(std, stats.uniform.pdf(std, 0, 50))
plt.xlabel('heights (cm) std')
plt.ylabel('density')
plt.show()

#%% [markdown]
# ### Code 4.13

#%%
samples = int(1e4)
sample_mu = np.random.normal(loc = 178, scale = 20, size = samples)
sample_sigma = np.random.uniform(low= 0, high = 50, size = samples)
prior_h = np.random.normal(loc = sample_mu, scale = sample_sigma, size = samples)
sns.distplot(prior_h)
plt.xlabel('height (cm)')
plt.ylabel('density')
plt.show()

#%% [markdown]
# ### Code 4.14

#%%
mu_list = np.linspace(140,160,num = 200)
sigma_list = np.linspace(4,9, num = 200)
post = np.array(np.meshgrid(mu_list, sigma_list)).reshape(2, -1).T
log_likeli=  [sum(stats.norm.logpdf(x = d2.height ,loc = post[i,0],scale = post[i,1])) for i in range(len(post))]
post_prodll = (log_likeli 
               + stats.norm.logpdf(post[:,0], loc= 178, scale = 20)
               + stats.uniform.logpdf(post[:,1], loc = 0 ,scale = 50)
              )
post_prod = np.exp(post_prodll - max(post_prodll))


#%% [markdown]
# ### Code 4.15

#%%
#code from pymc3
# post = np.mgrid[140:160:0.1, 4:9:0.1].reshape(2,-1).T

# likelihood = [sum(stats.norm.logpdf(d2.height, loc=post[:,0][i], scale=post[:,1][i])) for i in range(len(post))]

# post_prod = (likelihood + 
#              stats.norm.logpdf(post[:,0], loc=178, scale=20) + 
#              stats.uniform.logpdf(post[:,1], loc=0, scale=50))
# post_prob = np.exp(post_prod - max(post_prod))
#%% [markdown]
# ### Code 4.15
# this one was borrowed from the PYMC3 as I needed help

#%%
xi = np.linspace(post[:,0].min(), post[:,0].max(), 100)
yi = np.linspace(post[:,1].min(), post[:,1].max(), 100)
zi = griddata((post[:,0], post[:,1]), post_prod, (xi[None,:], yi[:,None]))

plt.contour(xi, yi, zi)
plt.xlabel('height')
plt.ylabel('height sigma')

#%% [markdown]
# ### Code 4.16-.17
#%%
size = int(1e5)
nrows, ncols = post.shape
np.random.seed(42)
post_sample = np.random.choice(np.arange(nrows),size = size, replace = True, p =( post_prod/post_prod.sum()))
sample_mu = post[post_sample,0]
sample_sig = post[post_sample,1]

#%% [markdown]
# ### Code 4.18
#%%
sns.jointplot(sample_mu, sample_sig, kind = 'hex',)
plt.show()

#%% [markdown]
# ### Code 4.19-.20
#%%
sns.kdeplot(sample_mu)
plt.xlabel('sample_mu')
plt.ylabel('density')
plt.show()
sns.kdeplot(sample_sig)
plt.xlabel('sample_sigma')
plt.ylabel('density')
plt.show()


#%%
print(az.hpd(sample_mu, credible_interval=0.5))
az.hpd(sample_sig, credible_interval=0.5)


#%% [markdown]
# ### Code 4.21
#%%
d3 = np.random.choice(d2.height, size = 20, replace = False)

#%% [markdown]
# ### Code 4.22-.23
# We are doing this because we want to show that the posterior is not always
# gaussian in shape. This is not driven by the mean it's more driven by the variance which tends to have this right tail
#%%
mu_list = np.linspace(150,170,num = 200)
sigma_list = np.linspace(4,20, num = 200)
post = np.array(np.meshgrid(mu_list, sigma_list)).reshape(2, -1).T
log_likeli=  [sum(stats.norm.logpdf(x = d3 ,loc = post[i,0],scale = post[i,1])) for i in range(len(post))]
post_prodll = (log_likeli 
               + stats.norm.logpdf(post[:,0], loc= 178, scale = 20)
               + stats.uniform.logpdf(post[:,1], loc = 0 ,scale = 50)
              )
post_prod = np.exp(post_prodll - max(post_prodll))

size = int(1e5)
nrows, ncols = post.shape
np.random.seed(42)
post_sample = np.random.choice(np.arange(nrows),size = size, replace = True, p =( post_prod/post_prod.sum()))
sample_mu = post[post_sample,0]
sample_sig = post[post_sample,1]

# look at the tails of the distribution and the which is driven by the std dev
sns.distplot(sample_sig)
plt.show()
sns.jointplot(sample_mu, sample_sig, kind = 'hex',)
plt.show()

#%% [markdown]
# ### Code 4.24
#%%
d = pd.read_csv('.\data\Howell1.csv', sep = ";")
d2 = d.query('age >= 18')



#%% [markdown]
# ### Code 4.25
#%%
with pm.Model() as m4_1:
    mu = pm.Normal('mu', mu=178, sd=20)
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    height = pm.Normal('height', mu=mu, sd=sigma, observed=d2.height)


#%% [markdown]
# ### Code 4.26
#%%
with m4_1:
    trace4_1 = pm.sample(1000, tune=1000)
pm.traceplot(trace4_1, varnames = ['mu','sigma'])

#%% [markdown]
# ### Code 4.27
#%%
# from the book, "The means that the plausibility of each value of mu, after averaging over the plausibilities of each value of sigma
# , is given by a gaussian distribution with mean of 154.6 and std dev of .4"
pm.summary(trace4_1)

#%% [markdown]
# ### Code 4.28
#%%
# now say we want to inform the prior with the mean of our data
with pm.Model() as m4_1:
    mu = pm.Normal('mu', mu=178, sd=20, testval=d2.height.mean())
    sigma = pm.Uniform('sigma', lower=0, upper=50, testval=d2.height.std())
    height = pm.Normal('height', mu=mu, sd=sigma, observed=d2.height)
    trace4_1 = pm.sample(1000, tune=1000)

pm.summary(trace4_1)

#%%
# Just want to look at the traceplot of this
pm.traceplot(trace4_1, varnames = ['mu','sigma'])

#%% [markdown]
# ### Code 4.29
#%%
# now say we want to inform the prior with the mean of our data
with pm.Model() as m4_2:
    mu = pm.Normal('mu', mu=158, sd=.1)                                 #this is our prior were making it stronger by decreasing sd
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    height = pm.Normal('height', mu=mu, sd=sigma, observed=d2.height)
    trace4_2 = pm.sample(1000, tune=1000)

pm.summary(trace4_2)

#%%
# Just want to look at the traceplot of this
pm.traceplot(trace4_2, varnames = ['mu','sigma'])
plt.show()


#%% [markdown]
# ### Code 4.30 - 4.32
#%%
# What does this tell us?
# It tells us how the paramters vary within themselves AND how they are related to each other
trace_df = pm.trace_to_dataframe(trace4_1)
trace_df.cov()
#%%
#this just shows us the variances and how correlated each are to each other
np.diag(trace_df.cov())
# given the values are so low basically knowing the mean tells us nothing about sigma
trace_df.corr()
trace_df.head()

#%% [markdown]
# ### Code 4.33
#%%
pm.summary(trace4_1)

#%% [markdown]
# ### Code 4.34
#%%
# In the book this is the underlying extract samples
stats.multivariate_normal.rvs(mean= trace_df.mean(), cov = trace_df.cov(), size = 100)

#%% [markdown]
# ### Code 4.35 - 4.36
#%%
with pm.Model() as m4_1_log:
    mu = pm.Normal('mu', mu=158, sd=.1)                                 #this is our prior were making it stronger by decreasing sd
    sigma = pm.Lognormal('sigma', mu = 2, tau = .01)
    height = pm.Normal('height', mu=mu, sd=np.exp(sigma), observed=d2.height)
    trace4_1log = pm.sample(1000, tune=1000)

pm.summary(trace4_1log)
pm.traceplot(trace4_1log)
plt.show()


#%% [markdown]
# ### Code 4.37
#%%
plt.plot(d2.height, d2.weight, marker = 'o', linestyle = '')
plt.show()

#%% [markdown]
# ### Code 4.38
#%%
with pm.Model() as m4_3:
    a = pm.Normal('a',mu = 156, sigma=100)
    b = pm.Normal('b',mu = 0, sigma=10)
    mu = a + b * d2.weight
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    height = pm.Normal('height', mu=mu, sd=sigma, observed=d2.height)
    trace4_3 = pm.sample(1000, tune=1000)
#%%
pm.summary(trace4_3)

#%% [markdown]
# ### Code 4.39
#%%
with pm.Model() as m4_39:
    a = pm.Normal('a',mu = 178, sigma=100)
    b = pm.Normal('b',mu = 0, sigma=10)
    mu = a + b*d2.weight
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    height = pm.Normal('height', mu=mu, sd=sigma, observed=d2.height)
    trace4_39 = pm.sample(1000, tune=1000)
#%% [markdown]
# ### Code 4.40
#%%
pm.summary(trace4_39)

#%% [markdown]
# ### Code 4.41
trace_df_439 = pm.trace_to_dataframe(trace4_39)
trace_df_439.corr()


#%% [markdown]
# ### Code 4.43
#%%
d2['weight_c'] = d2.weight - d2.weight.mean()

#%% [markdown]
# ### Code 4.44
#%%
with pm.Model() as m4_44:
    a = pm.Normal('a',mu = 178, sigma=100)
    b = pm.Normal('b',mu = 0, sigma=10)
    mu = a + b*d2.weight_c
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    height = pm.Normal('height', mu=mu, sd=sigma, observed=d2.height)
    tracem4_44 = pm.sample(1000, tune=1000)

#%% [markdown]
# ### Code 4.44_cont
#%%
pm.summary(tracem4_44)
#%%
trace_df_444 = pm.trace_to_dataframe(tracem4_44)
trace_df_444.corr()

#%% [markdown]
# ### Code 4.45
#%%
plt.plot(d2.weight_c, d2.height, marker = '.', linestyle = '')
plt.plot(d2.weight_c, trace_df_444['a'].mean() + trace_df_444['b'].mean()*d2.weight_c)
plt.ylabel('height')
plt.xlabel('weight Centered')
#%% [markdown]
# ### Code 4.46
#%%
trace_df_444

#%% [markdown]
# ### Code 4.47
#%%
trace_df_444[1:5]

#%% [markdown]
# ### Code 4.48
#%%
row_sel = 10
dn = d2.iloc[range(0,row_sel),:]


#%% [markdown]
# ### Code 4.49
#%%
with pm.Model() as m4_49:
    a = pm.Normal('a',mu = 178, sigma=100)
    b = pm.Normal('b',mu = 0, sigma=10)
    mu = a + b*dn.weight_c
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    height = pm.Normal('height', mu=mu, sd=sigma, observed=dn.height)
    tracem4_49 = pm.sample(1000, tune=1000)
# 
#%%
trace_df_449 = pm.trace_to_dataframe(tracem4_49)
trace_df_449
#%%
for i in range(0,21):
    plt.plot(dn.weight_c, dn.height, marker = '.', linestyle = '', color = 'blue', alpha = .2)
    plt.plot(dn.weight_c, trace_df_449['a'][i] + trace_df_449['b'][i]*dn.weight_c, color = 'grey', alpha = .2)
    plt.ylabel('height', color = 'white')
    plt.xlabel('weight Centered')
    plt.title('N=10')

#%% [markdown]
# ### Code 4.50
#%%
# we have to use the one that's not centered
mu_at_50 = trace4_39['a'] + trace4_39['b']*50

#%%
sns.distplot(mu_at_50, hist = False)
plt.xlabel('mu|weight = 50')
plt.ylabel('Density')

#%% [markdown]
# ### Code 4.52
#%%
az.hpd(mu_at_50, credible_interval=0.89)


#%% [markdown]
# ### Code 4.53-4.55
#%%
weight_seq = np.arange(25,76)
condensed = trace4_39[::10]
mu = np.zeros(shape=(len(condensed['a']),len(weight_seq)))
for col, weight in enumerate(weight_seq):
    mu[:,col] = condensed['a'] + condensed['b']*weight
#%%
plt.plot(weight_seq, mu.T, 'C0.', alpha = .2)
plt.show()

#%% [markdown]
# ### Code 4.56
#%%
# "THese are just different kinds of summaries of the distributions in mu, with each column being a different weight value"
mu_mean = np.mean(mu,axis = 1)                # average at each weight value
mu_hpd = az.hpd(mu, credible_interval=.89)   # 89% highest density estimate for each weight
#%%
# as the book states it's helpful to plot and take a look at things
plt.plot(np.mean(mu, axis = 1), marker = '.', linestyle = '', alpha = .2)
plt.xlabel('record')
plt.ylabel('estimated mu')
plt.show()

#%% [markdown]
# ### Code 4.57
#%%
plt.plot(d2.weight, d2.height, marker = '.', linestyle = '', alpha = .5)
plt.plot(d2.weight, trace_df_439['a'].mean() + trace_df_439['b'].mean()*d2.weight)
plt.fill_between(weight_seq,mu_hpd[:,0], mu_hpd[:,1], color = 'white', alpha = .2 )
plt.xlabel('weight')
plt.ylabel('height')
plt.show()

#%% [markdown]
# ### Here's the reciepe for generating predictions and intervals from the posterior
# 1. The book uses the link function but we used code 4.53 or loop over the different valeus
#   - besides pymc3 drew 4000 samples so we are computing that
# 2. use summary functions mean and HPDI to find mean, lower and upper
# 3. plot the lines and shade the HPDI to see the plausbility
# 
#%% [markdown]
# ### Code 4.58
# 1. see 4.53
#%% [markdown]
# ### Code 4.59
#%%
# We are sampling from the posterior. We are generating 400 samples of the 352 records
height_samps = pm.sample_posterior_predictive(trace4_39, samples = 400,model = m4_39)

#%% [markdown]
# ### Code 4.60
#%%
height_hpd = az.hpd(height_samps['height'], credible_interval=.89)
height_hpd.shape
height_hpd_sort = np.sort(height_hpd, axis=0)

#%%
# the jaggedness is the simulation variance in the tails of the distrubtion
plt.plot(d2.weight, d2.height, marker = '.', linestyle = '', alpha = .5)
plt.plot(d2.weight, trace_df_439['a'].mean() + trace_df_439['b'].mean()*d2.weight, color = 'white')
plt.fill_between(weight_seq,mu_hpd[:,0], mu_hpd[:,1], color = 'white', alpha = .5 )
plt.fill_between(d2.weight.sort_values(),height_hpd_sort[:,0],height_hpd_sort[:,1], color = 'white', alpha = .2 )
plt.xlabel('weight')
plt.ylabel('height')
plt.show()

#%% [markdown]
# ### Code 4.63
# It's useful to know how to manuall sample. THe below was taken from the pymc3 devs
#%%
weigth_seq = np.arange(25, 71)
post_samples = []
for _ in range(1000): # number of samples from the posterior
    i = np.random.randint(len(trace4_39))
    mu_pred = trace4_39['a'][i] + trace4_39['b'][i] * weigth_seq
    sigma_pred = trace4_39['sigma'][i]
    post_samples.append(np.random.normal(mu_pred, sigma_pred))


#%% [markdown]
# ### Code 4.64
#%%
d = pd.read_csv('.\data\Howell1.csv', sep = ";")
d.head()


#%% [markdown]
# ### Code 4.65
#%%
plt.plot('weight','height', data = d, marker = '.', linestyle = '')
plt.show()


#%% [markdown]
# ### Code 4.65
#%%
d['weight_sd'] = (d.weight - d.weight.mean())/d.weight.std()
d['weight_sd2'] = d.weight_sd**2
plt.plot('weight_sd','height', data = d, marker = '.', linestyle = '')
plt.show()

#%% [markdown]
# ### Code 4.66
#%%
with pm.Model() as m4_66:
    a = pm.Normal('alpha',mu = 178, sigma=100)
    b1 = pm.Normal('beta1',mu = 0, sigma=10)
    b2 = pm.Normal('beta2',mu = 0, sigma=10)
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    mu =  pm.Deterministic('mu', a + b1*d.weight_sd + b2*d.weight_sd2)
    height = pm.Normal('height', mu=mu, sd=sigma, observed=d.height)
    trace466 = pm.sample(1000, tune=1000)


#%% [markdown]
# ### Code 4.67
#%%
pm.summary(trace466, var_names = ['alpha', 'beta1','beta2','sigma'])

#%% [markdown]
# ### Code 4.68
#%%
idx = np.argsort(d.weight_sd)
height_samps = pm.sample_posterior_predictive(trace466, samples = 1000,model = m4_66)
mu_pred = trace466['mu']
mu_hpd = az.hpd(mu_pred, credible_interval=.89)[idx]
height_hpd = az.hpd(height_samps['height'], credible_interval=.89)
height_hpd_sort = np.sort(height_hpd, axis=0)
#%% [markdown]
# ### Code 4.69
#%%
# the jaggedness is the simulation variance in the tails of the distrubtion
plt.plot(d.weight_sd, d.height, marker = '.', linestyle = '', alpha = .5)
plt.fill_between(d.weight_sd[idx], mu_hpd[:,0], mu_hpd[:,1], color='C2', alpha=0.7)
plt.fill_between(d.weight_sd[idx],height_hpd_sort[:,0],height_hpd_sort[:,1], color = 'white', alpha = .2 )
plt.xlabel('weight')
plt.ylabel('height')
plt.show()

#%% [markdown]
# ### Code 4.70
#%%
d['weight_sd3'] = d.weight_sdd**3

with pm.Model() as m4_70:
    a = pm.Normal('alpha',mu = 178, sigma=100)
    b1 = pm.Normal('beta1',mu = 0, sigma=10)
    b2 = pm.Normal('beta2',mu = 0, sigma=10)
    b3 = pm.Normal('beta3',mu = 0, sigma=10)
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    mu =  pm.Deterministic('mu', a + b1*d.weight_sd + b2*d.weight_sd2 + b3*d.weight_sd3)
    height = pm.Normal('height', mu=mu, sd=sigma, observed=d.height)
    trace470 = pm.sample(1000, tune=1000)


#%%
pm.summary(trace470, var_names = ['alpha', 'beta1','beta2','beta3','sigma'])

#%%
idx = np.argsort(d.weight_sd)
height_samps = pm.sample_posterior_predictive(trace470, samples = 1000,model = m4_70)
mu_pred = trace470['mu']
mu_hpd = az.hpd(mu_pred, credible_interval=.89)[idx]
height_hpd = az.hpd(height_samps['height'], credible_interval=.89)
height_hpd_sort = np.sort(height_hpd, axis=0)
#%%
# the jaggedness is the simulation variance in the tails of the distrubtion
plt.plot(d.weight_sd, d.height, marker = '.', linestyle = '', alpha = .5)
plt.fill_between(d.weight_sd[idx], mu_hpd[:,0], mu_hpd[:,1], color='C2', alpha=0.7)
plt.fill_between(d.weight_sd[idx],height_hpd_sort[:,0],height_hpd_sort[:,1], color = 'white', alpha = .2 )
plt.xlabel('weight')
plt.ylabel('height')
plt.show()



#%% [markdown]
# # Practice Questions
# ## Medium
# ### 4M1
#%%
# simulating observations for the prior
nums = int(1e4)
mu = np.random.normal(0,10,size = nums)
sigma = np.random.uniform(0,10,size = nums)
y_sim = np.random.normal(mu, sigma , size = nums)
#%% [markdown]
# ### 4M2
#%%
with pm.Model() as m4M1:
    mu = pm.Normal('mu', mu=0, sd=10)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    y = pm.Normal('y', mu=mu, sd=sigma, observed = y_sim)
    trace4M1 = pm.sample(1000, tune=1000)

pm.traceplot(trace4M1, varnames = ['mu','sigma'])
pm.summary(trace4M1)

#%% [markdown]
# # Practice Questions
# ## Hard
# ### 4H1
#%%
d = pd.read_csv('.\data\Howell1.csv', sep = ";")
d['weight_sd'] = (d.weight - d.weight.mean())/d.weight.std()
d.head()

shared_x = tt.shared(d.weight_sd.values)
shared_y = tt.shared(d.height.values)

#%%
with pm.Model() as m4h1:
    a = pm.Normal('alpha',mu = 178, sigma=100)
    b = pm.Normal('beta',mu = 0, sigma=10)
    mu = a + b*shared_x
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    height = pm.Normal('height', mu=mu, sd=sigma, observed=shared_y)
    tracem4H1 = pm.sample(1000, tune=1000)

pm.traceplot(tracem4H1, varnames = ['mu','sigma', 'beta','alpha'])
pm.summary(tracem4H1)
#%%
# predicting out of sample via PYMC3
# pymc3 or better yet theano assumes that when you pass data into a model you are giving Theano 
# permission to keep the data constant and optimize as it sees fit
# https://docs.pymc.io/notebooks/api_quickstart.html#4.1-Predicting-on-hold-out-data
ind_weights = np.array([46.95,43.72,64.78,32.59,54.63])
ind_weight_cent = (ind_weights - d.weight.mean())/d.weight.std()
shared_x.set_value(ind_weight_cent)
shared_y.set_value(np.repeat(0, repeats = len(ind_weight_cent)))

#%%
with m4h1:
    post_pred = pm.sample_posterior_predictive(tracem4H1,samples = 400,model=m4h1)
height_hpd = az.hpd(post_pred['height'], credible_interval= 0.89)
post_pred['height'].mean(axis = 0)
del post_pred 

#%% [markdown]
# ## Hard
# ### 4H2
#%%
d_lt18 = d.query('age < 18')
d_lt18.weight.values
#%%
shared_x = tt.shared(d_lt18.weight.values)
shared_y = tt.shared(d_lt18.height.values)

#%%
with pm.Model() as m4h2:
    a = pm.Normal('alpha',mu = 50, sigma=100)
    b = pm.Normal('beta',mu = 0, sigma=10)
    mu = pm.Deterministic('mu',a + b*shared_x)
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    height = pm.Normal('height', mu=mu, sd=sigma, observed=shared_y)
    tracem4H2 = pm.sample(1000, tune=1000)

pm.traceplot(tracem4H2, varnames = ['sigma', 'beta','alpha'])
pm.summary(tracem4H2,varnames = ['sigma', 'beta','alpha'])

#%%
#4H2a - A child gets roughly 27
print(f'for every 10 units of increase in weight an individual will be {tracem4H2["beta"].mean()*10} taller')

#%%
#4H2b
pred_height_samps = pm.sample_posterior_predictive(tracem4H2, samples = 2000, model = m4h2)

#%%
mu_pred = tracem4H2['mu']
mu_pred_sort = np.sort(
    az.hpd(mu_pred, credible_interval = .89), axis = 0)
height_hpd_sort = np.sort(
    az.hpd(pred_height_samps['height'], credible_interval = .89), axis = 0)

#%%
plt.figure(figsize = (10,10))
plt.plot(d_lt18.weight, d_lt18.height, marker = '.', linestyle = '', alpha = .5, color = 'orange')
plt.plot(d_lt18.weight, tracem4H2['alpha'].mean() + tracem4H2['beta'].mean()*d_lt18.weight, color = 'white')
plt.fill_between(np.sort(d_lt18.weight), mu_pred_sort[:,0], mu_pred_sort[:,1], color='white', alpha=0.7)
plt.fill_between(np.sort(d_lt18.weight),height_hpd_sort[:,0], height_hpd_sort[:,1], color = 'grey' )
plt.ylabel('height')
plt.xlabel('weight')
plt.show()

#%% [markdown]
# ##4H2c
# The aspects of this that concern me are we are assuming the relationship is linear when it appears to be non-linear
# Adding in a non-linear term to the model would help improve the model


#%% [markdown]
# ## Hard
# ### 4H3

#%%
shared_x = tt.shared(d.weight.values)
shared_y = tt.shared(d.height.values)

#%%
with pm.Model() as m4h3:
    a = pm.Normal('alpha',mu = 50, sigma=100)
    b = pm.Normal('beta',mu = 0, sigma=10)
    mu = pm.Deterministic('mu',a + b*np.log(shared_x))
    sigma = pm.Uniform('sigma', lower=0, upper=50)
    height = pm.Normal('height', mu=mu, sd=sigma, observed=shared_y)
    tracem4H3 = pm.sample(1000, tune=1000)

pm.traceplot(tracem4H3, varnames = ['sigma', 'beta','alpha'])
pm.summary(tracem4H3,varnames = ['sigma', 'beta','alpha'])

#%%
mu_pred = tracem4H3['mu']
mu_pred_sort = np.sort(
    az.hpd(mu_pred, credible_interval = .97), axis = 0)

pred_height_samps = pm.sample_posterior_predictive(tracem4H3, samples = 2000, model = m4h3)
height_hpd_sort = np.sort(
    az.hpd(pred_height_samps['height'], credible_interval = .97), axis = 0)
#%%
plt.figure(figsize=(10,8))
plt.plot(d.weight, d.height, color = 'orange', marker = '.', linestyle = '')
plt.plot(d.weight.sort_values(), tracem4H3['alpha'].mean() + tracem4H3['beta'].mean()*np.log(d.weight.sort_values()), color = 'white', alpha = 1)
plt.fill_between(np.sort(d.weight), mu_pred_sort[:,0], mu_pred_sort[:,1], color='white', alpha=0.3)
plt.fill_between(np.sort(d.weight),height_hpd_sort[:,0], height_hpd_sort[:,1], color = 'grey' )
plt.xlabel('weight')
plt.ylabel('height')
plt.title('Howell Data with log transformed weight', fontsize = 18)
plt.show()



#%%
