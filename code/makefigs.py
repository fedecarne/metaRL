import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import linear_model
from collections import defaultdict
from itertools import izip
from matplotlib.ticker import FormatStrFormatter
np.set_printoptions(threshold=np.inf)

plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 

def plot_reward(filename,figs_folder):

  def running_mean(x, N):
      cumsum = np.cumsum(np.insert(x, 0, 0)) 
      return (cumsum[N:] - cumsum[:-N]) / N 

  with open(filename) as csvfile:
    reader = csv.DictReader(csvfile,delimiter='\t')
    total_rew = []
    max_rew = []
    performance = []
    for row in reader:
      total_rew.append(float(row['total_rew']))
      max_rew.append(float(row['max_reward'])) 
      performance.append(100.0*float(row['total_rew'])/float(row['max_reward']))

    plt.figure(figsize=(8, 6))
    plt.plot(np.array(range(len(performance)))/1000.0, performance)
    N = 100
    smooth_performance = running_mean(performance, N)
    plt.plot(smooth_performance,'r')
    axes = plt.gca()
    axes.set_ylim([0,110])  
    plt.ylabel('Reward',size=20)      
    plt.xlabel('Trial # (K)',size=20)
    plt.savefig(figs_folder+'/total_rew.png')
    plt.close()  
      
          
def plot_trial(filename,i,figs_folder):
  
  with open(filename) as csvfile:
    reader = csv.DictReader(csvfile,delimiter='\t')
    h = []
    action = []
    stim = []
    correct = []
    value = []
    for row in reader:
      h.append(float(row['h']))
      action.append(float(row['a']))
      stim.append(float(row['s1']))
      correct.append(float(row['c']))
      value.append((float(row['v'])))
                 
    plt.figure(figsize=(9, 3))
    plt.plot(correct,'ro')
    plt.plot(action)
    plt.plot(stim)
    axes = plt.gca()
    axes.set_ylim([-0.1, 1.1])  
    plt.title(str(h[0]))
    plt.savefig(figs_folder+'/episode' + str(i) + '.png')
    plt.close()      
    

def plot_value(filename,i,figs_folder):

  with open(filename) as csvfile:
    reader = csv.DictReader(csvfile,delimiter='\t')
    action = []
    stim = []
    reward = []
    correct = []
    value = []
    for row in reader:
      action.append(float(row['a']))
      stim.append(float(row['s1']))
      reward.append(float(row['rew']))
      correct.append(float(row['c']))
      value.append((float(row['v'])))
           
    plt.scatter(stim, value, marker='o', facecolors='b')
    plt.savefig(figs_folder+'/value' + str(i) + '.png')
    plt.close()    
    

def plot_rate_time(episode_filename,activity_filename,episode_ind,figs_folder):

  neuron_toplot=[4,9, 13]
  n = len(neuron_toplot)  
  h = []
  action = []
  stim = []
  rate=[[] for k in range(n)]

  # loop over activity file  
  with open(activity_filename, 'r') as activity_file:
    activity_reader = csv.DictReader(activity_file,delimiter='\t')
    for row in activity_reader:
      for k in range(n):     
        rate[k].append(float(row[str(neuron_toplot[k])]))

  with open(episode_filename,'r') as episode_file:
    episode_reader = csv.DictReader(episode_file,delimiter='\t')
    for row in episode_reader:
      h.append(float(row['h']))
      action.append(float(row['a']))
      stim.append(float(row['s1']))
           
  for k in range(n):     
    plt.plot(rate[k])

  axes = plt.gca()
  plt.title('h: '+str(h[0]))
  plt.savefig(figs_folder+'/rate_time' + str(episode_ind) + '.png')
  plt.close()






def plot_tuning(params,episode_toplot,log_folder, figs_folder):
  

  #################################################################
  # plot activity in time

  clrs={0:'blue', 1: 'red'}

  neuron_toplot = range(64)
  selected_episodes = range(100)
  n = len(neuron_toplot) 
  
  rate = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
  
  for i in range(len(selected_episodes)):

    row_number =0
    with open(log_folder+"/episode" + str(selected_episodes[i]) + ".csv") as episode_file, \
         open(log_folder+"/activity" + str(selected_episodes[i]) + ".csv") as rate_file:
    
        episode_reader = csv.DictReader(episode_file,delimiter='\t')
        rate_reader = csv.DictReader(rate_file,delimiter='\t')

        for episode_row, rate_row in izip(episode_reader, rate_reader):
          
          h = int(episode_row['h'])

          for k in range(n): 
            rate[neuron_toplot[k]][h][i].append(float(rate_row[str(neuron_toplot[k])]))
          

  for neuron_rate in rate:

    figure, ax = plt.subplots(1, figsize=(4,4))

    for h in range(2):    
      for episode in rate[neuron_rate][h]:
        ax.plot(rate[neuron_rate][h][episode],color=clrs[h],linewidth=1,alpha=0.1)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('Firing rate',size=20)      
    ax.set_xlabel('Trial #',size=20)
    plt.tight_layout()
    plt.savefig(figs_folder+'/timecourse-' + str(neuron_rate) + '.png')
    plt.close()
    #################################################################


  neuron_toplot=range(64)
  n = len(neuron_toplot) 
  selected_episodes_ = range(1000)
  selected_episodes = [trial for trial in selected_episodes_ if trial<len(episode_toplot)]
  
  stim_rate = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

  for i in range(len(selected_episodes)):

    row_number =0
    with open(log_folder+"/episode" + str(selected_episodes[i]) + ".csv") as episode_file, \
         open(log_folder+"/activity" + str(selected_episodes[i]) + ".csv") as rate_file:
    
        episode_reader = csv.DictReader(episode_file,delimiter='\t')
        rate_reader = csv.DictReader(rate_file,delimiter='\t')

        for episode_row, rate_row in izip(episode_reader, rate_reader):
          
          row_number += 1

          stim = float(episode_row['s1'])
          action = float(episode_row['a'])
          trial_type = int(episode_row['h'])

          for k in range(n): 
            rate = float(rate_row[str(neuron_toplot[k])])
            stim_rate[neuron_toplot[k]][trial_type][row_number-1].append([stim,action,rate])
  

  #################################################################
  # plot tuning
  n_centers = 10
  bin_loweredge = np.linspace(0.0,0.9,n_centers)
  bin_center = bin_loweredge+0.05

  # plot stim rate for selected time points
  time_toplot = [0, 10, 40, 60]
  clrs={0:'blue', 1: 'red'}
  for neuron in stim_rate:

      figure, ax = plt.subplots(1,4, figsize=(16,4))
      ymin=[]
      ymax=[]
      for i,t in enumerate(time_toplot):

        if t in stim_rate[neuron][0].keys():
          
          for h in range(2):
            
            stim = [trial[0] for trial in stim_rate[neuron][h][t]]
            rate = [trial[2] for trial in stim_rate[neuron][h][t]]
            ax[i].scatter(stim,rate, marker='o', facecolors=clrs[h],alpha=0.1)
            
            tuning_count = dict(zip(range(n_centers), [[] for _ in range(n_centers)]))
            for s,r in zip(stim,rate):
              bin_index = np.digitize(s, bin_loweredge)-1
              if bin_index<0: # if evidence is less than 0, assign to zero bin 
                bin_index = 0
              tuning_count[bin_index].append(r)


            # plt average rate for binned stimuli
            tuning_curve = {'mean':[], 'std':[]}
            for bini in tuning_count:
              tuning_curve['mean'].append(np.mean(tuning_count[bini]))
              tuning_curve['std'].append(np.std(tuning_count[bini]))

            tuning_curve['mean'] = np.array(tuning_curve['mean'])
            tuning_curve['std'] = np.array(tuning_curve['std'])
            ax[i].plot(bin_center,tuning_curve['mean'],color=clrs[h],linewidth=3)
            ax[i].fill_between(bin_center, tuning_curve['mean']-tuning_curve['std']/2, tuning_curve['mean']+tuning_curve['std']/2, color=clrs[h], alpha=0.3)
            
          ax[i].set_xlim([-0.25,1.25])
          low, high = ax[i].get_ylim()
          ymin.append(low)
          ymax.append(high)
          ax[i].yaxis.set_major_formatter(FormatStrFormatter('%1.1f'))
          ax[i].set_title('t: ' + str(t),size=20)
      
      for a in ax:
        a.set_ylim([-0.05,max(ymax)])

      ax[0].set_ylabel('Firing rate',size=20)      
      ax[0].set_xlabel('Sensory evidence',size=20)
      plt.tight_layout()
      plt.savefig(figs_folder+'/tuning-' + str(neuron) + '.png')
      plt.close()
    #################################################################





  #################################################################
  # plot linear regression coefficients 

  lincoeff = defaultdict(lambda: defaultdict(list))

  for neuron in stim_rate:

    plt.figure(figsize=(8, 8))         

    for h in range(2):

      for t in stim_rate[neuron][0]:

        stim = [trial[0] for trial in stim_rate[neuron][h][t]]
        rate = [trial[2] for trial in stim_rate[neuron][h][t]]

        X = np.reshape(stim,[-1,1])
        y = rate

        m = linear_model.LinearRegression()
        m.fit(X,y)

        lincoeff[neuron][h].append([m.intercept_,m.coef_])

        beta0 = [item[0] for item in lincoeff[neuron][h]]
        beta1 = [item[1] for item in lincoeff[neuron][h]]

        if h==0:
          clrs = 'Blues'
        else:
          clrs = 'Reds'

        plt.scatter(np.array(beta0), np.array(beta1), c = 20+np.arange(len(beta0)), cmap = clrs)

    axes = plt.gca()
#    axes.set_xlim([-1.5,1.5])
#    axes.set_ylim([-1.5,1.5])        
    plt.xlabel('beta_0')
    plt.ylabel('beta_1')
    plt.savefig(figs_folder+'/lincoeff-' + str(neuron) + '.png')
    plt.close()
      

def plot_psycho(filename,episode,figs_folder):

    stim_action = defaultdict(list)
    
    with open(filename) as csvfile:
      reader = csv.DictReader(csvfile,delimiter='\t')

      for row in reader:

        stim = float(row['s1'])
        action = float(row['a'])

        stim_action[0].append([stim,action])

    psychoplot(stim_action, figs_folder,['psych' + str(episode)])
    


def plot_avg_psycho(episode_toplot,log_folder, figs_folder):
  
  stim_action = defaultdict(list)
  
  for i in range(len(episode_toplot)):
        
    with open(log_folder+"/episode" + str(episode_toplot[i]) + ".csv") as csvfile:
      reader = csv.DictReader(csvfile,delimiter='\t')
      for row in reader:
        stim = float(row['s1'])
        action = float(row['a'])
        trial_type = float(row['h'])
        
        stim_action[trial_type].append([stim,action])

  psychoplot(stim_action, figs_folder,['avg_psycho','bias'])


def plot_time_psycho(params,episode_toplot,log_folder, figs_folder):
  
  stim_action = defaultdict(defaultdict)

  for i in range(len(episode_toplot)):

    correct_action = []
        
    with open(log_folder+"/episode" + str(episode_toplot[i]) + ".csv") as csvfile:
      reader = csv.DictReader(csvfile,delimiter='\t')
      row_number = 0
      for row in reader:
        
        row_number += 1 

        stim = float(row['s1'])
        action = float(row['a'])
        trial_type = float(row['h'])
        correct_action.append(float(row['c']))

        if row_number-1 in stim_action[trial_type]:
          stim_action[trial_type][row_number-1].append([stim,action,np.average(correct_action)])
        else:
          stim_action[trial_type][row_number-1] = [[stim,action,np.average(correct_action)]]

        
  for k in stim_action.keys():
    psychoplot(stim_action[k], figs_folder,['time_psycho%d' % k,'time_bias%d' % k],params)


def psychoplot(stim_action, figs_folder, fig_name,params=None):
  '''
    Receives a dict stim_action. Each value is a list of [stimulus, action]
    Plots a psycho function for each key
  '''
  def model(x):
    return 1 / (1 + np.exp(-x))

  
  n_centers = 10
  bin_loweredge = np.linspace(0.0,0.9,n_centers)
  bin_center = bin_loweredge+0.05

  prior = {}
  action_count = {}
  psychometric = {}  
  n = {}
  linreg = {}

  for key in stim_action:

    prior[key] = []
    action_count[key] = np.zeros(n_centers)
    psychometric[key] = np.zeros(n_centers)
    n[key] = np.zeros(n_centers)

    for item in stim_action[key]:
        
        stim = item[0]
        action = item[1]

        bin_index = np.digitize(stim, bin_loweredge)-1
        if bin_index<0: # if evidence is less than 0, assign to zero bin 
          bin_index = 0

        action_count[key][bin_index] += action
        n[key][bin_index] += 1.0

        if len(item)>2:
          prior[key].append(item[2])

         
    for bin_index in range(len(psychometric[key])):
      
      if n[key][bin_index] != 0:
        psychometric[key][bin_index] = action_count[key][bin_index]/n[key][bin_index]
      else:
        psychometric[key][bin_index] = float('nan')


    linreg[key] = linear_model.LogisticRegression()

    X = np.reshape(np.transpose(stim_action[key])[0],[-1,1])
    y = np.transpose(stim_action[key])[1]

    if len(np.unique(y))>1:  
      linreg[key].fit(X,y)
    else:
      linreg[key] = []
      print 'Unable to fit logistic regression, only one action'
  


  ###########################################33
  X_test = np.linspace(0, 1, 100)
  plt.figure(figsize=(5,5))    
    
  color_idx = np.linspace(0, 1, len(psychometric))
  for k, p in psychometric.items():
    #plot psycho
    clr = plt.cm.winter(color_idx[int(k)])
    plt.plot(bin_center,p,color=clr,linestyle='None',marker='o',markersize=10)

    # plot linear regression 
    if linreg[k] != []:
      linreg_psycho = model(X_test * linreg[k].coef_ + linreg[k].intercept_).ravel()    
      plt.plot(X_test, linreg_psycho, color=clr, linewidth=5)

  axes = plt.gca()
  axes.set_xlim([-0.1,1.1])
  axes.set_ylim([-0.1,1.1])
  axes.set_xlabel('Sensory evidence',fontsize=25)
  axes.set_ylabel('P(a=1)',fontsize=25)
  plt.tight_layout()
  plt.savefig(figs_folder+ '/' + fig_name[0] + '.png')
  plt.close()


  if len(fig_name)>1:
    # plot bias in a separate figure
    plt.figure(figsize=(5,5))  

    for k in linreg.keys():
      
      if linreg[k] != []:
        bias = model(0.5 * linreg[k].coef_ + linreg[k].intercept_).ravel()    
        #bias = (0.5 - linreg[k].intercept_)/linreg[k].coef_
        plt.scatter(int(k), bias, marker='o', facecolors=plt.cm.winter(color_idx[int(k)]),s=300)
        #plt.scatter(int(k),  np.average(prior[k]))
    if params!=None:   
      for v in params['priors']:
        plt.plot([0,100], [v,v], linestyle='--', color='black', linewidth=3)

    axes = plt.gca()
    axes.set_ylim([-0.1,1.1])
    plt.tight_layout()
    plt.savefig(figs_folder+'/' + fig_name[1] + '.png')
    plt.close()
  
  #-----------
