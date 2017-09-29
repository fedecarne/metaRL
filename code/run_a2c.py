import os, datetime
import a2c
import makefigs

def run_a2c(gamma, rnn_size, nTrials, noise, task, model_folder, nEpisodes, l2l, reset, train, rnn_type, activation):

  do_train = True
  priors = task[0]
  rewards = task[1]

  if train:

    #########
    # Train #
    #########

    figs_folder =  model_folder + '/train/figs'
    if not os.path.exists(figs_folder):
      os.makedirs(figs_folder)

    log_folder =  model_folder + '/train/log'
    if not os.path.exists(log_folder):
      os.makedirs(log_folder)


    load_model = False

    num_episodes = nEpisodes[0] #How many episodes of game environment to train network with.

    trials_tosave_ = range(1,10)+range(10000,10010)+range(20000,20010)+range(30000,30010) + \
                     range(40000,40010)+range(50000,50010)+range(60000,60010) + \
                     range(70000,70010)+range(80000,80010)+range(90000,90010) + \
                     range(100000,100010)+range(110000,110010)+range(120000,120010) 

    trials_tosave = [trial for trial in trials_tosave_ if trial<num_episodes]

    params = {'num_episodes': num_episodes,
              'rnn_size':   rnn_size, 
              'rnn_type': rnn_type, 
              'activation': activation,
              'l2l':        l2l,
              'reset':        reset,
              'nTrials':    nTrials,
              'noise':      noise,
              'priors':      priors,
              'gamma':      gamma,
              'load_model':      load_model,
              'do_train':   do_train,
              'trials_tosave': trials_tosave,
              'rewards':      rewards,
              'model_folder':      model_folder,
              'log_folder':      log_folder}

    a2c.a2c(**params)

    # Makefigs 
    makefigs.plot_reward(log_folder+'/all_episodes.csv',figs_folder)
    trials_toplot = trials_tosave 
    for i in range(len(trials_toplot)):
      makefigs.plot_trial(log_folder+'/episode' + str(trials_toplot[i]) + '.csv', trials_toplot[i],figs_folder)
      makefigs.plot_psycho(log_folder+'/episode' + str(trials_toplot[i]) + '.csv', trials_toplot[i],figs_folder)

 


  ########
  # Test #
  ########

  ########################################
  #1. Same task
  print 'Testing same task'


  #do_train = False
  load_model = True

  figs_folder =  model_folder + '/test-same/figs'
  if not os.path.exists(figs_folder):
    os.makedirs(figs_folder)

  log_folder =  model_folder + '/test-same/log'
  if not os.path.exists(log_folder):
    os.makedirs(log_folder)

  num_episodes = nEpisodes[1]

  trials_tosave = range(0,num_episodes,1)

  nTrials = [nTrials[0]-1,nTrials[0]]

  params = {'num_episodes': num_episodes,
            'rnn_size':   rnn_size,
            'rnn_type': rnn_type,
            'activation': activation, 
            'l2l':        l2l,
            'reset':        reset,  
            'nTrials':    nTrials,
            'noise':      noise,
            'priors':      priors,
            'gamma':      gamma,
            'load_model':      load_model,
            'do_train':   do_train,
            'trials_tosave': trials_tosave,
            'rewards':      rewards,
            'model_folder':      model_folder,
            'log_folder':      log_folder}

  a2c.a2c(**params)


  # Makefigs 
  makefigs.plot_reward(log_folder+'/all_episodes.csv',figs_folder)

  trials_toplot = range(0,5,1)
  for i in range(len(trials_toplot)):
    makefigs.plot_trial(log_folder+'/episode' + str(trials_toplot[i]) + '.csv', trials_toplot[i],figs_folder)
    makefigs.plot_psycho(log_folder+'/episode' + str(trials_toplot[i]) + '.csv', trials_toplot[i],figs_folder)
    #makefigs.plot_value(log_folder+'/episode' + str(trials_toplot[i]) + '.csv', trials_toplot[i],figs_folder)
    makefigs.plot_rate_time(log_folder+'/episode' + str(trials_toplot[i]) + '.csv', log_folder+'/activity' + str(trials_toplot[i]) + '.csv', trials_toplot[i],figs_folder)
    
  makefigs.plot_avg_psycho(trials_tosave,log_folder,figs_folder) 
  makefigs.plot_time_psycho(params,trials_tosave,log_folder,figs_folder) 
  makefigs.plot_tuning(params,trials_tosave,log_folder,figs_folder) 


  
  ########################################
  # 2. Generalization to other priors  
  print 'Testing generalization to other priors'


  priors = [0.9,0.7,0.5,0.4,0.3,0.1]
  rewards = [1.0, 1.0]

  figs_folder =  model_folder + '/test-general-prior/figs'
  if not os.path.exists(figs_folder):
    os.makedirs(figs_folder)

  log_folder =  model_folder + '/test-general-prior/logs'
  if not os.path.exists(log_folder):
    os.makedirs(log_folder)

  #do_train = False
  load_model = True

  num_episodes = nEpisodes[1]
  trials_tosave = range(0,num_episodes,1)

  nTrials = [nTrials[0]-1,nTrials[0]]

  params = {'num_episodes': num_episodes,
            'rnn_size':   rnn_size,
            'rnn_type': rnn_type, 
            'activation': activation,
            'l2l':        l2l,
            'reset':        reset,
            'nTrials':    nTrials,
            'noise':      noise,
            'priors':      priors,
            'gamma':      gamma,
            'load_model':      load_model,
            'do_train':   do_train,
            'trials_tosave': trials_tosave,
            'rewards':      rewards,
            'model_folder':      model_folder,
            'log_folder':      log_folder}

  a2c.a2c(**params)
  # Makefigs 

  makefigs.plot_reward(log_folder+'/all_episodes.csv',figs_folder)

  trials_toplot = range(0,5,1)
  for i in range(len(trials_toplot)):
    makefigs.plot_trial(log_folder+'/episode' + str(trials_toplot[i]) + '.csv', trials_toplot[i],figs_folder)
    makefigs.plot_psycho(log_folder+'/episode' + str(trials_toplot[i]) + '.csv', trials_toplot[i],figs_folder)
    #makefigs.plot_value(log_folder+'/episode' + str(trials_toplot[i]) + '.csv', trials_toplot[i],figs_folder)
    
  makefigs.plot_avg_psycho(trials_tosave,log_folder,figs_folder) 
  makefigs.plot_time_psycho(params,trials_tosave,log_folder,figs_folder) 

  
  ########################################
  #3. Generalization to other rewards
  print 'Testing generalization to other rewards'

  priors = [0.5,0.5]
  rewards = [3.2, 0.8]

  figs_folder =  model_folder + '/test-general-reward/figs'
  if not os.path.exists(figs_folder):
    os.makedirs(figs_folder)

  log_folder =  model_folder + '/test-general-reward/logs'
  if not os.path.exists(log_folder):
    os.makedirs(log_folder)

  #do_train = False
  load_model = True

  num_episodes = nEpisodes[1]
  trials_tosave = range(0,num_episodes,1)

  nTrials = [nTrials[0]-1,nTrials[0]]

  params = {'num_episodes': num_episodes,
            'rnn_size':   rnn_size, 
            'rnn_type': rnn_type, 
            'activation': activation,
            'l2l':        l2l,
            'reset':        reset,
            'nTrials':    nTrials,
            'noise':      noise,
            'priors':      priors,
            'gamma':      gamma,
            'load_model':      load_model,
            'do_train':   do_train,
            'trials_tosave': trials_tosave,
            'rewards':      rewards,
            'model_folder':      model_folder,
            'log_folder':      log_folder}

  a2c.a2c(**params)
  # Makefigs 

  makefigs.plot_reward(log_folder+'/all_episodes.csv',figs_folder)

  trials_toplot = range(0,5,1)
  for i in range(len(trials_toplot)):
    makefigs.plot_trial(log_folder+'/episode' + str(trials_toplot[i]) + '.csv', trials_toplot[i],figs_folder)
    makefigs.plot_psycho(log_folder+'/episode' + str(trials_toplot[i]) + '.csv', trials_toplot[i],figs_folder)
    #makefigs.plot_value(log_folder+'/episode' + str(trials_toplot[i]) + '.csv', trials_toplot[i],figs_folder)
    
  makefigs.plot_avg_psycho(trials_tosave,log_folder,figs_folder)
  makefigs.plot_time_psycho(params,trials_tosave,log_folder,figs_folder) 
  
