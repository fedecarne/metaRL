import os, datetime
import run_a2c




nEpisodes = {'forTraining': 150101,
             'forTesting': 5000}


########################################################################################
rnn_size = 64


###########
label = '70-100-ptask-64-l2l'
l2l = True;
nTrials = [70,100]
noise = 0.1
priors = [0.8,0.2]
rewards = [1.0, 1.0]
gamma = .80

model_folder = os.path.join(os.getcwd(), datetime.datetime.now().strftime('model-%m-%d_%H-%M-'+label))
if not os.path.exists(model_folder):
  os.makedirs(model_folder)

# use trained model
#model_folder = 'model-09-18_18-10-70-100-ptask-gamma0.8-l2l'

run_a2c.run_a2c(gamma, rnn_size, nTrials, noise, priors, rewards, label,model_folder,nEpisodes, l2l)



###########

label = '70-100-rtask-64-l2l'
l2l = True;
nTrials = [70,100]
noise = 0.1
priors = [0.5,0.5]
rewards = [1.6, 0.4]
gamma = .80

model_folder = os.path.join(os.getcwd(), datetime.datetime.now().strftime('model-%m-%d_%H-%M-'+label))
if not os.path.exists(model_folder):
  os.makedirs(model_folder)

# use trained model
#model_folder = 'model-09-18_22-38-70-100-rtask-gamma0.8-l2l'


run_a2c.run_a2c(gamma, rnn_size, nTrials, noise, priors, rewards, label,model_folder,nEpisodes, l2l)


########################################################################################
rnn_size = 32


###########
label = '70-100-ptask-32-l2l'
l2l = True;
nTrials = [70,100]
noise = 0.1
priors = [0.8,0.2]
rewards = [1.0, 1.0]
gamma = .80

model_folder = os.path.join(os.getcwd(), datetime.datetime.now().strftime('model-%m-%d_%H-%M-'+label))
if not os.path.exists(model_folder):
  os.makedirs(model_folder)

# use trained model
#model_folder = 'model-09-18_18-10-70-100-ptask-gamma0.8-l2l'

run_a2c.run_a2c(gamma, rnn_size, nTrials, noise, priors, rewards, label,model_folder,nEpisodes, l2l)



###########

label = '70-100-rtask-32-l2l'
l2l = True;
nTrials = [70,100]
noise = 0.1
priors = [0.5,0.5]
rewards = [1.6, 0.4]
gamma = .80

model_folder = os.path.join(os.getcwd(), datetime.datetime.now().strftime('model-%m-%d_%H-%M-'+label))
if not os.path.exists(model_folder):
  os.makedirs(model_folder)

# use trained model
#model_folder = 'model-09-18_22-38-70-100-rtask-gamma0.8-l2l'


run_a2c.run_a2c(gamma, rnn_size, nTrials, noise, priors, rewards, label,model_folder,nEpisodes, l2l)

########################################################################################
rnn_size = 16


###########
label = '70-100-ptask-16-l2l'
l2l = True;
nTrials = [70,100]
noise = 0.1
priors = [0.8,0.2]
rewards = [1.0, 1.0]
gamma = .80

model_folder = os.path.join(os.getcwd(), datetime.datetime.now().strftime('model-%m-%d_%H-%M-'+label))
if not os.path.exists(model_folder):
  os.makedirs(model_folder)

# use trained model
#model_folder = 'model-09-18_18-10-70-100-ptask-gamma0.8-l2l'

run_a2c.run_a2c(gamma, rnn_size, nTrials, noise, priors, rewards, label,model_folder,nEpisodes, l2l)



###########

label = '70-100-rtask-16-l2l'
l2l = True;
nTrials = [70,100]
noise = 0.1
priors = [0.5,0.5]
rewards = [1.6, 0.4]
gamma = .80

model_folder = os.path.join(os.getcwd(), datetime.datetime.now().strftime('model-%m-%d_%H-%M-'+label))
if not os.path.exists(model_folder):
  os.makedirs(model_folder)

# use trained model
#model_folder = 'model-09-18_22-38-70-100-rtask-gamma0.8-l2l'


run_a2c.run_a2c(gamma, rnn_size, nTrials, noise, priors, rewards, label,model_folder,nEpisodes, l2l)
