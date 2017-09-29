import itertools
import os, datetime
import run_a2c
import json

def get_param(dicts):
  """
  >>> list(get_param(dict(number=[1,2], character='ab')))
  [{'character': 'a', 'number': 1},
   {'character': 'a', 'number': 2},
   {'character': 'b', 'number': 1},
   {'character': 'b', 'number': 2}]
  """
  return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

 
param_list = {'nEpisodes':  [[120101,5000]],
              'rnn_size':   [64], 
              'rnn_type' :  ['lstm'],#['vanilla'],#,'lstm'],
              'activation': ['relu'],
              'l2l':        [True,False],
              'reset':      [True,False],
              'nTrials':    [[70,100]],
              'noise':      [0.1],
              'task':       [[[0.8,0.2],[1.0,1.0]],[[0.5, 0.5],[1.6, 0.4]]],
              'gamma':      [.80],
              'train':      [True]}

for i, params in enumerate(get_param(param_list)):

  # Create output folder with timestamp and parameter label
  if params['task'][0][0]!=params['task'][0][1]:
    task = 'ptask'
  else:
    task = 'rtask'
  
  foldername = 'model-%m-%d_%H-%M-'+ task + \
               '_' + params['activation'] + \
               '_l2l-' + str(params['l2l']) + \
               '_reset-' + str(params['reset'])

  model_folder = '../' + datetime.datetime.now().strftime(foldername)
  if not os.path.exists(model_folder):
    os.makedirs(model_folder)

  # Save parameters
  with open(model_folder + '/params.txt', 'w') as file:
    file.write(json.dumps(params))

  params['model_folder'] = model_folder

  # Run model
  run_a2c.run_a2c(**params)






