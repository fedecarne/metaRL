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

 
model_folder = '../model-09-28_16-51-ptask-relu'

with open(model_folder + '/params.txt', 'r') as f:
  json_str = f.read()
  params = json.loads(json_str)

params['model_folder'] = model_folder
params['train'] = False
params['nEpisodes'] = [0,1000]


# Run model
run_a2c.run_a2c(**params)

