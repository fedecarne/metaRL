import os, datetime
import numpy as np
import copy
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
import time

import makefigs
import akitask

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer
        
    
class AC_Network():
  def __init__(self,a_size,rnn_size, rnn_type = 'vanilla', activation='tanh'):
    
    rnn_types = ['vanilla', 'lstm']
    if rnn_type not in rnn_types:
      raise ValueError("Invalid rnn type. Expected one of: %s" % rnn_types)

    activations = ['tanh', 'relu']
    if activation not in activations:
      raise ValueError("Invalid activation type. Expected one of: %s" % activations)

    #Input and visual encoding layers
    self.state = tf.placeholder(shape=[None,1],dtype=tf.float32)
    self.prev_rewards = tf.placeholder(shape=[None,1],dtype=tf.float32)
    self.prev_actions = tf.placeholder(shape=[None],dtype=tf.int32)
    self.timestep = tf.placeholder(shape=[None,1],dtype=tf.float32)

    #Transforms an analog variable into a one_hot vector of size a_size
    self.prev_actions_onehot = tf.one_hot(self.prev_actions,a_size,dtype=tf.float32)

    #Concatenates inputs (flattening the state input)
    hidden = tf.concat([slim.flatten(self.state),self.prev_rewards,self.prev_actions_onehot,self.timestep],1)

    #Recurrent network for temporal dependencies

    if rnn_type == 'lstm':

      ########
      # LSTM
      lstm_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size,state_is_tuple=True)
      c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
      h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
      self.state_init = [c_init, h_init]
      c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
      h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
      self.state_in = (c_in, h_in)
      rnn_in = tf.expand_dims(hidden, [0])
      step_size = tf.shape(self.prev_rewards)[:1]
      state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
      lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
          lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
          time_major=False)
      lstm_c, lstm_h = lstm_state
      self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
      rnn_out = tf.reshape(lstm_outputs, [-1, rnn_size])

    elif rnn_type == 'vanilla':

      ########
      # RNN
      if activation == 'tanh':
        rnn_cell = tf.contrib.rnn.BasicRNNCell(rnn_size)
      elif activation == 'relu':
        rnn_cell = tf.contrib.rnn.BasicRNNCell(rnn_size, activation=tf.nn.relu)

      c_init = np.zeros((1, rnn_cell.state_size), np.float32)
      h_init = np.zeros((1, rnn_cell.state_size), np.float32)
      self.state_init = [c_init, h_init]
      c_in = tf.placeholder(tf.float32, [1, rnn_cell.state_size])
      h_in = tf.placeholder(tf.float32, [1, rnn_cell.state_size])
      self.state_in = (c_in, h_in)
      rnn_in = tf.expand_dims(hidden, [0])
      step_size = tf.shape(self.prev_rewards)[:1]
      state_in = c_in
      rnn_outputs, last_states = tf.nn.dynamic_rnn(rnn_cell, 
          rnn_in, initial_state=state_in, sequence_length=step_size, time_major=False)
      self.state_out = (last_states[:1, :], last_states[:1, :])
      rnn_out = tf.reshape(rnn_outputs, [-1, rnn_size])



    self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
    self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)

    #Output layers for policy and value estimations
    self.policy = slim.fully_connected(rnn_out,a_size,
        activation_fn=tf.nn.softmax,
        weights_initializer=normalized_columns_initializer(0.01),
        biases_initializer=None)

    self.value = slim.fully_connected(rnn_out,1,
        activation_fn=None,
        weights_initializer=normalized_columns_initializer(1.0),
        biases_initializer=None)

    self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
    self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)

    self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

    #Loss functions
    self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
    self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 1e-7))
    self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs + 1e-7)*self.advantages)
    self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.05

    self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
    self.updateModel = self.trainer.minimize(self.loss)         
        
def train(ACnet,rollout,sess,gamma,bootstrap_value):
    rollout = np.array(rollout)
    states = rollout[:,0]
    actions = rollout[:,1]
    rewards = rollout[:,2]
    timesteps = rollout[:,3]
    prev_rewards = [0] + rewards[:-1].tolist()
    prev_actions = [0] + actions[:-1].tolist()
    values = rollout[:,5]

    pr = prev_rewards
    pa = prev_actions
    
    # Here we take the rewards and values from the rollout, and use them to 
    # generate the advantage and discounted returns. 
    # The advantage function uses "Generalized Advantage Estimation"
    rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
    discounted_rewards = discount(rewards_plus,gamma)[:-1]
    value_plus = np.asarray(values.tolist() + [bootstrap_value])
    advantages = rewards + gamma * value_plus[1:] - value_plus[:-1]
    advantages = discount(advantages,gamma)

    # Update the global network using gradients from loss
    # Generate network statistics to periodically save
    rnn_state = ACnet.state_init

    feed_dict = {ACnet.target_v:discounted_rewards,
        ACnet.state:np.stack(states,axis=0),
        ACnet.prev_rewards:np.vstack(prev_rewards),
        ACnet.prev_actions:prev_actions,
        ACnet.actions:actions,
        ACnet.timestep:np.vstack(timesteps),
        ACnet.advantages:advantages,
        ACnet.state_in[0]:rnn_state[0],
        ACnet.state_in[1]:rnn_state[1]}

    _ = sess.run([ACnet.updateModel],feed_dict=feed_dict)
    
    return
    
def a2c(gamma, rnn_size, load_model, do_train, num_episodes, nTrials, trials_tosave, noise, priors, rewards, model_folder, log_folder, l2l, reset, rnn_type, activation):

  # Fixed parameters
  a_size = 2 
  do_makefigs = True
  verbose = 1

  model_path = model_folder+'/model'
  if do_train:
    if not os.path.exists(model_path):
      os.makedirs(model_path)

  ## Task
  env = akitask.akitask(nTrials,noise,priors,rewards)


  tf.reset_default_graph()

  ACnet = AC_Network(a_size,rnn_size, rnn_type,activation) # Generate global network

  # Create a saver for saving checkpoints
  saver = tf.train.Saver(max_to_keep=5)


  total_episode_rewards = []
  total_episode_max_rewards = []
  total_episode_lengths = []
  total_episode_mean_values = []
      
  episodefile = open(log_folder +'/all_episodes.csv', 'w')
  episodefile.write("total_rew\tmax_reward\n")

      
  with tf.Session() as sess:

      if load_model == True:
        print 'Loading Model...'
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)

      else:
        sess.run(tf.global_variables_initializer())
          
      if reset==False:
        rnn_state = ACnet.state_init # this I change to remove inter-episode reset
        r = 0 # this I change to remove inter-episode reset
        a = 0 # this I change to remove inter-episode reset
        t = 0 # this I change to remove inter-episode reset 
      

      # Loop over episodes    
      for episode_count in range(num_episodes):

          if episode_count % 100 == 0 and verbose>0:
            start_time = time.time()

          if episode_count in trials_tosave:          
            trialfile = open(log_folder + '/episode' + str(episode_count) + '.csv', 'w')
            trialfile.write("h\ts1\ta\trew\tc\tv\n")
            activityfile = open(log_folder + '/activity' + str(episode_count) + '.csv', 'w')
            header = ''.join(str(i)+'\t' for i in range(rnn_size))
            activityfile.write(header[:-1]+'\n')

          episode_buffer = []
          episode_values = []
          episode_frames = []
          episode_reward = 0
          episode_max_reward = 0
          episode_step_count = 0
          
          #HOY

          if reset==True:
            rnn_state = ACnet.state_init # this I change to remove inter-episode reset
            r = 0 # this I change to remove inter-episode reset
            a = 0 # this I change to remove inter-episode reset
            t = 0 # this I change to remove inter-episode reset

          # start new episode
          s1,c1 = env.reset()
          d = False

          while d == False:
              
              s = s1
              c = c1

              if l2l: # send the previous reward
                a_dist,v,rnn_state_new = sess.run([ACnet.policy, ACnet.value, ACnet.state_out], 
                    feed_dict={
                    ACnet.state:[s],
                    ACnet.prev_rewards:[[r]],
                    ACnet.timestep:[[t]],
                    ACnet.prev_actions:[a],
                    ACnet.state_in[0]:rnn_state[0],
                    ACnet.state_in[1]:rnn_state[1]})

              else: # don't send the previous reward      
                a_dist,v,rnn_state_new = sess.run([ACnet.policy, ACnet.value, ACnet.state_out], 
                    feed_dict={
                    ACnet.state:[s],
                    ACnet.prev_rewards:[[0]],
                    ACnet.timestep:[[t]],
                    ACnet.prev_actions:[0],
                    ACnet.state_in[0]:rnn_state[0],
                    ACnet.state_in[1]:rnn_state[1]})
        

              #Take an action using probabilities from policy network output.
              a = np.random.choice(a_dist[0],p=a_dist[0])
              a = np.argmax(a_dist == a)

              rnn_state = rnn_state_new

              s1,r,d,t,c1 = env.pullArm(a)
              episode_buffer.append([s,a,r,t,d,v[0,0]])
              episode_values.append(v[0,0])
              episode_reward += r
              episode_max_reward += rewards[c1]
              episode_step_count += 1
              
              if episode_count in trials_tosave:
                trialfile.write(str(env.h)+"\t")
                trialfile.write("%1.2f" % s[0] + "\t")
                trialfile.write(str(a)+"\t")
                trialfile.write(str(r)+"\t")
                trialfile.write(str(c)+"\t")
                trialfile.write(str(v[0][0]))
                trialfile.write("\n")

                np.savetxt(activityfile, rnn_state[0], delimiter="\t")


          if episode_count in trials_tosave:                        
            trialfile.close()
            activityfile.close()
          
          total_episode_rewards.append(episode_reward)
          total_episode_max_rewards.append(episode_max_reward)
          total_episode_lengths.append(episode_step_count)
          total_episode_mean_values.append(np.mean(episode_values))

          # Update the network using the experience buffer at the end of the episode.
          # (This takes time)
          if len(episode_buffer) != 0 and do_train == True:          
            train(ACnet,episode_buffer,sess,gamma,0.0)

          ## Write episode data to file
          ## This has the state, action and reward of the last trial of the episode
          ## plus the total collected reward for that episode
          episodefile.write(str(total_episode_rewards[episode_count]))
          episodefile.write("\t")
          episodefile.write(str(total_episode_max_rewards[episode_count]))
          episodefile.write("\n")
                                                        
          mean_reward = np.mean(total_episode_rewards[-10:])
          mean_length = np.mean(total_episode_lengths[-10:])
          mean_value = np.mean(total_episode_mean_values[-10:])

          ## Print on screen
          if episode_count % 100 == 0 and verbose>0:
            print "Episode " + str(episode_count) + ': ' + str(total_episode_rewards[episode_count])


          ## saving model takes time, so only every 10000
          if do_train and episode_count % 10000 == 0 and verbose>0:
            saver.save(sess,model_path+'/model-'+str(episode_count)+'.cptk')


      episodefile.close()
     

