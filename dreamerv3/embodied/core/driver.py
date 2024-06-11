import collections

import numpy as np

from .basics import convert
from copy import deepcopy

def transform_single(x, proj, bias):
    x = x.reshape(1, -1)
    hk = np.dot(x, proj)
    hk = np.tanh(hk)
    hk += bias
    hk = 1 * (hk > 0.5) - 1 * (hk < -0.5)
    return tuple(hk.squeeze().tolist())

# Obtained from C-BET
# Expects X is be batched even if it is a single observation
# Always returns a batched output
def _hash_key(x, proj=None, bias=None):
  
  if proj is None:
    return [tuple(obs.flatten().tolist()) for obs in x]
  
  return [transform_single(obs, proj, bias) for obs in x]

class Driver:

  _CONVERSION = {
      np.floating: np.float32,
      np.signedinteger: np.int32,
      np.uint8: np.uint8,
      bool: bool,
  }

  def __init__(self, env, use_intrinsic_reward=False, use_pseudocounts=False, hash_bits=128, intr_reward_coeff=0.001, ignore_extr_reward=False, **kwargs):
    assert len(env) > 0
    self._env = env
    self._kwargs = kwargs
    self._on_steps = []
    self._on_episodes = []
    self.count_reset_prob = 0.001
    self.intr_reward_coeff = intr_reward_coeff
    self.state_counts = dict()
    self.change_counts = dict()
    self.use_intrinsic_reward = use_intrinsic_reward
    self.ignore_extrinsic_reward = ignore_extr_reward
    
    proj_size = np.prod(env.obs_space['image'].shape)
    proj_dim = hash_bits
    self.state_proj = self.state_bias = self.change_proj = self.change_bias = None
    
    if use_pseudocounts and use_intrinsic_reward:
      self.state_proj = np.random.normal(0, 1, (proj_dim, proj_size, 1))
      self.state_bias = np.random.uniform(-1, 1, (proj_dim, 1))
      self.change_proj = np.random.normal(0, 1, (proj_dim, proj_size, 1))
      self.change_bias = np.random.uniform(-1, 1, (proj_dim, 1))
    
    self.reset()

  def reset(self):
    self._acts = {
        k: convert(np.zeros((len(self._env),) + v.shape, v.dtype))
        for k, v in self._env.act_space.items()}
    self._acts['reset'] = np.ones(len(self._env), bool)
    self._eps = [collections.defaultdict(list) for _ in range(len(self._env))]
    self._state = None
    
    # Used for intrinsic reward
    self._current_obs_img = np.zeros((len(self._env),) + self._env.obs_space['image'].shape, self._env.obs_space['image'].dtype)

  def on_step(self, callback):
    self._on_steps.append(callback)

  def on_episode(self, callback):
    self._on_episodes.append(callback)

  def __call__(self, policy, steps=0, episodes=0):
    step, episode = 0, 0
    while step < steps or episode < episodes:
      step, episode = self._step(policy, step, episode)

  def _step(self, policy, step, episode):
    assert all(len(x) == len(self._env) for x in self._acts.values())
    acts = {k: v for k, v in self._acts.items() if not k.startswith('log_')}
    obs = self._env.step(acts)
    obs = {k: convert(v) for k, v in obs.items()}
    assert all(len(x) == len(self._env) for x in obs.values()), obs
    
    # Calculate intrinsic reward
    if self.use_intrinsic_reward:
      obs = self.calc_intrinsic_reward(obs)
    
    self._current_obs_img = obs['image']
    
    acts, self._state = policy(obs, self._state, **self._kwargs)
    acts = {k: convert(v) for k, v in acts.items()}
    if obs['is_last'].any():
      mask = 1 - obs['is_last']
      acts = {k: v * self._expand(mask, len(v.shape)) for k, v in acts.items()}
    acts['reset'] = obs['is_last'].copy()
    self._acts = acts
    trns = {**obs, **acts}
    if obs['is_first'].any():
      for i, first in enumerate(obs['is_first']):
        if first:
          self._eps[i].clear()
    for i in range(len(self._env)):
      trn = {k: v[i] for k, v in trns.items()}
      [self._eps[i][k].append(v) for k, v in trn.items()]
      [fn(trn, i, **self._kwargs) for fn in self._on_steps]
      step += 1
    if obs['is_last'].any():
      for i, done in enumerate(obs['is_last']):
        if done:
          ep = {k: convert(v) for k, v in self._eps[i].items()}
          [fn(ep.copy(), i, **self._kwargs) for fn in self._on_episodes]
          episode += 1
    return step, episode

  def _expand(self, value, dims):
    while len(value.shape) < dims:
      value = value[..., None]
    return value

  def calc_intrinsic_reward(self, obs):
        
    extr_reward = obs['reward']
    
    # Used during pretraining
    if self.ignore_extrinsic_reward:
      extr_reward = np.zeros_like(extr_reward)
    
    # Compute difference between next and current state
    state_diff = obs['image'] - self._current_obs_img
    
    # Update state counts
    self.update_counts(self.state_counts, _hash_key(obs['image'], self.state_proj, self.state_bias))
    self.update_counts(self.change_counts, _hash_key(state_diff, self.change_proj, self.change_bias))
    
    new_reward = np.zeros_like(extr_reward)

    intr_rewards = []
    
    for i in np.ndindex(extr_reward.shape):
        image_key_list = _hash_key([obs['image'][i]], self.state_proj, self.state_bias)
        state_diff_key_list = _hash_key([state_diff[i]], self.change_proj, self.change_bias)
        
        intr_reward = self.intr_reward_coeff / (self.state_counts[image_key_list[0]] + self.change_counts[state_diff_key_list[0]])
        new_reward[i] = extr_reward[i] + intr_reward
        intr_rewards.append(intr_reward)
    
    obs['reward'] = new_reward
    obs['intrinsic_reward'] = np.array(intr_rewards)
    
    # If random number is less than reset probability, reset the state count
    if np.random.rand() < self.count_reset_prob:
      self.state_counts.clear()
    if np.random.rand() < self.count_reset_prob:
      self.change_counts.clear()
    
    return obs
  
  def update_counts(self, var, keys):
    for key in keys:
        if key in var:
            var[key] += 1
        else:
            var[key] = 1