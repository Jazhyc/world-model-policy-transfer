import collections

import numpy as np

from .basics import convert


class Driver:

  _CONVERSION = {
      np.floating: np.float32,
      np.signedinteger: np.int32,
      np.uint8: np.uint8,
      bool: bool,
  }

  def __init__(self, env, mode='train', **kwargs):
    assert len(env) > 0
    self._env = env
    self._kwargs = kwargs
    self._on_steps = []
    self._on_episodes = []
    self.count_reset_prob = 0.001
    self.intr_reward_coeff = 0.005
    self.state_counts = dict()
    self.change_counts = dict()
    self.mode = mode
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
    if self.mode == 'train':
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
    
    # Compute difference between current and previous state
    state_diff = np.abs(self._current_obs_img - obs['image'])
    
    # Update state counts
    self.update_counts(self.state_counts, obs['image'])
    self.update_counts(self.change_counts, state_diff)
    
    new_reward = np.zeros_like(extr_reward)

    for i in np.ndindex(extr_reward.shape):
        image_key = obs['image'][i].tostring()
        state_diff_key = state_diff[i].tostring()
        
        new_reward[i] = extr_reward[i] + self.intr_reward_coeff * 1 / (self.state_counts[image_key] + self.change_counts[state_diff_key])
    
    obs['reward'] = new_reward
    
    # If random number is less than reset probability, reset the state count
    if np.random.rand() < self.count_reset_prob:
      self.state_counts.clear()
      self.change_counts.clear()
    
    return obs
  
  def update_counts(self, var, keys):
    for key in keys:
        
        # Use string for hashing
        key = key.tostring()
        if key in var:
            var[key] += 1
        else:
            var[key] = 1