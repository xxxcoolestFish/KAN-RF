"""Fast three-way KAN policy diagnosis with real-time progress."""
import sys, torch, numpy as np, json, os, tempfile, time
sys.path.insert(0, '.')
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from cpbn.protokan_sb3_policy import ProtoKANFeaturesExtractor
from scripts.prescreen_hopper_physics_shifts import SHIFTS, make_shifted_env
import gymnasium as gym

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_ENVS = 8

class ProgressCallback(BaseCallback):
    def __init__(self, total, label=""):
        super().__init__(verbose=0)
        self.total = total
        self.label = label
        self.start = time.time()
    def _on_step(self):
        pct = self.num_timesteps / self.total * 100
        elapsed = time.time() - self.start
        speed = self.num_timesteps / max(elapsed, 0.1)
        print(f"    {self.label} {self.num_timesteps}/{self.total} ({pct:.0f}%) "
              f"{speed:.0f} steps/s", flush=True)
        return True

class ZWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        s_dim = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(s_dim+3,), dtype=np.float32)
        self.z = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        return np.concatenate([obs, self.z]), info
    def step(self, action):
        obs, r, t, tr, info = self.env.step(action)
        return np.concatenate([obs, self.z]), r, t, tr, info

def run_exp(label, use_z=False, l2_lambda=0.0, steps=150000):
    torch.manual_seed(1811)
    print(f'  Creating {N_ENVS} envs...', flush=True, end=' ')
    if use_z:
        vec_env = DummyVecEnv([
            lambda i=i: ZWrapper(make_shifted_env(SHIFTS['source'], 1811+i, 'hopper')())
            for i in range(N_ENVS)
        ])
    else:
        vec_env = DummyVecEnv([
            lambda i=i: make_shifted_env(SHIFTS['source'], 1811+i, 'hopper')()
            for i in range(N_ENVS)
        ])
    vec_env = VecNormalize(vec_env, training=True, norm_obs=True, norm_reward=True)

    model = PPO('MlpPolicy', vec_env, n_steps=2048, batch_size=256,
        n_epochs=5, learning_rate=3e-4, device=DEVICE, verbose=0,
        policy_kwargs=dict(
            features_extractor_class=ProtoKANFeaturesExtractor,
            features_extractor_kwargs=dict(n_prototypes=16, out_dim=256, grid_range=1.0),
            net_arch=[]))
    print('training...', flush=True)
    model.learn(total_timesteps=steps, callback=ProgressCallback(steps, label))

    if l2_lambda > 0:
        opt = model.policy.optimizer
        for _ in range(50):
            l2 = sum((p**2).sum() for p in model.policy.features_extractor.parameters())
            (l2_lambda * l2).backward()
            opt.step()
            opt.zero_grad()

    with tempfile.TemporaryDirectory() as tmp:
        mpath, npath = os.path.join(tmp,'m'), os.path.join(tmp,'n.pkl')
        model.save(mpath); vec_env.save(npath)
        if use_z:
            base = DummyVecEnv([lambda: ZWrapper(make_shifted_env(SHIFTS['source'], 1911, 'hopper')())])
        else:
            base = DummyVecEnv([lambda: make_shifted_env(SHIFTS['source'], 1911, 'hopper')()])
        env_eval = VecNormalize.load(npath, base)
        env_eval.training = False; env_eval.norm_reward = False
        m2 = PPO.load(mpath, env=env_eval, device=DEVICE)
        returns = []
        for _ in range(10):
            obs = env_eval.reset(); total = 0.0
            while True:
                a, _ = m2.predict(obs, deterministic=True)
                obs, r, d, i = env_eval.step(a)
                total += float(r[0])
                if d[0]: break
            returns.append(total)
        env_eval.close()
    mean_r = float(np.mean(returns))
    print(f'  => {mean_r:.1f}', flush=True)
    return mean_r

if __name__ == '__main__':
    configs = [
        ('Exp1: ProtoKAN+L2, no z',  False, 5.0),
        ('Exp2: ProtoKAN, no z',     False, 0.0),
        ('Exp3: ProtoKAN + z',       True,  0.0),
    ]
    results = {}
    for label, use_z, l2 in configs:
        print(f'\n=== {label} ===', flush=True)
        results[label] = run_exp(label, use_z=use_z, l2_lambda=l2)

    print('\n========== RESULTS ==========', flush=True)
    for k, v in results.items():
        print(f'{k}: {v:.1f}', flush=True)
    json.dump(results, open('results/kan_diagnosis_3way.json', 'w'), indent=2)
