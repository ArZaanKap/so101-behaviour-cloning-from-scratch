# il/data.py
import h5py
import numpy as np
import torch

DATA_PATH = "data/demos.hdf5"

# Which observation keys to feed the policy, concatenated in this order.

# (key, which columns). cube_1_pos drops z: blue never moves, so its height carries no signal.
OBS_SPEC = [
    ("joint_pos",   slice(None)),   # 6 arm joints + gripper opening, i.e. where the robot is
    ("tcp_pos",     slice(None)),   # 3 fingertip point
    ("cube_1_pos",  slice(0, 2)),   # 2 (x, y only) - where to place
    ("cube_2_pos",  slice(None)),   # 3 what to pick
    ("cube_2_quat", slice(None)),   # 4: the cube's rotation, needed to predict the gripper's yaw
]

def build_obs(d, cat):
    """d: dict key -> array (numpy here, torch at rollout). cat: how to concatenate a list."""
    return cat([d[k][..., sl] for k, sl in OBS_SPEC])



class Normalizer:
    """Per-dimension z-score: (x - mean) / std, with a floor on std."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @classmethod
    def fit(cls, x, min_std=1e-2):
        return cls(x.mean(axis=0), np.maximum(x.std(axis=0), min_std))

    def normalize(self, x):
        return (x - self.mean) / self.std

    def unnormalize(self, x):
        return x * self.std + self.mean

    def state_dict(self):    # saved inside your model checkpoint
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, d):
        return cls(d["mean"], d["std"])


def load_episodes(path=DATA_PATH):
    episodes = []
    with h5py.File(path, "r") as f:
        names = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
        for name in names:
            g = f["data"][name]
            obs = build_obs({k: g["obs"][k][:] for k, _ in OBS_SPEC}, lambda xs: np.concatenate(xs, axis=-1))
            act = g["actions"][:]
            episodes.append({"obs": obs.astype(np.float32), "act": act.astype(np.float32)})
    return episodes



def split_episodes(episodes, val_frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(episodes))
    n_val = int(len(episodes) * val_frac)
    val = [episodes[i] for i in idx[:n_val]]
    train = [episodes[i] for i in idx[n_val:]]
    return train, val


class ChunkDataset(torch.utils.data.Dataset):

    def __init__(self, episodes, obs_norm, act_norm, chunk=1):
        self.chunk = chunk
        self.obs = [torch.from_numpy(obs_norm.normalize)]



if __name__ == "__main__":
    eps = load_episodes()
    train, val = split_episodes(eps)
    train_obs = np.concatenate([e["obs"] for e in train], axis=0)
    train_act = np.concatenate([e["act"] for e in train], axis=0)
    
    print(train_obs.shape, train_act.shape)

    lens = [len(e["act"]) for e in eps]


    obs_norm = Normalizer.fit(train_obs)
    act_norm = Normalizer.fit(train_act)
    np.set_printoptions(precision=3, suppress=True)
    print("act mean", act_norm.mean)
    print("act std ", act_norm.std)

    val_obs = np.concatenate([e["obs"] for e in val], axis=0)
    val_obs_n = obs_norm.normalize(val_obs)
    for i in range(val_obs.shape[1]):
        print(i, f"train std {obs_norm.std[i]:.5f}   max|val norm| {np.abs(val_obs_n[:, i]).max():.1f}")


    print(f"{len(eps)} episodes | obs_dim {eps[0]['obs'].shape[1]} | act_dim {eps[0]['act'].shape[1]}")
    print(f"length min/mean/max: {min(lens)} / {np.mean(lens):.1f} / {max(lens)}")
    print(f"train {len(train)} eps ({sum(len(e['act']) for e in train)} steps) | val {len(val)} eps")
