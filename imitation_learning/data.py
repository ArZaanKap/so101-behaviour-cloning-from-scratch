# il/data.py
import h5py
import numpy as np

DATA_PATH = "data/demos.hdf5"

# Which observation keys to feed the policy, concatenated in this order.
OBS_KEYS = [
    "joint_pos",    # 6: arm joints + gripper opening, i.e. where the robot is
    "tcp_pos",      # 3: fingertip point
    "cube_1_pos",   # 3: where to place
    "cube_2_pos",   # 3: what to pick
    "cube_2_quat",  # 4: the cube's rotation, needed to predict the gripper's yaw
]

def load_episodes(path=DATA_PATH, obs_keys=OBS_KEYS):
    episodes = []
    with h5py.File(path, "r") as f:
        # h5py sorts keys as text: demo_0, demo_1, demo_10, ... so sort by number
        names = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
        for name in names:
            g = f["data"][name]
            obs = np.concatenate([g["obs"][k][:] for k in obs_keys], axis=1).astype(np.float32)
            act = g["actions"][:].astype(np.float32)
            assert len(obs) == len(act), name
            episodes.append({"obs": obs, "act": act})
    return episodes


def split_episodes(episodes, val_frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(episodes))
    n_val = int(len(episodes) * val_frac)
    val = [episodes[i] for i in idx[:n_val]]
    train = [episodes[i] for i in idx[n_val:]]
    return train, val


if __name__ == "__main__":
    eps = load_episodes()
    train, val = split_episodes(eps)
    lens = [len(e["act"]) for e in eps]
    print(f"{len(eps)} episodes | obs_dim {eps[0]['obs'].shape[1]} | act_dim {eps[0]['act'].shape[1]}")
    print(f"length min/mean/max: {min(lens)} / {np.mean(lens):.1f} / {max(lens)}")
    print(f"train {len(train)} eps ({sum(len(e['act']) for e in train)} steps) | val {len(val)} eps")
