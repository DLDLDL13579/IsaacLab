import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject

def object_dropped(
    env: ManagerBasedRLEnv, 
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"), 
    drop_height: float = 0.04
) -> torch.Tensor:
    asset: RigidObject = env.scene[object_cfg.name]
    object_z = asset.data.root_pos_w[:, 2]
    return object_z < drop_height