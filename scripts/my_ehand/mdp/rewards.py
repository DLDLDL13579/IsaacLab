import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject

def object_survival(
    env: ManagerBasedRLEnv, 
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"), 
    min_height: float = 0.05
) -> torch.Tensor:
    asset: RigidObject = env.scene[object_cfg.name]
    object_z = asset.data.root_pos_w[:, 2]
    return torch.where(object_z > min_height, torch.ones_like(object_z), torch.zeros_like(object_z))
def wrist_zone_penalty(
    env: ManagerBasedRLEnv, 
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"), 
    wrist_x_threshold: float = 0.04
) -> torch.Tensor:
    """
    惩罚手腕停滞：手掌中心在 X=0.07。
    如果球的 X 坐标小于 0.04（滑向手腕基座），则输出 1 进行惩罚扣分。
    """
    asset: RigidObject = env.scene[object_cfg.name]
    object_x = asset.data.root_pos_w[:, 0]
    return torch.where(object_x < wrist_x_threshold, torch.ones_like(object_x), torch.zeros_like(object_x))