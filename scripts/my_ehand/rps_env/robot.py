import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

urdf_cfg = sim_utils.UrdfFileCfg(
    asset_path="/workspace/isaaclab/scripts/my_ehand/urdf/eHand-6-R.urdf",
    make_instanceable=False,  # 与你的盘球项目保持一致，避免网格变形
    fix_base=True,
    collider_type="convex_decomposition",  # 盘球项目的精确碰撞
)

if urdf_cfg.joint_drive is not None:
    urdf_cfg.joint_drive.gains.stiffness = 10.0
    urdf_cfg.joint_drive.gains.damping = 1.0

# +++ 核心修复：引入盘球项目中的防穿模与高精度物理层解算 +++
urdf_cfg.articulation_props = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=False,
    solver_position_iteration_count=8,
    solver_velocity_iteration_count=1,
)

EHAND_CFG = ArticulationCfg(
    spawn=urdf_cfg,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            # +++ 核心修复：接管所有 16 个关节，赋予刚度，防止像面条一样乱甩 +++
            joint_names_expr=["joint_.*"],
            effort_limit=10.0,
            velocity_limit=10.0,
            stiffness=10.0,
            damping=1.0,
        ),
    },
)