import argparse
import os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Load eHand URDF with Physics Fixes.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 纯净版导入，彻底移除 omni.isaac.core
import isaaclab.sim as sim_utils

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, "urdf", "eHand-6-R.urdf")
    
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)
    
    # ===== 核心修复区 =====
    urdf_cfg = sim_utils.UrdfFileCfg(
        asset_path=urdf_path,
        make_instanceable=False,
        fix_base=True,
    )
    
    # 修复 1：明确关闭自碰撞（彻底解决原件穿模弹开的问题）
    urdf_cfg.articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=1,
    )
    
    # 修复 2：加入弹簧刚度与阻尼（物理镇静剂）
    urdf_cfg.joint_drive.gains.stiffness = 2.0 
    urdf_cfg.joint_drive.gains.damping = 0.5  
    # ======================
    
    urdf_cfg.func("/World/eHand", urdf_cfg, translation=(0.0, 0.0, 0.5))
    
    sim.reset()
    sim.set_camera_view(eye=[0.5, 0.5, 0.5], target=[0.0, 0.0, 0.5])
    
    print("\n====== 🚀 灵巧手物理参数修复成功！请在窗口中查看。 ======\n")

    while simulation_app.is_running():
        sim.step(render=True)

if __name__ == "__main__":
    main()
    simulation_app.close()
