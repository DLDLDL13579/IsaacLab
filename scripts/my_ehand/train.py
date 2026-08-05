import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Sim2Real RPS Hand")
parser.add_argument("--task", type=str, default="Isaac-EHand-RPS-Direct-v0", help="要运行的注册任务名")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import rps_env  
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rps_env.env import RPSHandEnvCfg
from rps_env.agents import rsl_rl_ppo_cfg  

def main():
    print(f"[INFO] 正在拉起任务: {args_cli.task}")
    
    env_cfg = RPSHandEnvCfg()
    env = gym.make(args_cli.task, cfg=env_cfg)
    vec_env = RslRlVecEnvWrapper(env)
    runner_cfg = rsl_rl_ppo_cfg.to_dict()
    
    runner = OnPolicyRunner(vec_env, runner_cfg, log_dir="logs/rps", device="cuda:0")
    
    # 开始大规模训练
    runner.learn(num_learning_iterations=runner_cfg["max_iterations"], init_at_random_ep_len=True)
    
    vec_env.close()
    
    """
    [Sim2Real 导出提示]
    训练完成后，RSL-RL 会在 logs/rps/ehand_rps_sim2real/v_final 目录下自动生成 policy.pt。
    由于实际控制器中通常使用 C++ 或 TensorRT 运行推理网络以保证毫秒级下发，
    你需要通过 RSL-RL 自带的 export 脚本将 policy.pt 转换为 policy.onnx 格式：
    
    执行指令示例：
    python -m rsl_rl.scripts.export_policy --log_dir logs/rps/ehand_rps_sim2real/v_final
    """

if __name__ == "__main__":
    main()
    simulation_app.close()