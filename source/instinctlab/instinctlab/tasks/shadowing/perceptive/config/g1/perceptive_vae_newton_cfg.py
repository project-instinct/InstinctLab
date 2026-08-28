from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from instinctlab.assets.unitree_g1 import G1_REFERENCE_CFG
from instinctlab.tasks.shadowing.perceptive.config.g1.perceptive_shadowing_newton_cfg import G1PerceptiveNewtonSceneCfg
from instinctlab.tasks.shadowing.perceptive.config.g1.perceptive_vae_cfg import (
    G1_CFG,
    G1PerceptiveVaeEnvCfg,
    G1PerceptiveVaeEnvCfg_PLAY,
    motion_reference_cfg,
)
from instinctlab.tasks.utils.newton import newton_sim_cfg


@configclass
class G1PerceptiveVaeNewtonSceneCfg(G1PerceptiveNewtonSceneCfg):
    pass


@configclass
class G1PerceptiveVaeNewtonEnvCfg(G1PerceptiveVaeEnvCfg):
    scene: G1PerceptiveVaeNewtonSceneCfg = G1PerceptiveVaeNewtonSceneCfg(
        num_envs=4096,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        motion_reference=motion_reference_cfg,
        height_scanner=None,
    )
    sim: SimulationCfg = newton_sim_cfg(njmax=256, nconmax=128, margin=0.01, gap=0.01, use_mujoco_contacts=False)


@configclass
class G1PerceptiveVaeNewtonEnvCfg_PLAY(G1PerceptiveVaeEnvCfg_PLAY):
    scene: G1PerceptiveVaeNewtonSceneCfg = G1PerceptiveVaeNewtonSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        robot_reference=G1_REFERENCE_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotReference"),
        motion_reference=motion_reference_cfg.replace(debug_vis=True),
    )
    sim: SimulationCfg = newton_sim_cfg(njmax=256, nconmax=128, margin=0.01, gap=0.01, use_mujoco_contacts=False)
