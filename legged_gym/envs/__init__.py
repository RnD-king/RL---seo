from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.jandi_base.jandi_base_config import JandiRobotBaseCfg, JandiRobotBaseCfgPPO
from legged_gym.envs.jandi_base.jandi_base_env import JandiRobotBase
from legged_gym.envs.jandi_walk.jandi_walk_config import JandiRobotWalkCfg, JandiRobotWalkCfgPPO
from legged_gym.envs.jandi_walk.jandi_walk_env import JandiWalkEnv
from legged_gym.envs.jandi_throw.jandi_throw_config import JandiThrowCfg, JandiThrowCfgPPO
from legged_gym.envs.jandi_throw.jandi_throw_env import JandiThrowEnv
from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry

task_registry.register( "jandi", JandiWalkEnv, JandiRobotWalkCfg(), JandiRobotWalkCfgPPO())
task_registry.register( "jandi_walk", JandiWalkEnv, JandiRobotWalkCfg(), JandiRobotWalkCfgPPO())
task_registry.register( "throw", JandiThrowEnv, JandiThrowCfg(), JandiThrowCfgPPO())
