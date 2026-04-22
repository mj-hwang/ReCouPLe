# Register Algorithms here.

from .offline.bc import BehaviorCloning
# from .offline.dp import DiffusionPolicy
# from .offline.idql import IDQL
from .offline.iql import IQL
from .piql import PIQL
from .mtpiql import MTPIQL
from .rpl import RPL
from .rpl_proj_eq import RPLProjEQ
from .rpl_proj_2bt import RPLProj2BT
from .rpl_proj_only_reason import RPLProjOnlyReason
from .online.dqn import DQN, DoubleDQN, SoftDoubleDQN, SoftDQN
from .online.drqv2 import DRQV2
from .online.ppo import PPO, AdaptiveKLPPO
from .online.sac import SAC
from .online.td3 import TD3
