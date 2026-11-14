"""PPO schedules for the four planned dexhand ablations."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


DEXHAND_METRICS = [
    "fingertip_error_rh",
    "fingertip_error_lh",
    "joint_error_rh",
    "joint_error_lh",
    "object_drop",
    "object_contact",
]


@configclass
class DexHandActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "ActorCriticDexHand"
    motion_latent_dim: int = 128
    std_clamp_range: tuple[float, float] = (0.001, 1.0)


@configclass
class DexHandPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    actor_learning_rate: float = 2.0e-5
    critic_learning_rate: float = 1.0e-3
    bc_anchor_coef: float = 1.0
    bc_anchor_final_coef: float = 0.1
    bc_anchor_decay_updates: int = 10_000
    entropy_final_coef: float = 5.0e-4
    entropy_decay_updates: int = 10_000


@configclass
class DexHandPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 64
    max_iterations = 50_000
    save_interval = 1_000
    experiment_name = "dexhand_gmt"
    run_name = "bc_rl_obj"
    empirical_normalization = None
    obs_groups = {
        "policy": ["policy", "policy_task"],
        "critic": ["policy", "policy_task", "critic"],
    }
    eval_during_training = False
    eval_interval = 2_000
    eval_metric_keys = DEXHAND_METRICS
    eval_max_steps = 2_000
    success_metric_dict = {"object_drop": 0.2}
    command_name = "motion"
    policy = DexHandActorCriticCfg(
        init_noise_std=0.135335283,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 512, 256, 128],
        critic_hidden_dims=[512, 512, 256, 128],
        activation="silu",
    )
    algorithm = DexHandPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=2,
        num_mini_batches=32,
        actor_learning_rate=2.0e-5,
        critic_learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        bc_anchor_coef=1.0,
        bc_anchor_final_coef=0.1,
        bc_anchor_decay_updates=10_000,
        entropy_final_coef=5.0e-4,
        entropy_decay_updates=10_000,
    )


@configclass
class DexHandScratchPPORunnerCfg(DexHandPPORunnerCfg):
    run_name = "rl_obj_scratch"
    algorithm = DexHandPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=2,
        num_mini_batches=32,
        actor_learning_rate=2.0e-5,
        critic_learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        bc_anchor_coef=0.0,
        bc_anchor_final_coef=0.0,
        bc_anchor_decay_updates=0,
        entropy_final_coef=5.0e-4,
        entropy_decay_updates=10_000,
    )


@configclass
class DexHandNoObjectPPORunnerCfg(DexHandPPORunnerCfg):
    run_name = "bc_rl_noobj"

