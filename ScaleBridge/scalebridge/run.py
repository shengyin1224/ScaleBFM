import os
import sys
import time
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig
from loguru import logger
from scalebridge.online.controller_watchdog import ControllerWatchdog

@hydra.main(
    version_base=None,
    config_path="config",
    config_name="base"
)
def main(cfg: DictConfig) -> None:
    ControllerWatchdog().start()

    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, 'run.log')
    logger.remove()
    logger.add(hydra_log_path, level='DEBUG')

    console_log_level = os.environ.get('LOGURU_LEVEL', 'INFO').upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)
    logger.info(f'Log saved to {hydra_log_path}')

    agent = instantiate(cfg.agent)
    metadata_dict = agent.get_meta_data()
    
    env = instantiate(cfg.env, metadata_dict=metadata_dict)
    
    print_cnt = 0
    dt = env.dt
    obs_dict = env.reset()

    for _ in range(20):
        agent.get_action(obs_dict)

    while True:
        timer = time.time()
        
        agent.before_step(obs_dict)

        tgt_dof_pos, action = agent.get_action(obs_dict)
        
        obs_dict = env.step(tgt_dof_pos, action)

        if (print_cnt < 5):
            logger.info(f'[Loop] Step {print_cnt}')
            logger.info(f'[Loop] The policy runs at a frequency of {1/(time.time() - timer)}HZ')
            logger.info(f'[Loop] One step takes {time.time()- timer}')
            print_cnt += 1
        
        agent.after_step(obs_dict, action)

        time_until_next_step = dt - (time.time() - timer)
        if time_until_next_step > 0:
            try:
                time.sleep(time_until_next_step)
            except:
                exit()
        else:
            logger.warning(f"[Loop] The policy layer suffers from excessive frame drop: {time_until_next_step}")    


if __name__=="__main__":
    main()
