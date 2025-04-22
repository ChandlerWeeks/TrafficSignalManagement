import os
import argparse
import traci

from utils.config_parser import load_config
from env.multi_env       import CityEnv
from agents.policies     import Agent           # MultiDQNAgent alias
from utils.logger        import Logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('mode', choices=['train', 'eval'])
    args = ap.parse_args()

    cfg        = load_config(args.config)
    env_cfg    = cfg['ENV']
    agent_cfg  = cfg.get('AGENT', {})
    log_cfg    = cfg.get('LOG',  {})
    train_cfg  = cfg.get('TRAIN', {})
    eval_cfg   = cfg.get('EVAL',  {})

    env    = CityEnv(env_cfg)
    agent  = Agent(env.observation_space,  # list[int]
                   env.action_space,       # list[int]
                   agent_cfg)
    logger = Logger(log_cfg.get('LOG_DIR', 'logs'))

    if args.mode == 'train':
        episodes = int(train_cfg.get('EPISODES', 0))
        for ep in range(episodes):
            ep_ret = 0.0

            # ---------- episode starts ----------
            print(f"[TRAIN] ▶ Episode {ep + 1}/{episodes} starting …", flush=True)
            states, done = env.reset(), False 

            max_steps = 2400 # 0.75 hour of simulation time
            steps = 0

            while not done and steps < max_steps:
                actions            = agent.select_action(states)   # list[int]
                next_s, rewards, done, _ = env.step(actions)       # rewards list[float]

                n_tls = len(actions)
                agent.remember(states, actions, rewards, next_s, [done]*n_tls)
                agent.learn()

                states = next_s
                ep_ret += sum(rewards)      # aggregate episode return
                steps  += 1

            # ---------- episode ends ----------
            logger.log_episode(ep, ep_ret)
            print(f"[TRAIN] ✓ Episode {ep + 1}/{episodes} finished — "
                f"return: {ep_ret:.1f}", flush=True)

        agent.save(os.path.join(logger.log_dir,
                                agent_cfg.get('NAME', 'dqn_city')))

    else:  # ------------------------ eval ------------------------
        episodes = int(eval_cfg.get('EPISODES', 0))
        returns  = []
        for ep in range(episodes):

            print(f"[EVAL ] ▶ Episode {ep + 1}/{episodes} starting …", flush=True)
            s, done, tot = env.reset(), False, 0.0

            while not done:
                a            = agent.select_action(s, evaluate=True)
                s, r, done, _ = env.step(a)
                tot         += r

            returns.append(tot)
            print(f"[EVAL ] ✓ Episode {ep + 1}/{episodes} finished — "
                f"return: {tot:.1f}", flush=True)

        if returns:
            mean_ret = sum(returns) / len(returns)
            print(f"[EVAL ] Mean return over {episodes} episodes: {mean_ret:.1f}")



    env.close()


if __name__ == '__main__':
    main()
