import sys, os
#sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import argparse
from utils.config_parser import load_config
from utils.logger import Logger
from env.city_env import CityEnv
from agents.policies import get_agent_class

# from traffic_signal_nn.utils.config_parser import load_config
# from traffic_signal_nn.utils.logger        import Logger
# from traffic_signal_nn.env.city_env       import CityEnv
# from traffic_signal_nn.agents.policies    import get_agent_class

def main():
    parser = argparse.ArgumentParser(description="Multi‑RL Traffic Signal Control")
    parser.add_argument("--config",     required=True, help="Path to .ini config")
    parser.add_argument("mode",         choices=["train","eval","demo"],
                        help="train: learn; eval: run saved model; demo: render only")
    args = parser.parse_args()

    # load all sections into nested dicts
    cfg = load_config(args.config)

    # make env & agent
    env_cls = CityEnv
    env     = env_cls(cfg["ENV"])
    Agent   = get_agent_class(cfg["AGENT"]["NAME"])
    agent   = Agent(env.observation_space,
                    env.action_space,
                    cfg["AGENT"])

    # make logger
    logger = Logger(cfg["LOG"]["LOG_DIR"])

    if args.mode == "train":
        for episode in range(cfg["TRAIN"]["EPISODES"]):
            state, done, ep_reward = env.reset(), False, 0.0
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.learn()
                state, ep_reward = next_state, ep_reward + reward

            agent.update_target_network()
            logger.log_episode(episode, ep_reward, agent)

        os.makedirs(logger.log_dir, exist_ok=True)
        agent.save(os.path.join(logger.log_dir, "agent.pth"))

    elif args.mode == "eval":
        agent.load(os.path.join(cfg["LOG"]["LOAD_DIR"], "agent.pth"))
        returns = []
        for episode in range(cfg["EVAL"]["EPISODES"]):
            state, done, ep_reward = env.reset(), False, 0.0
            while not done:
                action = agent.select_action(state)
                state, reward, done, _ = env.step(action)
                ep_reward += reward
            returns.append(ep_reward)
        print(f"Mean return over {len(returns)} episodes: {sum(returns)/len(returns)}")

    elif args.mode == "demo":
        agent.load(os.path.join(cfg["LOG"]["LOAD_DIR"], "agent.pth"))
        state, done = env.reset(), False
        while not done:
            action = agent.select_action(state)
            state, _, done, _ = env.step(action)
            env.render()  # if you implement render()

if __name__ == "__main__":
    main()

