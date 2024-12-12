import time
from robot_reach_cube_env import RobotReachCubeEnv
from qlearning import QLearning

def train():
    env = RobotReachCubeEnv()
    ql_agent = QLearning(state_space=(10, 10, 10), action_space=ACTIONS)

    for episode in range(1000):
        env.reset_environment()
        state = env.get_obs()
        done = False
        total_reward = 0

        while not done:
            action = ql_agent.get_action(state)
            env.set_action(action)
            reward = env.compute_reward()
            next_state = env.get_obs()
            ql_agent.update_q_table(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            done = env.is_done()

        print(f"Episode {episode}: Total Reward: {total_reward}")
        time.sleep(1)

if __name__ == '__main__':
    train()
