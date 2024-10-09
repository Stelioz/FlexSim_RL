import gymnasium as gym
from flexsim_env import FlexSimEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def main():
    print("Initializing FlexSim environment...")

    # Create a FlexSim OpenAI Gym Environment
    env = FlexSimEnv(
        flexsimPath = "C....../FlexSim 2024 Update 2/program/flexsim.exe", # Edit Local Path to FlexSim executable, replace the dots with the actual path
        modelPath = "....../ChangeoverTimesRL.fsm", # Edit Local Path to FlexSim model, replace the dots with the actual path
        verbose = False,
        visible = False
        )
    
    check_env(env) # Check that an environment follows Gym API.

    # Training a baselines3 PPO model in the environment
    model = PPO("MlpPolicy", env, verbose=1)
    print("Training model...")
    model.learn(total_timesteps=1)
    
    # Save the model
    print("Saving model...")
    model.save("ChangeoverTimesModel")

    input("Waiting for input to do some test runs...")

    # Run test episodes using the trained model
    for i in range(2):
        env.seed(i)
        observation, _ = env.reset()  # Unpack the observation from the tuple
        env.render()
        done = False
        rewards = []
        while not done:
            action, _states = model.predict(observation)  # Pass only the observation
            observation, reward, terminated, truncated, info = env.step(action)  # Expecting five return values
            env.render()
            rewards.append(reward)
            if terminated:
                cumulative_reward = sum(rewards)
                print("Reward: ", cumulative_reward, "\n")

    env._release_flexsim()
    input("Waiting for input to close FlexSim...")
    env.close()

if __name__ == "__main__":
    main()