import gymnasium as gym
import os
import subprocess
import socket
import json
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
import numpy as np

class FlexSimEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array', 'ansi']}

    def __init__(self, flexsimPath, modelPath, address='localhost', port=5005, verbose=False, visible=False):
        self.flexsimPath = flexsimPath
        self.modelPath = modelPath
        self.address = address
        self.port = port
        self.verbose = verbose
        self.visible = visible

        self.lastObservation = ""

        self._launch_flexsim()
        
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

    # Updated reset() method to match the new Gym API
    def reset(self, seed=None, options=None):
        # Set the seed for the environment
        if seed is not None:
            self.seed(seed)
    
        # Call any other necessary reset logic
        self._reset_flexsim()
        state, reward, done = self._get_observation()
    
        return state, {}

    # Updated step() method to match the new Gym API 
    def step(self, action):
        self._take_action(action)
        state, reward, done = self._get_observation()

        # Determine if the episode is terminated or truncated
        terminated = done
        truncated = False  # You can set this based on your own logic
        info = {}

        return state, reward, terminated, truncated, info

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return np.array([0,0,0])
        elif mode == 'human':
            print(self.lastObservation)
        elif mode == 'ansi':
            return self.lastObservation
        else:
            super(FlexSimEnv, self).render(mode=mode)

    def close(self):
        self._close_flexsim()
        
    def seed(self, seed=None):
        self.seedNum = seed
        return self.seedNum

    
    def _launch_flexsim(self):
        if self.verbose:
            print("Launching " + self.flexsimPath + " " + self.modelPath)

        args = [self.flexsimPath, self.modelPath, "-training", self.address + ':' + str(self.port)]
        if self.visible == False:
            args.append("-maintenance")
            args.append("nogui")
        self.flexsimProcess = subprocess.Popen(args)

        self._socket_init(self.address, self.port)
    
    def _close_flexsim(self):
        self.flexsimProcess.kill()

    def _release_flexsim(self):
        if self.verbose:
            print("Sending StopWaiting message")
        self._socket_send(b"StopWaiting?")

    def _get_action_space(self):
        self._socket_send(b"ActionSpace?")
        if self.verbose:
            print("Waiting for ActionSpace message")
        actionSpaceBytes = self._socket_recv()
        
        return self._convert_to_gym_space(actionSpaceBytes)

    def _get_observation_space(self):
        self._socket_send(b"ObservationSpace?")
        if self.verbose:
            print("Waiting for ObservationSpace message")
        observationSpaceBytes = self._socket_recv()
        
        return self._convert_to_gym_space(observationSpaceBytes)

    def _reset_flexsim(self):
        if self.verbose:
            print("Sending Reset message")
        resetString = "Reset?"
        if hasattr(self, "seedNum"):
            resetString = "Reset:" + str(self.seedNum) + "?"
        self._socket_send(resetString.encode())

    def _get_observation(self):
        if self.verbose:
            print("Waiting for Observation message")
        observationBytes = self._socket_recv()
        self.lastObservation = observationBytes.decode('utf-8')
        state, reward, done = self._convert_to_observation(observationBytes)

        return state, reward, done
    
    def _take_action(self, action):
        actionStr = json.dumps(action, cls=NumpyEncoder)
        if self.verbose:
            print("Sending Action message: " + actionStr)
        actionMessage = "TakeAction:" + actionStr + "?"
        self._socket_send(actionMessage.encode())


    def _socket_init(self, host, port):
        if self.verbose:
            print("Waiting for FlexSim to connect to socket on " + self.address + ":" + str(self.port))

        self.serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serversocket.bind((host, port))
        self.serversocket.listen();

        (self.clientsocket, self.socketaddress) = self.serversocket.accept()
        if self.verbose:
            print("Socket connected")
        
        if self.verbose:
            print("Waiting for READY message")
        message = self._socket_recv()
        if self.verbose:
            print(message.decode('utf-8'))
        if message != b"READY":
            raise RuntimeError("Did not receive READY! message")

    def _socket_send(self, msg):
        totalsent = 0
        while totalsent < len(msg):
            sent = self.clientsocket.send(msg[totalsent:])
            if sent == 0:
                raise RuntimeError("Socket connection broken")
            totalsent = totalsent + sent

    def _socket_recv(self):
        chunks = []
        while 1:
            chunk = self.clientsocket.recv(2048)
            if chunk == b'':
                raise RuntimeError("Socket connection broken")
            if chunk[-1] == ord('!'):
                chunks.append(chunk[:-1])
                break;
            else:
                chunks.append(chunk)
        return b''.join(chunks)


    def _convert_to_gym_space(self, spaceBytes):
        paramsStartIndex = spaceBytes.index(ord('('))
        paramsEndIndex = spaceBytes.index(ord(')'), paramsStartIndex)
        
        type = spaceBytes[:paramsStartIndex]
        params = json.loads(spaceBytes[paramsStartIndex+1:paramsEndIndex])
        
        if type == b'Discrete':
            return gym.spaces.Discrete(params)
        elif type == b'Box':
            return gym.spaces.Box(np.array(params[0]), np.array(params[1]))
        elif type == b'MultiDiscrete':
            return gym.spaces.MultiDiscrete(params)
        elif type == b'MultiBinary':
            return gym.spaces.MultiBinary(params)

        raise RuntimeError("Could not parse gym space string")

    def _convert_to_observation(self, spaceBytes):
        observation = json.loads(spaceBytes)
        state = observation["state"]
        if isinstance(state, list):
            state = np.array(observation["state"])
        reward = observation["reward"]
        done = (observation["done"] == 1)
        return state, reward, done


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main():

    env = FlexSimEnv(
        flexsimPath = "C:/Program Files/FlexSim 2024 Update 2/program/flexsim.exe", # Edit Local Path to FlexSim executable
        modelPath = "C:/Users/steal/Documents/FlexSim/Test_Models/ChangeoverTimesRL.fsm", # Edit Local Path to FlexSim model
        verbose = True,
        visible = True
        )

    for i in range(2):
        env.seed(i)
        observation, _ = env.reset()  # Unpack the observation from the tuple
        env.render()
        done = False
        rewards = []
        while not done:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)  # Unpack all 5 return values
            env.render()
            rewards.append(reward)

            done = terminated or truncated  # Combine both to check if the episode is finished
        
            if done:
                cumulative_reward = sum(rewards)
                print("Reward: ", cumulative_reward, "\n")
                
    env._release_flexsim()
    input("Waiting for input to close FlexSim...")
    env.close()


if __name__ == "__main__":
    main()