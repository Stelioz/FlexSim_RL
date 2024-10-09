![ModelLayout](https://github.com/user-attachments/assets/65e31a9c-192a-4362-800a-4c26ce68e38d)

# [Reinforcement Learning with FlexSim]([https://hot.dit.hua.gr/](https://docs.flexsim.com/en/24.2/ModelLogic/ReinforcementLearning/KeyConcepts/KeyConcepts.html)) ![version](https://img.shields.io/badge/version-1.0.0-blue.svg)

The objective of this Repository is to develop an agent capable of making intelligent decisions using FlexSim. To achieve this we will follow these steps:
* Choose the [OpenAI Gymnasium ](https://gymnasium.farama.org/index.html) as our Reinforcement Learning library.
* Create a FlexSim model that uses the Reinforcement Learning tool.
* Wrap the FlexSim model as an environment.
* Train an agent, using that environment.
* Deploy the trained agent.
<br/><br/>

## Installation
1) Download and install VS Code, its Python extension, and Python 3 by following [Visual Studio Code's python tutorial](https://code.visualstudio.com/docs/python/python-tutorial).
2) Clone this repository to your local machine.
3) In the project directory create a virtual environment, running: `python -m venv venv`.
4) In the project directory activate the virtual environment, running: `.\FlexSim_RL_env\Scripts\Activate`.
5) In the project directory install necessary dependencies, running: `pip install -r requirements.txt`
6) Edit `flexsimPath` and `modelPath` in `flexsim_env.py` to correspond to your local paths.
7) Edit `flexsimPath` and `modelPath` in `flexsim_training.py` to correspond to your local paths.
8) Edit `model` in `flexsim_inference.py` to correspond to your Trained Model.
<br/>

## Troubleshoot execution policies
If `.\venv\Scripts\activate` cannot be loaded because running scripts is disabled, you can resolve the issue by following these steps:
* Open PowerShell as Admin.
* Check Current Execution Policy, running: `Get-ExecutionPolicy -List`.
* Change Execution Policy, running: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.
* Activate the Virtual Environment, running: `.\FlexSim_RL_env\Scripts\Activate`.
<br/>

## Software Development Team
* Stylianos Zindros
<br/>

## Special thanks
For the development of this repository, we used [FlexSim Documentation](https://docs.flexsim.com/en/24.2/Introduction/Welcome/Welcome.html).
