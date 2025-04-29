Elastic Resource Allocation for 5G Telecommunication Networks with Deep Reinforcement Learning


This project investigates the use of reinforcement learning techniques for elastic resource allocation in 5G network slicing environments. A simulation framework was developed to model a realistic and dynamic system, incorporating multi-dimensional resources, service heterogeneity, user priority levels, time-varying demand, and random failure events. To further emulate real-world conditions, a mechanism was introduced to allocate additional resources to users experiencing poor connectivity, ensuring service quality is maintained during link degradation.
Two reinforcement learning algorithms—Q-Learning and Deep Q-Network (DQN)—were implemented and evaluated within this environment. The experimental results demonstrate that DQN significantly outperforms Q-Learning in complex, high-dimensional scenarios, particularly in adapting to system fluctuations and achieving higher cumulative rewards. While Q-Learning performs adequately in simpler setups, it lacks scalability in dynamic settings. The study also highlights the importance of hyperparameter tuning and environmental realism in achieving stable learning outcomes.

Usage
1.Install the virtual environment (venv_gpu):
python -m venv venv_gpu

2.Activate the environment using VS Code:
Open your project folder in VS Code.
Press Ctrl + Shift + P and type Python: Select Interpreter.
Select the interpreter located in your venv_gpu virtual environment, typically found in:
Windows: 
venv_gpu\Scripts\python.exe
Linux/MacOS: 
venv_gpu/bin/python

3.Run the Project:
Open a new terminal in VS Code.
Activate the virtual environment (if not already activated):
On Windows:
.\venv_gpu\Scripts\activate
On Linux/MacOS:
source venv_gpu/bin/activate

Dependencies
To run this project, make sure you install the following Python libraries in your virtual environment:
pip install numpy torch matplotlib
Library Descriptions:
numpy – for numerical operations, arrays, and probability distributions.
torch (PyTorch) – for building and training deep learning models such as DQNs.
matplotlib – for visualizing data, such as plotting reward curves.
