import torch
import numpy as np

### Modify the policy class to implement a neural network policy for the Sphero robot. T
# he policy should take the current state as input and output the action to be taken.
# You can use PyTorch to define the neural network architecture.
# You will need to train this policy using reinforcement learning or imitation learning techniques,
# which should be done in a separate training script (not provided).
class Policy(torch.nn.Module):
    """
    Neural network policy for the Sphero robot.

    SUBMISSION REQUIREMENTS:
      - Upload your trained weights and name the file `weight.pth` with this file.
      - The weights must be loadable with:
            policy.load_state_dict(torch.load("./weight.pth"))
        so save them with torch.save(policy.state_dict(), "./weight.pth").
      - The model must be callable as:
            action = policy(torch.tensor(obs, dtype=torch.float32)).detach().numpy()
        where:
            obs    : the observed state, in the format [x, y, heading, speed]
            action : the returned action, in the format [speed, turn_rate_cmd]
    """
    def __init__(self, d_in=4,hidden=64,d_out=2):
        super(Policy, self).__init__()

        self.model = torch.nn.Sequential(torch.nn.Linear(d_in, hidden),
                                         torch.nn.ReLU(),
                                   torch.nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.model(x)

