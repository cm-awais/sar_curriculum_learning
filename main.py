from utils import *


warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # for multi-GPU setups

# Disable CuDNN heuristics
cudnn.benchmark = False
cudnn.benchmark = False

"""
The following functions are for training the baseline models without any weights intervention.
"""
baseline_norm()
curriculum_norm()

"""
The following functions are for training the model from a starting weights that are based on the class ratio.
"""
train_baseline_weights()
train_curriculum_weights()

"""
The following functions are for training the model from random weight assignment and updating the weights based on previous scores.
"""
train_baseline_weightsU0()
train_curriculum_weightsU0()

"""
The following functions are for training the model from a starting weights that are based on the class ratio, and updating the weights after 5 epochs.
"""
train_baseline_weightsU()
train_curriculum_weightsU()