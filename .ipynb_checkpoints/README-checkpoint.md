## For datasets

- [x] [Street View House Numbers (SVHN)](https://www.kaggle.com/datasets/stanfordu/street-view-house-numbers/data)

## For models

- **Model 1: Baseline (Simple MLP)**
  - Model Architecture:
    - Input Layer: 3072 units (flattened 32*32*3 image)
    - Hidden Layer: 512 units with ReLU activation
    - Output Layer: 10 units (for 10 classes)
  - Hyperparameters:
    - Learning Rate: 0.001
    - Batch Size: 64
    - Epochs: 15
    - Optimizer: Adam
    - Loss Function: Cross-Entropy Loss
  - Model 1 Result:
    - Train Loss: 0.2654
    - Test Accuracy: 0.86

- **Model 2: Improved MLP Variant (Still an MLP, but improved)**
  - Model Architecture:
    - Input Layer: 3072 units (flattened 32*32*3 image)
    - Hidden Layer 1: 1024 units with LeakyReLU activation
    - Hidden Layer 2: 512 units with LeakyReLU activation
    - Hidden Layer 3: 256 units with LeakyReLU activation
    - Output Layer: 10 units (for 10 classes)
  - Hyperparameters:
    - Learning Rate: 0.0005
    - Batch Size: 64
    - Epochs: 20
    - Optimizer: AdamW (Adam with Weight Decay)
    - Loss Function: Cross-Entropy Loss
  - Model 2 Result:
    - Train Loss: 0.1987
    - Test Accuracy: 0.89

  You should include at least two improvements, chosen from related techniques
  such as
  - Activation function change
  - Regularization
  - Optimizer or learning rate tuning
  - Architecture adjustment such as using deeper/wider network
    …
    You should clearly state what you changed and discuss whether the change
    improved performance.

- Model 3: Improved Model (Free choice)

  You are free to choose a more advanced model (i.e. CNN, RNN or any other
  advanced architecture) that you believe better matches your task.
