# Vehicle Classification CNN

A convolutional neural network built from scratch in PyTorch to classify vehicle images into 8 categories.

## Classes
Bicycle, Bus, Car, Motorcycle, NonVehicles, Taxi, Truck, Van

## Model Architecture
- 4 convolutional blocks (Conv2d → BatchNorm2d → ReLU → MaxPool2d)
- Fully connected head (Linear 4096→512 → ReLU → Dropout → Linear 512→8)
- Loss: CrossEntropyLoss
- Optimizer: Adam (lr=0.001, weight_decay=1e-4)
- LR Scheduler: StepLR (step_size=7, gamma=0.1)

## Results
| Metric | Score |
|---|---|
| Final Training Accuracy | 88.20% |
| Final Validation Accuracy | 86.24% |
| Epochs | 20 |
| Train/Val Split | 80/20 |
