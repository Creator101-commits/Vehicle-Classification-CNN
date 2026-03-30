# Vehicle Classification CNN
> A convolutional neural network built from scratch in PyTorch to classify vehicle images into 8 categories.

A deep learning model that classifies vehicle images into 8 distinct categories using a custom CNN architecture built with PyTorch. The model achieves over 86% validation accuracy across classes including Bicycle, Bus, Car, Motorcycle, NonVehicles, Taxi, Truck, and Van.

## Installation

OS X & Linux:

```sh
pip install -r requirements.txt
```

Windows:

```sh
pip install -r requirements.txt
```

## Usage example

Train the model by running the main training script:

```sh
python train.py
```

_For more examples and usage, please refer to the [Wiki][wiki]._

## Development setup

Install all dependencies and run the training pipeline:

```sh
pip install -r requirements.txt
python train.py
```

## Model Architecture

- 4 convolutional blocks (Conv2d → BatchNorm2d → ReLU → MaxPool2d)
- Fully connected head (Linear 4096→512 → ReLU → Dropout → Linear 512→8)
- Loss: CrossEntropyLoss
- Optimizer: Adam (lr=0.001, weight_decay=1e-4)
- LR Scheduler: StepLR (step_size=7, gamma=0.1)

## Classes

Bicycle, Bus, Car, Motorcycle, NonVehicles, Taxi, Truck, Van

## Results

| Metric | Score |
|---|---|
| Final Training Accuracy | 88.20% |
| Final Validation Accuracy | 86.24% |
| Epochs | 20 |
| Train/Val Split | 80/20 |

## Release History

* 0.1.0
    * The first proper release

## Meta

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

[https://github.com/Creator101-commits/Vehicle-Classification-CNN](https://github.com/Creator101-commits/Vehicle-Classification-CNN)

## Contributing

1. Fork it (<https://github.com/Creator101-commits/Vehicle-Classification-CNN/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

[wiki]: https://github.com/Creator101-commits/Vehicle-Classification-CNN/wiki
