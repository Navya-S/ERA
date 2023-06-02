
# MNIST data clasiification using CNN

This repository showcases image classification of MNIST dataset using CNN. Along with code modularization.




## Documentation

[Documentation](https://linktodocumentation)

model.py --> defines model architecture and holds methods for training and testing model.

utils.py --> contains utilityy functions.

s5.ipynb --> acts as main file, where the data loading, transformation, training, testing and evaluation methods are called. Also, used to experiment with various hyperparameter tuning. 
## Usage/Examples
torch summary provides the model architecture, info on total parameters

```python
!pip install torchsummary
from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
summary(model, input_size=(1, 28, 28))
```
