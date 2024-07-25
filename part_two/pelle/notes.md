# Notes - part 1

## TODOs

* [X] caricare i datasets
* [X] provare modello semplicissimo
* [X] copiare e adattare il trainer dell'es 5
* [ ] Implementare un tester da lanciare sul test data loader --> a quanto pare non necessario
* [X] lanciare (Simple, Medium, Complex, ResNet-18) con account diversi ==> ultimi due troppo lenti --> provarli con batch size diversi
* [ ] motivare meglio le scelte di design --> troppo didattiche
* [X] provare anche ad implementare una rete nota (ResNet-18)
* [ ] provare batch_size al massimo (256 o 512)

## Models

### Simple

* **Single Convolutional Layer** :
  `self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)`
  This layer takes a 3-channel input (e.g., an RGB image) and outputs 8 feature maps. The kernel size of 3x3 is a standard choice for capturing local spatial features. Padding of 1 ensures that the output size remains the same as the input size after the convolution operation.
* **Fully Connected Layers** :
  `self.fc1 = nn.Linear(8 * 112 * 112, 64)`
  Reduces the flattened tensor from the convolutional layer to 64 units.
  `self.fc2 = nn.Linear(64, n_classes)`
  Maps these 64 units to the number of classes for classification.
* **Activation and Pooling** :
  ReLU is used as the activation function to introduce non-linearity. Max pooling reduces the spatial dimensions by half, helping to downsample the input and reduce computational complexity.

### Medium

* **Three Convolutional Layers** :
  `self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)`
  `self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)`
  `self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)`
  Each layer doubles the number of feature maps, progressively increasing the capacity of the network to learn complex patterns. Padding ensures the spatial dimensions are preserved after each convolution.
* **Fully Connected Layers** :
  `self.fc1 = nn.Linear(64 * 28 * 28, 128)`
  `self.fc2 = nn.Linear(128, n_classes)`
  The fully connected layers follow a similar pattern to SimpleCNN but with more units, allowing for greater capacity and more complex feature combinations.
* **Activation and Pooling** :
  See SimpleCNN

### Complex

* **Four Convolutional Layers with Batch Normalization** :
  `self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)`
  `self.bn1 = nn.BatchNorm2d(32)`
  `self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)`
  `self.bn2 = nn.BatchNorm2d(64)`
  `self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)`
  `self.bn3 = nn.BatchNorm2d(128)`
  `self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)`
  `self.bn4 = nn.BatchNorm2d(256)`
  Batch normalization is added after each convolutional layer to stabilize and accelerate training by normalizing the output of each layer.
* **Fully Connected Layers with Dropout** :
  `self.fc1 = nn.Linear(256 * 14 * 14, 512)`
  `self.dropout = nn.Dropout(p=0.5)`
  `self.fc2 = nn.Linear(512, n_classes)`
  Dropout is added to prevent overfitting by randomly setting a fraction of the input units to 0 at each update during training time.
* **Activation and Pooling** :
  The combination of ReLU activation, batch normalization, and max pooling after each convolutional layer ensures robust learning, better generalization, and reduced overfitting.

### ResNet-N

* **Depth** : N=18 layers (including convolutional and fully connected layers).
* **Residual Connections** : Helps in mitigating vanishing gradients and allows for deeper networks by enabling easier flow of gradients.
* **Batch Normalization** : Stabilizes learning and improves convergence.
* **Adaptability** : The `_make_layer` method allows for easy configuration of different ResNet variants by changing the number of blocks per layer.
  We used a basic block of [2,2,2,2] so 16 convolutional layers + first + last = 18

## Best Results

| Model     | Params |
| --------- | ------ |
| Simple    | 0.38   |
| Medium    | 0.51   |
| Complex   |        |
| ResNet-18 |        |

# Notes - part 2
