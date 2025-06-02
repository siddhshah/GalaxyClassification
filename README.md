# Galaxy Classification
JAX-based deep neural network to distinguish between different images of galaxies from the Sloan Digital Sky Survey (SDSS) dataset.

Downloads and unpacks a compressed galaxy dataset (dataG.pkl.gz), visualizes raw train/test images and labels, and defines network parameters (W1, b1, W2, b2).

Implements a forward pass, loss function, and gradient‐descent updates using jax.numpy.

Once the model is trained accuracy and loss curves are plotted, and mini-batch training using jax.vmap is explored.

Achieved 75-80% accurate label classification for galaxy images, correctly labeling galaxy shapes out of 5 possible types.

## Example Galaxy
![Screenshot 2025-06-01 031604](https://github.com/user-attachments/assets/100471d7-b3b5-40d8-bf3a-a42ecd9787b9)

## Training Results
![Training Accuracy](https://github.com/user-attachments/assets/46876690-4a55-4713-b398-a585a775057f)
