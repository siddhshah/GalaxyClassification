# GalaxyClassification
JAX-based deep neural network to distinguish between different images of galaxies from the Sloan Digital Sky Survey (SDSS) dataset.

Downloads and unpacks a compressed galaxy dataset (dataG.pkl.gz), visualizes raw train/test images and labels, and defines network parameters (W1, b1, W2, b2).

Implements a forward pass, loss function, and gradient‐descent updates using jax.numpy.

Once the model is trained accuracy and loss curves are plotted, and mini-batch training using jax.vmap is explored.

Achieved 75-80% accurate label classification for galaxy images, correctly labeling galaxy shapes out of 5 possible types.
