# Sparse-Vae
Sparse version of VAE developed at MCN @ Woods Hole 2018, further refined over the following year

SVAE.ipynb is the original modified VAE file from Woods Hole - included since it was used to generate some of the figures, but better off using the other verions.

VAE.m implements a simple matlab VAE implementation. No special requirements

SVAE-laplace and SVAE-cauchy implement sparse VAEs (laplacian and cauchy version, respectively). 
They are both based off of the same tensorflow template (not mine), which does standard VAE
Each code is capable of doing standard VAE and its version of SVAE

Both require tensorflow, the MNIST and CIFAR datasets. See comments in code for sources
SVAE-cauchy additionally requires the levy package in order to use the cauchy distribution

In order to swap between VAE/SVAE and different data sets
- SVAE-cauchy only : change alpha to 1 or 2 for SVAE-cauchy or regular VAE respectively
- change n_samples under xavier_init defintition to the number of inputs for the task
- In class VariationalAutoencoder uncomment the correct latent loss block for VAE or SVAE, comment the other
- in def train, uncomment the correct block for the desired task, comment others out
- In network_architecture, set n_input and n_z to be the size of the task (see comments there)
- For plot outputs, uncomment the block pertaining to each task, and change the output size to match each task



The other three files are presentations and writeups of results from the SVAE. They include changes to the VAE model, generated data examples, and an analysis of the SVAE performance and benefits.
