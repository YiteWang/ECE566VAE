# ECE566 Final Project Variational Auto-Encoders (vanilla, IWAE, MIWAE)

Yite Wang, Yulun Wu

This is the Pytorch implementation of VAE, IWAE and MIWAE.



To run our code:

##### vanilla VAE

`python main.py --stamp VAE_final --vae_type vanilla --batch_size 128`

##### IWAE

`python main.py --stamp iwae_final --vae_type IWAE --batch_size 128 --num_particles 5`

##### MIWAE

`python main.py --stamp MIWAE_t1 --vae_type MIWAE --batch_size 128 --num_samples 3 --num_particles 5`

## Acknowledgement

1. We adopt and modify the neural network architecture and some hyper-parameters from this [repo](https://github.com/AntixK/PyTorch-VAE).
2. Original paper: [VAE](https://arxiv.org/abs/1312.6114), [IWAE](https://arxiv.org/abs/1509.00519), [MIWAE](https://arxiv.org/abs/1802.04537).

