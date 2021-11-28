## This is file for the Basic VAE model.

## General imports
import torch
import torch.nn as nn
import torch.nn.functional as F

## Class for vanilla convolutional VAE
class vanillaVAE(nn.Module):

    def __init__(self, input_channel=3, h_channels=[64,64,64,64,64], latent_size=100,):
        '''
        ::param input_channel: number of channels of the input tensor
        ::param h_channels: number of channels of the hidden layers
        ::param latent_size: size of the latent space
        '''
        super(vanillaVAE, self).__init__()

        assert len(h_channels) == 5, "h_channels must be a list of length = 5"
        self.channels = [input_channel] + h_channels

        # List for constructing the encoder/decoder architecture
        encoder_arch = []
        decoder_arch = []

        def conv_block(in_channels, out_channels, kernel_size=3, stride=2, padding=1):
            '''
            ::param in_channels: number of channels of the input tensor
            ::param out_channels: number of channels of the output tensor
            ::param kernel_size: size of the convolutional kernel
            ::param stride: stride of the convolutional kernel
            ::param padding: padding of the convolutional kernel
            '''
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def transconv_block(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
            '''
            ::param in_channels: number of channels of the input tensor
            ::param out_channels: number of channels of the output tensor
            ::param kernel_size: size of the convolutional kernel
            ::param stride: stride of the convolutional kernel
            ::param padding: padding of the convolutional kernel
            '''
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        ## Define the encoder architecture
        for i in range(len(self.channels)-1):
            encoder_arch += [
                conv_block(self.channels[i], self.channels[i+1]),
            ]
        encoder_arch += [nn.Flatten()]

        ## Define the decoder architecture
        for i in range(len(self.channels)-2):
            decoder_arch += [
                transconv_block(self.channels[-i-1], self.channels[-i-2]),
            ]
        
        decoder_arch += [nn.ConvTranspose2d(self.channels[1], self.channels[1], kernel_size=3, stride=2, padding=1, output_padding=1),
                        nn.BatchNorm2d(self.channels[1]),
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(self.channels[1], self.channels[0], kernel_size=3, stride=1, padding=1),
                        nn.Tanh()]

        ## Define the encoder and decoder architecture
        self.encoder = nn.Sequential(*encoder_arch)
        self.encoder_fc_mu = nn.Linear(self.channels[-1]*4, latent_size)
        self.encoder_fc_logvar = nn.Linear(self.channels[-1]*4, latent_size)

        self.decoder_fc = nn.Linear(latent_size, self.channels[-1]*4)
        self.decoder = nn.Sequential(*decoder_arch)

    def encode(self, x):
        x = self.encoder(x)
        mu = self.encoder_fc_mu(x)
        logvar = self.encoder_fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(-1, self.channels[-1], 2, 2)
        x = self.decoder(x)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def compute_loss(self, x, coeff=0.1, **kwargs):
        x_recon, mu, logvar = self.forward(x)
        assert x_recon.shape == x.shape, "x_recon.shape = {} and x.shape = {}".format(x_recon.shape, x.shape)
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + coeff * kl_loss

    def generate(self, n_samples=64):
        z = torch.randn(n_samples, self.latent_size)
        return self.decode(z)