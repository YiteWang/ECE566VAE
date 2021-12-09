## This is file for the IWAE model.

## General imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torchdist

def logmeanexp(inputs, dim=1): # ***
    if inputs.size(dim) == 1:
        return inputs
    else:
        return torch.logsumexp(inputs, dim=dim) - torch.log(torch.Tensor([inputs.size(dim)]).to(inputs.device))

## Class for IWAE convolutional VAE
class MIWAE(nn.Module):

    def __init__(self, input_channel=3, h_channels=[64,64,64,64,64], latent_size=100,):
        '''
        ::param input_channel: number of channels of the input tensor
        ::param h_channels: number of channels of the hidden layers
        ::param latent_size: size of the latent space
        '''
        super(MIWAE, self).__init__()

        assert len(h_channels) == 5, "h_channels must be a list of length = 5"
        self.channels = [input_channel] + h_channels
        self.latent_size = latent_size
        
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
                nn.LeakyReLU(),
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
                nn.LeakyReLU(),
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
                        nn.LeakyReLU(),
                        nn.Conv2d(self.channels[1], self.channels[0], kernel_size=3, stride=1, padding=1),
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
        distribtuion = torchdist.Normal(mu, std)
        z = distribtuion.rsample()
        return z, distribtuion

    def forward(self, x):
        mu, logvar = self.encode(x) # size of batch x num_samples x latent_size
        z, dist = self.reparameterize(mu, logvar) # size of batch x num_samples x latent_size
        return self.decode(z), z, dist, mu, logvar

    def compute_loss(self, x, coeff=0.1, num_samples = 3, num_particles = 5, **kwargs):
        B, C, H ,W = x.size()
        x = x.repeat(num_samples, num_particles, 1, 1, 1 ,1).permute(2, 0, 1, 3, 4, 5).contiguous().view(B*num_samples*num_particles, C, H, W) # Batch*num_samples*num_particles x C x H x W
        x_recon, z, dist_qz_x, mu, logvar = self.forward(x)
        assert x_recon.shape == x.shape, "x_recon.shape = {} and x.shape = {}".format(x_recon.shape, x.shape)

        # Find log q(z|x)
        log_qz_x = dist_qz_x.log_prob(z) # shape = batch*num_samples*num_particles x latent_size
        log_qz_x = log_qz_x.view(B, num_samples, num_particles, self.latent_size) # shape = batch x num_samples x num_particles x latent_size
        log_qz_x = torch.sum(log_qz_x, dim=-1) # sum over latent_size, now size = batch x num_samples x num_particles

        # Find log p(z)
        mu_prior = torch.zeros_like(z)
        std_prior = torch.ones_like(z)
        dist_prior = torchdist.Normal(mu_prior, std_prior)
        log_pz = dist_prior.log_prob(z) # size should be batch*numsamples*num_particles x latent_size
        log_pz = log_pz.view(B, num_samples, num_particles, self.latent_size) # batch x num_samples x num_particles x latent_size
        log_pz = torch.sum(log_pz, dim=-1) # size should be batch x num_samples x num_particles

        # Find log p(x|z)
        log_px_z = -(x_recon-x).pow(2).view(B, num_samples, num_particles, C, H, W).flatten(3).mean(-1) # Reduce to batch_size x num_samples x num_particles

        w = log_px_z + coeff * (log_pz - log_qz_x) # size = batch x num_samples x num_particles
        loss = -torch.mean(torch.mean(logmeanexp(w, 2),dim=1),dim=0) # size = 1

        return loss

    def test_loss(self, x, coeff=0.1, num_particles = 64, **kwargs):
        B, C, H ,W = x.size()
        x = x.repeat(num_particles, 1, 1, 1 ,1).permute(1, 0, 2, 3, 4).contiguous().view(B*num_particles, C, H, W) # Batch x num_particles x C x H x W
        x_recon, z, dist_qz_x, mu, logvar = self.forward(x)
        assert x_recon.shape == x.shape, "x_recon.shape = {} and x.shape = {}".format(x_recon.shape, x.shape)

        # Find log q(z|x)
        log_qz_x = dist_qz_x.log_prob(z) # shape = batch*num_particles x latent_size
        log_qz_x = log_qz_x.view(B, num_particles, self.latent_size) # shape = batch x num_particles x latent_size
        log_qz_x = torch.sum(log_qz_x, dim=-1) # sum over latent_size, now size = batch x num_particles

        # Find log p(z)
        mu_prior = torch.zeros_like(z)
        std_prior = torch.ones_like(z)
        dist_prior = torchdist.Normal(mu_prior, std_prior)
        log_pz = dist_prior.log_prob(z) # size should be batch*numsamples x latent_size
        log_pz = log_pz.view(B, num_particles, self.latent_size)
        log_pz = torch.sum(log_pz, dim=-1) # size should be batch x num_particles

        # Find log p(x|z)
        log_px_z = -(x_recon-x).pow(2).view(B, num_particles, C, H, W).flatten(2).mean(-1) # Reduce to batch_size x num_particles
        recontruction_loss = -(x_recon-x).pow(2).view(B, num_particles, C, H, W).flatten(2).sum(-1).mean()

        w = log_px_z + coeff * (log_pz - log_qz_x) # size = batch x num_particles
        IWAE_loss = -torch.mean(logmeanexp(w, 1)) # size = 1

        return IWAE_loss, recontruction_loss

    def generate(self, n_samples=64):
        z = torch.randn(n_samples, self.latent_size).cuda()
        return self.decode(z)