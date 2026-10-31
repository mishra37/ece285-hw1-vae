"""The convolutional VAE used for the chest X-rays, plus its loss.

The image size must be divisible by 16 (one halving per conv layer); we use 64x64.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, img_size=64, latent_dim=128, conv_channels=(32, 64, 128, 256)):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv_channels = conv_channels
        # Each conv layer halves the height and width: 64 -> 32 -> 16 -> 8 -> 4.
        self.smallest_size = img_size // 2 ** len(conv_channels)
        flattened_size = conv_channels[-1] * self.smallest_size ** 2  # 256 * 4 * 4 = 4096 by default

        # Encoder: every block makes the image smaller and adds channels.
        encoder_layers = []
        in_channels = 1  # grayscale input
        for out_channels in conv_channels:
            encoder_layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2),
            ]
            in_channels = out_channels
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(flattened_size, latent_dim)

        # Decoder: the encoder run backwards; every transposed conv doubles the image size.
        self.fc_dec = nn.Linear(latent_dim, flattened_size)
        decoder_channels = list(reversed(conv_channels))  # 256, 128, 64, 32 by default
        decoder_layers = []
        for in_channels, out_channels in zip(decoder_channels[:-1], decoder_channels[1:]):
            decoder_layers += [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2),
            ]
        decoder_layers += [
            nn.ConvTranspose2d(decoder_channels[-1], 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # keeps pixels in [0, 1], which BCE needs
        ]
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, images):
        """Return the mean and log-variance of q(z | x)."""
        features = self.encoder(images).flatten(start_dim=1)
        return self.fc_mu(features), self.fc_logvar(features)

    def reparameterize(self, mu, logvar):
        """Sample z = mu + sigma * noise, so gradients can flow through mu and sigma."""
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        return mu + std * noise

    def decode(self, z):
        features = self.fc_dec(z)
        features = features.view(-1, self.conv_channels[-1], self.smallest_size, self.smallest_size)
        return self.decoder(features)

    def forward(self, images):
        mu, logvar = self.encode(images)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    @torch.no_grad()
    def sample(self, num_images, device):
        """Generate new images by decoding z drawn from the N(0, I) prior."""
        z = torch.randn(num_images, self.latent_dim, device=device)
        return self.decode(z)


def vae_loss(reconstruction, images, mu, logvar, beta=1.0):
    """BCE reconstruction loss plus beta times the KL term.

    Both terms are summed over pixels / latent dimensions and averaged over the batch.
    beta=1 gives the usual VAE loss. Since pixel values are continuous, BCE is only a
    stand-in for the true log-likelihood, so this is a surrogate for the negative ELBO.

    Returns (total_loss, reconstruction_loss, kl_loss).
    """
    batch_size = images.size(0)
    reconstruction_loss = F.binary_cross_entropy(reconstruction, images, reduction="sum") / batch_size
    # closed-form KL between N(mu, sigma^2) and N(0, I)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    total_loss = reconstruction_loss + beta * kl_loss
    return total_loss, reconstruction_loss, kl_loss
