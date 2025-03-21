import torch
import torch.nn as nn
import torch.nn.functional as F


l1_loss = torch.nn.L1Loss()


class SPADE(nn.Module):
    """Spatially-Adaptive Denormalization Layer"""

    def __init__(self, norm_nc, label_nc):
        """
        Parameters:
        - norm_nc: aantal feature map kanalen (uit de convolutionele laag)
        - label_nc: aantal segmentatiemaskerkanalen (input maskers)
        """
        super().__init__()
        # Standaard batch normalisatie zonder affine parameters
        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)

        # Convoluties voor SPADE-aanpassing
        self.mlp_gamma = nn.Conv2d(label_nc, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(label_nc, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        """Pas SPADE-normalisatie toe"""
        segmap = segmap.float()
        segmap = F.interpolate(segmap, size=x.shape[2:], mode="nearest")
        normalized = self.param_free_norm(x)  # Standaard batchnorm
        gamma = self.mlp_gamma(segmap)  # Bereken SPADE-gamma uit segmentatiemasker
        beta = self.mlp_beta(segmap)  # Bereken SPADE-beta uit segmentatiemasker
        return normalized * (1 + gamma) + beta  # Pas SPADE aan


class Block(nn.Module):
    """Basic convolutional building block

    Parameters
    ----------
    in_ch : int
        number of input channels to the block
    out_ch : int     
        number of output channels of the block
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)  # Batch norm voor stabiliteit #  # batch normalisation
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch) # 
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)  # Leaky ReLU als activatie #  # leaky ReLU


    def forward(self, x):
        """Performs a forward pass of the block
       
        x : torch.Tensor
            the input to the block
        torch.Tensor
            the output of the forward pass
        """
        # a block consists of two convolutional layers
        # with ReLU activations
        # use batch normalisation
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu(x)
        # 
        return x


class Encoder(nn.Module):
    """The encoder part of the VAE.

    Parameters
    ----------
    spatial_size : list[int]
        size of the input image, by default [64, 64]
    z_dim : int
        dimension of the latent space
    chs : tuple
        hold the number of input channels for each encoder block
    """

    def __init__(self, z_dim=256, chs=(1, 64, 128, 256), spatial_size=[64, 64]):
        super().__init__()
        # convolutional blocks
        self.enc_blocks = nn.ModuleList(
            [Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )
        self.pool = nn.MaxPool2d(2)  

        # max pooling
        _h, _w = spatial_size[0] // (2 ** (len(chs) - 1)), spatial_size[1] // (2 ** (len(chs) - 1))
        # height and width of images at lowest resolution level

        # flattening
        self.out = nn.Sequential(nn.Flatten(1), nn.Linear(chs[-1] * _h * _w, 2 * z_dim))

    def forward(self, x):
        """Performs the forward pass for all blocks in the encoder.

        Parameters
        ----------
        x : torch.Tensor
            input image to the encoder

        Returns
        -------
        list[torch.Tensor]    
            a tensor with the means and a tensor with the log variances of the
            latent distribution
        """

        for block in self.enc_blocks:
            x = block(x)
            x = self.pool(x)

        x = self.out(x)          
        return torch.chunk(x, 2, dim=1)  # 2 chunks, 1 each for mu and logvar


class Generator(nn.Module):
    """Generator of the VAE

    Parameters
    ----------
    z_dim : int 
        dimension of latent space
    chs : tuple
        holds the number of channels for each block
    h : int, optional
        height of image at lowest resolution level, by default 8
    w : int, optional
        width of image at lowest resolution level, by default 8    
    """

    def __init__(self, z_dim=256, chs=(256, 128, 64, 32), label_nc=1, h=8, w=8):

        super().__init__()
        self.chs = chs
        self.h = h  
        self.w = w  
        self.z_dim = z_dim  
        self.proj_z = nn.Linear(
            self.z_dim, self.chs[0] * self.h * self.w
        )  # fully connected layer on latent space
        self.reshape = lambda x: torch.reshape(
            x, (-1, self.chs[0], self.h, self.w)
        )  # reshaping

        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(chs[i], chs[i+1], kernel_size=4, stride=2, padding=1) for i in range(len(chs)-1)]
        )
            # : transposed convolution         
        # SPADE-convolutielagen
        self.spade_blocks = nn.ModuleList(
            [SPADE(chs[i+1], label_nc) for i in range(len(chs) - 1)]
        )   

        self.dec_blocks = nn.ModuleList(
            [Block(chs[i+1], chs[i+1]) for i in range(len(chs) - 1)]
        )
            # : conv block           
        self.head = nn.Sequential(
            nn.Conv2d(self.chs[-1], 1, 1),
            nn.Tanh(),
        )  # output layer

    def forward(self, z,segmap):
        """Performs the forward pass of generator

        Parameters
        ----------
        z : torch.Tensor
            input to the generator
        
        Returns
        -------
        x : torch.Tensor
        
        """
        x = self.proj_z(z) # : fully connected layer
        x = self.reshape(x) # : reshape to image dimensions
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x) # : transposed convolution
            x = self.spade_blocks[i](x, segmap)  # PAS SPADE TOE!
            x = self.dec_blocks[i](x) # : convolutional block
        return self.head(x)


class VAE(nn.Module):
    """A representation of the VAE

    Parameters
    ----------
    enc_chs : tuple 
        holds the number of input channels of each block in the encoder
    dec_chs : tuple 
        holds the number of input channels of each block in the decoder
    """
    def __init__(
        self,
        z_dim=256,
        enc_chs=(1, 64, 128, 256),
        dec_chs=(256, 128, 64, 32),
    ):
        super().__init__()
        self.encoder = Encoder(z_dim, enc_chs)
        self.generator = Generator(z_dim, dec_chs)



    def forward(self, x, segmap):
        """Performs a forwards pass of the VAE and returns the reconstruction
        and mean + logvar.

        Parameters
        ----------
        x : torch.Tensor
            the input to the encoder

        Returns
        -------
        torch.Tensor
            the reconstruction of the input image
        float
            the mean of the latent distribution
        float
            the log of the variance of the latent distribution
        """
        mu, logvar = self.encoder(x)
        latent_z = sample_z(mu, logvar)
        
        if segmap is None:
            raise ValueError("Segmentatiemasker (segmap) is None, dit mag niet gebeuren.")

        output = self.generator(latent_z, segmap)
        
        return output, mu, logvar


def get_noise(n_samples, z_dim, device="cpu"):
    """Creates noise vectors.
    
    Given the dimensions (n_samples, z_dim), creates a tensor of that shape filled with 
    random numbers from the normal distribution.

    Parameters
    ----------
    n_samples : int
        the number of samples to generate
    z_dim : int
        the dimension of the noise vector
    device : str
        the type of the device, by default "cpu"
    """
    return torch.randn(n_samples, z_dim, device=device)


def sample_z(mu, logvar):
    """Samples noise vector from a Gaussian distribution with reparameterization trick.

    Parameters
    ----------
    mu : float
        the mean of the distribution
    logvar : float
        the log of the variance of the distribution
    """
    eps = torch.randn(mu.size(), device=mu.device).to(mu.dtype)
    return (logvar / 2).exp() * eps + mu


def kld_loss(mu, logvar):
    """Computes the KLD loss given parameters of the predicted 
    latent distribution.

    Parameters
    ----------
    mu : float
        the mean of the distribution
    logvar : float
        the log of the variance of the distribution

    Returns
    -------
    float
        the kld loss

    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def vae_loss(inputs, recons, mu, logvar, beta=1):
    """Computes the VAE loss, sum of reconstruction and KLD loss

    Parameters
    ----------
    inputs : torch.Tensor
        the input images to the vae
    recons : torch.Tensor
        the predicted reconstructions from the vae
    mu : float
        the predicted mean of the latent distribution
    logvar : float
        the predicted log of the variance of the latent distribution

    Returns
    -------
    float
        sum of reconstruction and KLD loss
    """
    return l1_loss(inputs, recons) + beta*kld_loss(mu, logvar)
