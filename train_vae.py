import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

import utils
import vae

# to ensure reproducible training/validation split
random.seed(41)

# find out if a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# directorys with data and to store training checkpoints and logs
DATA_DIR = Path.cwd() / r"C:\Users\20202686\Documents\BiomedicalEngineering\Master\Year 1\8DM20 Capita\DevelopmentData\DevelopmentData"
CHECKPOINTS_DIR = Path.cwd() / "vae_model_weights"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOGDIR = "vae_runs"

# training settings and hyperparameters
NO_VALIDATION_PATIENTS = 2
IMAGE_SIZE = [64, 64]
BATCH_SIZE = 32
N_EPOCHS = 5
DECAY_LR_AFTER = 50
LEARNING_RATE = 1e-4
DISPLAY_FREQ = 10

# dimension of VAE latent space
Z_DIM = 256

# function to reduce the
def lr_lambda(the_epoch):
    """Function for scheduling learning rate"""
    return (
        1.0
        if the_epoch < DECAY_LR_AFTER
        else 1 - float(the_epoch - DECAY_LR_AFTER) / (N_EPOCHS - DECAY_LR_AFTER)
    )


# find patient folders in training directory
# excluding hidden folders (start with .)
patients = [
    path
    for path in DATA_DIR.glob("*")
    if not any(part.startswith(".") for part in path.parts)
]
random.shuffle(patients)
# patients = patients[:3]

# split in training/validation after shuffling
partition = {
    "train": patients[:-NO_VALIDATION_PATIENTS],
    "validation": patients[-NO_VALIDATION_PATIENTS:],
}

# load training data and create DataLoader with batching and shuffling

dataset = utils.ProstateMRDataset(partition["train"], IMAGE_SIZE, valid=True) # in my experiments the augmentations
# did not help, so I set valid=True to disable them
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)

# load validation data
valid_dataset = utils.ProstateMRDataset(partition["validation"], IMAGE_SIZE, valid=True)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)

# initialise model, optimiser
vae_model = vae.VAE(z_dim=Z_DIM).to(device) #  
optimizer = torch.optim.Adam(vae_model.parameters(), lr=LEARNING_RATE) #  
# add a learning rate scheduler based on the lr_lambda function
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda) # 

# training loop
writer = SummaryWriter(log_dir=TENSORBOARD_LOGDIR)  # tensorboard summary
for epoch in range(N_EPOCHS):
    current_train_loss = 0.0
    current_valid_loss = 0.0
    
    #  
    # training iterations
    for x_real, y_real in tqdm(dataloader, position=0):
        x_real = x_real.to(device)

        optimizer.zero_grad()
        x_recon, mu, logvar = vae_model(x_real, y_real.to(device).float())  # SPADE gebruikt segmentatiemasker

        loss = vae.vae_loss(x_real, x_recon, mu, logvar)
        loss.backward()
        optimizer.step()
        
        current_train_loss += loss.item()


    writer.add_scalar("Loss/train", current_train_loss / len(dataloader), epoch)
    scheduler.step() # step the learning step scheduler
    
    # evaluate validation loss
    with torch.no_grad():
        vae_model.eval()
        for x_real, y_real in tqdm(valid_dataloader, position=0):
            x_real = x_real.to(device)
            x_recon, mu, logvar = vae_model(x_real, y_real.to(device))

            loss = vae.vae_loss(x_real, x_recon, mu, logvar)
            current_valid_loss += loss.item()
        
        # write to tensorboard log
        writer.add_scalar(
            "Loss/validation", current_valid_loss / len(valid_dataloader), epoch
        )
    

        # save examples of real/fake images
        if (epoch + 1) % DISPLAY_FREQ == 0:
            img_grid = make_grid(
                torch.cat((x_recon[:5], x_real[:5])), nrow=5, padding=12, pad_value=-1
            )
            writer.add_image(
                "Real_fake", np.clip(img_grid[0][np.newaxis], -1, 1) / 2 + 0.5, epoch + 1
            )
            noise = vae.get_noise(10, Z_DIM, device)
            segmap = torch.randn(10, 1, 64, 64, device=device)
            image_samples = vae_model.generator(noise, segmap)  # Forward pass with noise
            # : sample noise 
            # : generate 10 images and display
            img_grid = make_grid(
                torch.cat((image_samples[:5].cpu(), image_samples[5:].cpu())),
                nrow=5,
                padding=12,
                pad_value=-1,
            )
            writer.add_image(
                "Samples",
                np.clip(img_grid[0][np.newaxis], -1, 1) / 2 + 0.5,
                epoch + 1,
            )
        vae_model.train()

weights_dict = {k: v.cpu() for k, v in vae_model.state_dict().items()}
torch.save(
    weights_dict,
    CHECKPOINTS_DIR / "vae_model.pth",
)

# Laad het model na training
vae_model.load_state_dict(torch.load(CHECKPOINTS_DIR / "vae_model.pth", map_location=device))
vae_model.eval()  # Zet het model in evaluatiemodus

# Print modelstructuur om te checken
print("VAE-model geladen")
print(vae_model)

import matplotlib.pyplot as plt

# Genereer een random vector in de latente ruimte
z = vae.get_noise(1, Z_DIM, device)  # 1 sample uit de latente ruimte
segmap = torch.randn(1, 1, 64, 64, device=device)  # Random segmentatiemasker genereren
generated_image = vae_model.generator(z, segmap).cpu().detach().squeeze()

# Toon het gegenereerde beeld
plt.imshow(generated_image, cmap="gray")
plt.axis("off")
plt.title("Gegenereerd beeld door de VAE")
plt.show()
