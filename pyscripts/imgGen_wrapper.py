import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler, DataLoader
import numpy as np
from matplotlib import pyplot as plt
import argparse
import sys
import torchvision


# import scripts
from Diffusion import Diffusion
from Unet import UNet


# Seed numpy
np.random.seed(32)


def argparser(args):
    parser = argparse.ArgumentParser()
    model_selection = parser.add_argument_group()
    model_params = parser.add_argument_group()
    training_params = parser.add_argument_group()
    model_eval = parser.add_argument_group()
    dataset_params = parser.add_argument_group()

    model_selection.add_argument("--model", type=str, dest='model_sel', default='DDPM', help="Selection of Generative Model. Available options are: DDPM, GAN, LDM, EBM", required=True)

    model_params.add_argument("--noise-steps", type=int, dest="noise_steps", default=1000, help="Noise steps to be used in the DDPM forward and backward processes.")
    model_params.add_argument("--beta-start", type=int, dest="beta_start", default=1e-4, help="Set the beta start parameter for DDPM")
    model_params.add_argument("--beta_end", type=int, default=0.02, dest="beta_end", help="Set the beta end parameter for DDPM")

    training_params.add_argument("--epochs", type=int, dest="epochs", default=100, help="Set the number of training iterations")
    training_params.add_argument("--batch_size", type=int, dest="batch_size", default=64, help="Selects batch size for training.")
    training_params.add_argument("--lr", type=int, default=3e-4, dest="lr", help="Set training learning rate")
    # TODO: Add optimizer/loss options if necessary. If not remove the parameter
    training_params.add_argument("--optimizer", type=str, default="Adam", dest="opt", help="Select optimizer to be used for training. Accepted values: Adam...")
    training_params.add_argument("--loss", type=str, default="MSE", dest="loss", help="Select loss function for model training. Accepted values: MSE, ...")
    training_params.add_argument("--model_output", type=str, required=True, dest="model_output_dir", help="Provide a directory to story your trained model.")

    # TODO: enable give a path for train, test, val datasets
    dataset_params.add_argument("--MNIST", dest="MNIST", action='store_const', const=True, default=False, help="Selects MNIST dataset for training, validation, and test")
    dataset_params.add_argument("--train_dataset", dest="train_dataset", default=None, type=str, help="Give the path for a folder to use as training dataset")
    dataset_params.add_argument("--test_dataset", dest="test_dataset", default=None, type=str, help="Give the path for a folder to use as test dataset")
    dataset_params.add_argument("--val_dataset", dest="val_dataset", default=None, type=str, help="Give the path for a folder to use as validation dataset")
    dataset_params.add_argument("--img_size", dest="img_size", required=True, type=int, help="Provide the size of the images used for training and therefore generation size. Please use square images")

    # TODO: Add options for evaluation. E.g. FID

    argsvalue = parser.parse_args(args)
    return argsvalue


def split_indices(n, val_pct):
    n_val = int(val_pct*n)
    idxs = np.random.permutation(n)
    return idxs[n_val:], idxs[:n_val]


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([torch.cat([i for i in images.cpu()], dim=-1)], dim=-2).permute(1,2,0).cpu(), cmap='gray')
    plt.show()


def train(epochs, train_loader, val_loader, model, optimizer, diffusion, device, loss_fn):
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(epochs)):
        train_loss = 0
        for x, _ in train_loader:
            optimizer.zero_grad()
            x = x.to(device)
            t = diffusion.sample_timesteps(x.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(x, t)
            
            predicted_noise = model(x_t, t)
            loss = loss_fn(noise, predicted_noise)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        val_loss = 0
        with torch.no_grad():
            model.eval()
            for x, _ in val_loader:
                #print(type(x))
                #print(x.shape)
                x = x.to(device)
                t = diffusion.sample_timesteps(x.shape[0]).to(device)
                x_t, noise = diffusion.noise_images(x, t)
                predicted_noise = model(x_t, t)
                batch_val_loss = loss_fn(noise, predicted_noise)
                val_loss += batch_val_loss.item()
        
        # Sample only every 3 epochs for opt purposes
        if epoch%3 == 0:
            sampled_images = diffusion.sample(model, n=x.shape[0])
            plot_images(sampled_images)
            
        # Save loss
        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        print(f"Epoch [{epoch+1}/100] | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Retrieve dataset
    if args.MNIST:
        transforms = torchvision.transforms.Compose([
            #torchvision.transforms.Resize(80),
            #torchvision.transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)) 
        ])
        dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms, download=True)
        test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transforms, download=True)

        train_indices, val_indices = split_indices(len(dataset), 0.2)
        train_sampler = SubsetRandomSampler(train_indices)
        train_loader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=True)
        val_sampler = SubsetRandomSampler(val_indices)
        val_loader = DataLoader(dataset, args.batch_size, sampler=val_sampler)

        print("MNIST dataset loaded")

    # TODO
    elif args.train_dataset != None and args.test_dataset != None and args.val_dataset != None:
        pass
    
    else:
        print("Error retreiving train, test, and validation datasets. Please double check you provided a right path")
        raise

    # Init model
    if args.model_sel == "DDPM":
        model = UNet(device=device).to(device)
    elif args.model_sel == "LDM":
        pass
    else:
        print("Given model is not available")
        raise

    # Init opt
    if args.opt == "Adam":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    else:
        print("Given optimizer is not available")
        raise

    # Init loss fn
    if args.loss == "MSE":
        loss_fn = nn.MSELoss()
    else:
        print("Given loss function is not available")
        raise

    # Init diffusion or any other model
    if args.model_sel == "DDPM":
        diffusion = Diffusion(img_size=args.img_size, device=device)
    
    length = len(train_loader)

    train(args.epochs, train_loader, val_loader, model, optimizer, diffusion, device, loss_fn)

    torch.save(model.state_dict(), args.model_output_dir)



if __name__ == "__main__":
    argsvalue = argparser(sys.argv[1:])
    main(argsvalue)
