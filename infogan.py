# -*- coding: utf-8 -*-
import os
import pdb
import pickle
import argparse
import time

import warnings
warnings.filterwarnings("ignore")

# Numpy & Scipy imports
import numpy as np
import scipy
import scipy.misc

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# Local imports
import utils
from data_loader import get_mnist_data, get_celeba_loader


#SEED = 11
#
## Set the random seed manually for reproducibility.
#np.random.seed(SEED)
#torch.manual_seed(SEED)
#if torch.cuda.is_available():
#    torch.cuda.manual_seed(SEED)


class log_gaussian:
  def __call__(self, x, mu, var):
      # mu = mean
      # var = exponential standard deviation
      var = var.pow(2)
      logli = -0.5*(var.mul(2*np.pi)+1e-6).log() - \
      (x-mu).pow(2).div(var.mul(2.0)+1e-6)
      
      return logli.sum(1).mean().mul(-1)

def print_models(DQ, D, Q, G):
    """Prints model information for the generators and discriminators.
    """
    print("                    DQ                  ")
    print("---------------------------------------")
    print(DQ)
    print("---------------------------------------")

    print("                    D                  ")
    print("---------------------------------------")
    print(D)
    print("---------------------------------------")
    
    print("                    Q                  ")
    print("---------------------------------------")
    print(Q)
    print("---------------------------------------")

    print("                    G                  ")
    print("---------------------------------------")
    print(G)
    print("---------------------------------------")
    

def create_model(opts):
    """Builds the generators and discriminators.
    """
    
    if opts.dataset == 'CelebA':
        from models_celeba import Generator, Discriminator, Recognition, SharedPartDQ
    else: # This is the MNIST dataset (default)
        from models import Generator, Discriminator, Recognition, SharedPartDQ
        
        
    G = Generator(noise_size=opts.noise_size)
    D = Discriminator()
    
    # Amount of variables is amount of categoricals times their sizes
    nr_latent_cat_values = opts.cat_dims_count * opts.cat_dim_size
    
    Q = Recognition(categorical_dims=nr_latent_cat_values, continuous_dims=opts.cont_dims_count)
    DQ = SharedPartDQ()
    
    if opts.display_debug:
        print_models(DQ, D, Q, G)

    if torch.cuda.is_available():
        G.cuda()
        D.cuda()
        Q.cuda()
        DQ.cuda()
        if opts.display_debug:
            print('Models moved to GPU.')

    return G, D, Q, DQ


def checkpoint(iteration, G, D, Q, DQ, opts):
    """Saves the parameters of all the models.
    """
    # Save locally or to your google drive:
    if opts.colab:
        dir_path =F"/content/gdrive/My Drive/{opts.directory}/model/"
    else:
        dir_path = os.path.join(opts.directory, 'model')
    
    G_path = os.path.join(dir_path, 'G.pkl')
    D_path = os.path.join(dir_path, 'D.pkl')
    Q_path = os.path.join(dir_path, 'Q.pkl')
    DQ_path = os.path.join(dir_path, 'DQ.pkl')
    Opts_path = os.path.join(dir_path, 'opts.pkl')
        
    torch.save(G.state_dict(), G_path)
    torch.save(D.state_dict(), D_path)
    torch.save(Q.state_dict(), Q_path)
    torch.save(DQ.state_dict(), DQ_path)
    pickle.dump( opts, open( Opts_path, "wb" ) )
    
    return

    
def load_checkpoint(opts):
    """Loads the models from checkpoints.
    """
    # If none was selected but this function has been called (when externally called),
    # we assume the opts.directory is the place to find the models
    if opts.load == None:
        print("None selected, thus we assume we load from checkpoint_dir.")
        load_path = os.path.join(opts.directory, 'model')
    else:
        print("Use opts.load")
        load_path = os.path.join(opts.load, 'model')
    
    G_path = os.path.join(load_path, 'G.pkl')
    D_path = os.path.join(load_path, 'D.pkl')
    Q_path = os.path.join(load_path, 'Q.pkl')
    DQ_path = os.path.join(load_path, 'DQ.pkl')

    G, D, Q, DQ = create_model(opts)

    G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
    D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
    Q.load_state_dict(torch.load(Q_path, map_location=lambda storage, loc: storage))
    DQ.load_state_dict(torch.load(DQ_path, map_location=lambda storage, loc: storage))

    if torch.cuda.is_available():
        G.cuda()
        D.cuda()
        Q.cuda()
        DQ.cuda()
        if opts.display_debug:
            print('Models moved to GPU.')

    return G, D, Q, DQ


def create_image_grid(array, ncols=None):
    """ Creates an image grid of images in array.
    """
    num_images, channels, cell_h, cell_w = array.shape

    if not ncols:
        ncols = int(np.sqrt(num_images))
    nrows = int(np.math.floor(num_images / float(ncols)))
    result = np.zeros((cell_h*nrows, cell_w*ncols, channels), dtype=array.dtype)
    for i in range(0, nrows):
        for j in range(0, ncols):
            result[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w, :] = array[i*ncols+j].transpose(1, 2, 0)

    if channels == 1:
        result = result.squeeze()
    return result


def save_samples(G, fixed_noise, iteration, opts, extra_name):
    """ Saves samples of G(fixed_noise) in a grid to disk.
    """
    generated_images = G(fixed_noise)
    generated_images = utils.to_data(generated_images)
    
    grid = create_image_grid(generated_images, ncols=10)

    # Save to google drive?
    if opts.colab:
        dir_path = F"/content/gdrive/My Drive/{opts.directory}/samples/"
    else:
        dir_path = os.path.join(opts.directory, 'samples')
        
    path = os.path.join(dir_path, 'c{}_sample-{:06d}.png'.format(extra_name, iteration))
    scipy.misc.imsave(path, grid)
    if opts.display_debug:
        print('Saved {}'.format(path))


def sample_noise(opts):
    """
    Generate a PyTorch Variable of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - noise_size: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Variable of shape (batch_size, noise_size, 1, 1) containing uniform
      random noise in the range (-1, 1).
    """
    # Create full tensor with random values
    batch_noise = utils.to_var(torch.rand(batch_size, opts.noise_size) * 2 - 1)
    
    # Save the target categories to this, already fully created, placeholder
    target_categories = -1*np.ones(shape=(batch_size, opts.cat_dims_count))
    
    # For each categorical value switch the corresponding noise values 
    # for a onehot encoded representation of the category
    for cat_ind in range(opts.cat_dims_count):
        # Generate random ints
        random_categories = np.random.randint(low=0, high=opts.cat_dim_size, size=batch_size)        
        # Append random_categories to target variable
        target_categories[:, cat_ind] = random_categories
        # Apply onehot encoding
        onehot_categories = np.eye(opts.cat_dim_size)[random_categories]
        # Define indexes to place this set    
        ind_from = cat_ind * opts.cat_dim_size
        ind_to = ind_from + opts.cat_dim_size
        # Place 
        batch_noise[:, ind_from:ind_to] = torch.tensor(onehot_categories)
        
    # For each continuous dimension, add random values and also output these
    if opts.cont_dims_count > 0:
        cont_latent_variables = torch.zeros([batch_size, opts.cont_dims_count]).uniform_() * 2 - 1
        batch_noise[:, ind_to:ind_to + opts.cont_dims_count] = cont_latent_variables
        return batch_noise, torch.LongTensor(target_categories).cuda(), cont_latent_variables.cuda()
    else:
        return batch_noise, torch.LongTensor(target_categories).cuda(), None

def get_fixed_noise(opts, var=0):
    """
    Generate a PyTorch Variable of uniform random noise. Assumes a 10x10 grid.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - noise_size: Integer giving the dimension of noise to generate.
    - var:        Integer that makes extra options possible depending on the dataset

    Output:
    - A PyTorch Variable of shape (batch_size, dim, 1, 1) containing uniform
      random noise in the range (-1, 1).
    """
    # Use this to covert value to one hot encoding
    onehot_categories = np.eye(10)
    
    # Plots are per row, 
    # So first 10 entries are row 1
    # Second 10 entries are row 2
    
    
    if opts.dataset == 'CelebA':
        # If var=-1: create noise that loops through all categorical values 
        # while having the same 'other noise' (called 'z')
        if var == -1:
            batch_noise = utils.to_var(torch.rand(1, opts.noise_size) * 2 - 1).repeat(100,1)
        # else just make each row have the same noise, but different each row
        else:
            batch_noise = utils.to_var(torch.rand(100, opts.noise_size) * 2 - 1)
        
            # Set the noise of the 9 entries following every 10th entry to be equal
            for ind in range(10):
                batch_noise[ind*10:(ind+1)*10,:] = batch_noise[ind*10,:]
            
        # Set all categorical values to their middles (6th) value
        new_rows = torch.tensor(onehot_categories[5]).repeat(100, opts.cat_dims_count)
        batch_noise[:, :opts.cat_dim_size * opts.cat_dims_count] = new_rows        
        
        # Create new rows for one categorical
        # That is 10 rows with incrementing categorical value
        new_rows = torch.tensor(onehot_categories)
        if var == -1:
            for i in range(10):
                ind_from = i*10
                ind_to = (i+1)*10                  
                batch_noise[ind_from:ind_to, ind_from:ind_to] = new_rows
        else:
            #Set the var'th category to be changed per row
            ind_from = var*10
            ind_to = (var+1)*10  
            batch_noise[:, ind_from:ind_to] = new_rows.repeat(10,1)
        
    else:
        # In case of default or MNIST
        # Here var determines which continuous variable is changed
        
        # Create one set of noise values and repeat 100 times
        batch_noise = (torch.rand(1, opts.noise_size) * 2 - 1).repeat(100,1)
        
        # Set all values for the categorical to 0        
        batch_noise[:, :opts.cat_dim_size * opts.cat_dims_count] = 0    
           
        # Set all categorical to each value 
        for ind in range(10):
            batch_noise[ind*10:(ind+1)*10, :10] = torch.tensor(onehot_categories[ind])
            
        # Set all latent continous variables to 0
        batch_noise[:,10:10+opts.cont_dims_count] = 0
        # Set var'th continous variable:
        batch_noise[:,10+var] = torch.linspace(-1, 1, steps=10).repeat(10)
    
    return utils.to_var(batch_noise).cuda()

def training_loop(train_dataloader, opts):
    """Runs the training loop.
        * Saves checkpoints every opts.checkpoint_every iterations
        * Saves generated samples every opts.sample_every iterations
    """

    # Create new model, or load in a previous one
    if opts.load == None:
        # Create models
        G, D, Q, DQ = create_model(opts)
    else:
        # Load models
        G, D, Q, DQ = load_checkpoint(opts)

    # Create optimizers for the generators and discriminators
    d_optimizer = optim.Adam([{'params':DQ.parameters()}, {'params':D.parameters()}], opts.lrD, [opts.beta1, opts.beta2])
    g_optimizer = optim.Adam([{'params':G.parameters()}, {'params':Q.parameters()}], opts.lrG, [opts.beta1, opts.beta2])

    # Generate fixed noise for sampling from the generator    
    fixed_noise = []
    if opts.dataset == 'CelebA':
        # All 10 categorical values
        for i in range(opts.cat_dims_count):    # Depending on what kind of noise you want to put in
#        for i in range(opts.cont_dims_count):  # Depending on what kind of noise you want to put in
            fixed_noise.append(get_fixed_noise(opts, var=i))
        # Add an overview:
        fixed_noise.append(get_fixed_noise(opts, var=-1))
    else: # Do MNIST (default):
        for i in range(opts.cont_dims_count):
            fixed_noise.append(get_fixed_noise(opts, var=i))
        
    iteration = 1

    total_train_iters = opts.num_epochs * len(train_dataloader)
    
    # Loss functions:
    loss_criterion = torch.nn.BCELoss(reduction='elementwise_mean')
    if torch.cuda.is_available():
        loss_criterion.cuda()
        zeros_label = torch.autograd.Variable(torch.zeros(batch_size).float().cuda())
        ones_label = torch.autograd.Variable(torch.ones(batch_size).float().cuda())
        criterion_Q_dis = nn.CrossEntropyLoss().cuda()
        if opts.display_debug:
            print('MSE loss moved to GPU.')
    else:
        zeros_label = torch.autograd.Variable(torch.zeros(batch_size).float())
        ones_label = torch.autograd.Variable(torch.ones(batch_size).float())
        criterion_Q_dis = nn.CrossEntropyLoss()
    
    criterion_Q_con = log_gaussian()
    
    for epoch in range(opts.num_epochs):

        for real_images, real_labels in train_dataloader:            
            
            real_images, _ = utils.to_var(real_images), utils.to_var(real_labels).long().squeeze()
            

            ################################################
            ###         TRAIN THE DISCRIMINATOR         ####
            ################################################
            d_optimizer.zero_grad()
            
            # First do shared part, then apply discriminator
            D_real_images = D(DQ(real_images))
                        
            D_real_loss = loss_criterion(D_real_images, ones_label[:real_images.size()[0]])

            batch_noise, _, _= sample_noise(opts)
            fake_images = G(batch_noise)
            
            D_fake_images = D(DQ(fake_images))
            D_fake_loss = loss_criterion(D_fake_images, zeros_label[:fake_images.size()[0]])
            
            D_total_loss = (D_real_loss + D_fake_loss)
            
            D_total_loss.backward()
            d_optimizer.step()

            ###########################################
            ###          TRAIN THE GENERATOR        ###
            ###########################################
            g_optimizer.zero_grad()
            
            batch_noise, category_target, continous_target = sample_noise(opts)
            fake_images = G(batch_noise)
            
            DQ_fake_images = DQ(fake_images)
            
            G_loss_fake = loss_criterion(D(DQ_fake_images), ones_label[:fake_images.size()[0]])
            
            cat, cont_mu, cont_sigma = Q(DQ_fake_images)
            
            # Reshape cat to properly reflext each categorical 
            cat = cat.view(-1, opts.cat_dim_size, opts.cat_dims_count)
            
            # Calculate loss for each categorical dimension            
            dis_loss = criterion_Q_dis(cat, category_target)
            
            # If there are continuous variables used:
            if opts.cont_dims_count > 0:
                con_loss = criterion_Q_con(continous_target, cont_mu, cont_sigma)*opts.lambda_value
            else:
                con_loss = 0
 
            G_loss_total = G_loss_fake + dis_loss + con_loss

            G_loss_total.backward()
            g_optimizer.step()


            # Print the log info
            if iteration % opts.log_step == 0:
                if opts.cont_dims_count > 0:
                    print('Iteration [{:4d}/{:4d}] | D_real_loss: {:6.4f} | D_fake_loss: {:6.4f} | G_loss_fake: {:6.4f} | G_dis_loss: {:6.4f} | G_con_loss: {:6.4f}'.format(
                            iteration, total_train_iters, D_real_loss.item(), D_fake_loss.item(), G_loss_fake.item(), dis_loss.item(), con_loss.item()))
                else:
                    print('Iteration [{:4d}/{:4d}] | D_real_loss: {:6.4f} | D_fake_loss: {:6.4f} | G_loss_fake: {:6.4f} | G_dis_loss: {:6.4f}'.format(
                            iteration, total_train_iters, D_real_loss.item(), D_fake_loss.item(), G_loss_fake.item(), dis_loss.item()))

            # Save the generated samples
            if iteration % opts.sample_every == 0:
                for i in range(len(fixed_noise)):
                    save_samples(G, fixed_noise[i], iteration, opts, i)

            # Save the model parameters
            if iteration % opts.checkpoint_every == 0:
                checkpoint(iteration, G, D, Q, DQ, opts)

            iteration += 1


def main(opts):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """    
    # Select right dataset:
    if opts.dataset == 'CelebA':
        train_dataloader = get_celeba_loader(opts)
    else: # Default is MNIST
        train_dataloader, test_dataloader = get_mnist_data(opts)
    
    # Create checkpoint and sample directories
    # For colab it is possible to save results to google drive
    if opts.colab:
        dir_path =F"/content/gdrive/My Drive/{opts.directory}/model/"
        utils.create_dir(dir_path)
        dir_path =F"/content/gdrive/My Drive/{opts.directory}/samples/"
        utils.create_dir(dir_path)
    else:
        utils.create_dir(opts.directory)
        utils.create_dir(os.path.join(opts.directory, 'samples'))
        utils.create_dir(os.path.join(opts.directory, 'model'))

    # Some timing statistics
    startTime = time.time()
    training_loop(train_dataloader, opts)
    print("Training {} epochs took {:0.2f}s.".format(opts.num_epochs, time.time()-startTime))
    print("That is {:0.2f}s per epoch.".format((time.time()-startTime)/float(opts.num_epochs)))


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--dataset', type=str, default = 'MNIST', help='Select dataset, choose between MNIST or CelebA')
    parser.add_argument('--directory', type=str, default='Saved_run')
    
    # Training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')
#    parser.add_argument('--cont_dims_count', type=int , default=2)
    parser.add_argument('--lambda_value', type=int , default=1)
    
    # Directories and checkpoint/sample iterations
    parser.add_argument('--display_debug', type=bool, default=False)
    parser.add_argument('--log_step', type=int , default=100)
    parser.add_argument('--sample_every', type=int , default=500)
    parser.add_argument('--checkpoint_every', type=int , default=500)
    
    # Want to load a previously run model? Give the parent directory
    # Want to start over? Just set it as None
    parser.add_argument('--load', type=str, default=None)
    
    # Save files to google drive?
    # Set to True and follow the tooltips in the colab environment to save to drive!
    parser.add_argument('--colab', type=bool, default=False)
    
    return parser


if __name__ == '__main__':

    parser = create_parser()
    opts = parser.parse_args()

    batch_size = opts.batch_size
    
    # Set some default values based on the dataset
    if opts.dataset == 'CelebA':
        from models_celeba import Generator, Discriminator, Recognition, SharedPartDQ
        opts.noise_size = 228
        opts.cont_dims_count = 0
        opts.cat_dim_size = 10
        opts.cat_dims_count = 10
        opts.lrD = 2e-4
        opts.lrG = 1e-3
        opts.beta1 = 0.5
        opts.beta2 = 0.99
    else: # This is the MNIST dataset (default)
        from models import Generator, Discriminator, Recognition, SharedPartDQ
        opts.noise_size = 74
        opts.cont_dims_count = 2
        
        # DONT CHANGE THESE:
        opts.cat_dim_size = 10     # Size of each categorical value
        opts.cat_dims_count = 1   # Amount of categoricals
        
        opts.lrD = 2e-4
        opts.lrG = 1e-3
        opts.beta1 = 0.5
        opts.beta2 = 0.99
        
    # Setup connection to google drive to save data
    if opts.colab:
        from google.colab import drive
        drive.mount('/content/gdrive')
    
    print(opts)
    main(opts)