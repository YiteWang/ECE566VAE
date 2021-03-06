import torch
import torch.nn as nn
import torch.nn.functional as F
from models import vanillaVAE, IWAE, MIWAE
import argparse
from torchvision import transforms, utils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
import os
import time
import logging

vae_models = {'vanilla': vanillaVAE,
            'IWAE': IWAE,
            'MIWAE': MIWAE,}

def main(args):
    logging.basicConfig(filename= os.path.join(args.result_path,"VAE_results.log"),
                        format='%(asctime)s %(message)s',
                        filemode='w')
      
    # Creating a logger object
    logger=logging.getLogger()

    # Setting the threshold of logger to DEBUG
    logger.setLevel(logging.DEBUG)

    # Print all arguments
    logger.info('All arguments: \n')
    for arg_name in vars(args):
        logger.info('{}: {}'.format(arg_name, getattr(args, arg_name)))

    normalize = transforms.Lambda(lambda X: 2 * X - 1.)

    if args.dataset == 'celeba':
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(args.img_size),
                                            transforms.ToTensor(),
                                            normalize])
        train_data = CelebA(root=args.data_path, split="train", transform=transform, download=False)
        test_data = CelebA(root=args.data_path, split="test", transform=transform, download=False)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)
    else:
        raise NotImplementedError('Unsupported dataset')
    vae = vae_models[args.vae_type](input_channel = 3, h_channels=[32,64,128,256,512], latent_size=args.latent_size)
    vae.cuda()

    # logger.info('{}'.format(args.batch_size/len(train_loader.dataset)))
    
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    logger.info('start to train')
    # Start training
    for epoch in range(args.epochs):
        train(vae, train_loader, optimizer, epoch, args)
        test(vae, test_loader, epoch, args)
        scheduler.step()

def train(vae, train_loader, optimizer, epoch, args):
    logger = logging.getLogger(__name__)
    time_start = time.time()
    vae.train()
    train_loss = 0
    total_images = len(train_loader.dataset)
    for i, data in enumerate(train_loader):
        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        # loss = vae.compute_loss(x=images, labels=labels, coeff=1.0*args.batch_size/total_images, num_samples=args.num_samples)
        loss = vae.compute_loss(x=images, labels=labels, coeff=0.001, num_samples=args.num_samples, num_particles=args.num_particles)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if i % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * args.batch_size, total_images,
                100.0 * args.batch_size * i / total_images,
                loss.item()))
    time_end = time.time()
    logger.info('====> Epoch: {} Average loss: {:.4f}, Time used: {:.4f} s.'.format(epoch, train_loss / len(train_loader.dataset), time_end-time_start))

def test(vae, test_loader, epoch, args):
    logger = logging.getLogger(__name__)
    vae.eval()
    IWAE_64 = 0
    recon = 0
    total_images = len(test_loader.dataset)
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            # loss = vae.compute_loss(x=images, labels=labels, coeff=1.0*args.batch_size/total_images, num_samples=args.num_samples)
            IWAE_64_batch, recon_batch = vae.test_loss(x=images, labels=labels, coeff=0.001, num_particles=args.test_particles)
            IWAE_64 += IWAE_64_batch.item()
            recon += recon_batch.item()

        logger.info('Test set: IWAE_64 loss: {:.4f}'.format(IWAE_64 / len(test_loader.dataset)))
        logger.info('Test set: Recon loss: {:.4f}'.format(recon / len(test_loader.dataset)))
    
        # Sample one image from the test set and perform reconstruction tasks
        sample_data = next(iter(test_loader))
        images, labels = sample_data
        images, labels = images.cuda(), labels.cuda()
        recon = vae.forward(images)[0]
        utils.save_image(recon.data, os.path.join(args.result_path, 'recon_' + str(epoch) + '.png'), nrow=8, normalize=True)

        # Perform generation tasks
        generation = vae.generate(n_samples=100)
        utils.save_image(generation.data, os.path.join(args.result_path, 'gen_' + str(epoch) + '.png'), nrow=10, normalize=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("VAEs")
    parser.add_argument('--data_path', type=str, default='/home/yitew2/data/', help='Path to dataset')
    parser.add_argument('--result_path', type=str, default='./results/', help='Path to results')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--dataset', type=str, default='celeba', help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--test_batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--latent_size', type=int, default=128, help='Latent dimension')
    parser.add_argument('--vae_type', type=str, default='vanilla', choices=['vanilla','IWAE','MIWAE'], help='Type of VAE')
    parser.add_argument('--img_size', type=int, default=64, help='Image size after transformations')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataset')
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples for IWAE')
    parser.add_argument('--num_particles', type=int, default=5, help='Number of particles for MIWAE/IWAE')
    parser.add_argument('--test_particles', type=int, default=32, help='Number of particles during testing')
    parser.add_argument('--stamp', type=str, help='Stamp for saving results')
    args = parser.parse_args()

    if args.stamp:
        args.result_path = os.path.join(args.result_path, args.stamp)
    else:
        timestamp = "{:}".format(time.strftime('%h-%d-%C_%H-%M-%s', time.gmtime(time.time())))
        args.result_path = os.path.join(args.result_path, timestamp)
    
    assert torch.cuda.is_available(), 'CUDA is not available.'

    if not os.path.isdir(args.result_path):
        os.makedirs(args.result_path)

    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    main(args)