"""
explain.py

Script to learn explanatory perturbation mask.
"""
import torch
from torchvision import models, transforms
import numpy as np
from PIL import Image
from utils import (get_device, set_gpu, blur_input_tensor, create_area_target,
                   get_short_class_name, tv_norm, area_norm, plot_curve, save)


def perturb_explanation(image='images/flute.jpg',
                        arch='alexnet',
                        input_size=227,
                        learning_rate=1e-1,
                        epochs=300,
                        l1_lambda=1e-2,
                        tv_lambda=1e-4,
                        tv_beta=3.,
                        blur_size=11,
                        blur_sigma=10.,
                        mask_size=28,
                        noise_std=0.,
                        area_lambda=0.,
                        area=1.,
                        ease_in_area=False,
                        ease_rate=1.,
                        print_iter=25):
    """Generate a perturbation mask for a given image and CNN.

    Args:
        image: String, path to an image file.
        arch: String, name of PyTorch CNN architecture.
        input_size: Integer, length of the side of CNN input.
        learning_rate: Float, learning rate to use.
        epochs: Integer, number of iterations for which to run optimization.
        l1_lambda: Float, coefficient term to weight L1 loss.
        tv_lambda: Float, coefficient term to weight (total variation) TV loss.
        tv_beta: Float, beta term to use in TV loss.
        blur_size: Integer, length of the side of the blur kernel.
        blur_sigma: Float, standard deviation for Gaussian blur kernel.
        mask_size: Integer, length of the side of the mask to learn.
        noise_std: Float, standard deviation for additive, Gaussian noise
            to add to the image when perturbing with the mask.
        area_lambda: Float, coefficient term to weight area loss.
        area: Float, target percentage that the mask should match
            (i.e., area = 0.7 denotes that 70% of the mask should be 0
            and 30% of the mask should be 1).
        ease_in_area: Boolean, if True, linearly ease in the area loss by
            the ease_rate over training epochs.
        ease_rate: Float, rate at which to ease in the area loss.
        print_iter: Integer, frequency at which to print log messages.

    Returns:
        final_mask: Numpy array, learned, upsampled mask with shape
            `input_size x input_size`.
    """
    # Get CUDA device.
    device = get_device()

    # Load pre-trained model.
    model = models.__dict__[arch](pretrained=True).to(device)

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_sigma = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_sigma),
    ])

    orig_img = Image.open(image)

    img = transform(orig_img).unsqueeze(0)
    blurred_img = blur_input_tensor(
        img, kernel_size=blur_size, sigma=blur_sigma)
    img = img.to(device)
    blurred_img = blurred_img.to(device)

    mask = torch.ones((1, 1, mask_size, mask_size),
                      requires_grad=True,
                      device=device)
    area_target = create_area_target(mask, area).to(device)

    upsample = torch.nn.Upsample(
        size=(input_size, input_size), mode='bilinear').to(device)
    softmax = torch.nn.Softmax(dim=1).to(device)

    optimizer = torch.optim.Adam([mask], lr=learning_rate)

    target = softmax(model(img))
    category = np.argmax(target.cpu().data.numpy())
    print("Category with highest probability: %s (%.4f)"
          % (get_short_class_name(category),
             target.cpu().data.numpy()[0][category]))
    print("Optimizing...")

    for i in range(epochs):
        upsampled_mask = upsample(mask)
        expanded_dims = (1, 3, upsampled_mask.size(2), upsampled_mask.size(3))
        upsampled_mask = upsampled_mask.expand(expanded_dims)

        # Use the mask to perturbated the input image.
        perturbated_input = img * upsampled_mask + \
            blurred_img * (1-upsampled_mask)

        noise = noise_std * torch.randn((1, 3, input_size, input_size),
                                        device=device)
        perturbated_input = perturbated_input + noise

        outputs = softmax(model(perturbated_input))
        l1_loss = l1_lambda*torch.mean(torch.abs(1-mask))
        tv_loss = tv_lambda*tv_norm(mask, tv_beta)
        if ease_in_area:
            curr_area_lambda = area_lambda * min(
                ease_rate*i / float(epochs), 1.)
        else:
            curr_area_lambda = area_lambda
        area_loss = curr_area_lambda*area_norm(mask, area_target)
        class_loss = outputs[0, category]
        tot_loss = l1_loss + tv_loss + class_loss + area_loss

        optimizer.zero_grad()
        tot_loss.backward()
        optimizer.step()

        mask.data.clamp_(0, 1)

        if i % print_iter == 0:
            print('Epoch %d\tL1 Loss %f\tTV Loss %f\tArea Loss %f\t'
                  'Class Loss %f\tTot Loss %f'
                  % (i, l1_loss.item(), tv_loss.item(), area_loss.item(),
                     class_loss.item(), tot_loss.item()))
            plot_curve(mask, area_target, suffix='_%d' % i)

    upsampled_mask = upsample(mask)
    title = 'Softmax=%.4f (area=%.4f)' % (class_loss.item(), area)
    save(upsampled_mask, orig_img, title=title)
    return np.squeeze(upsampled_mask.data.cpu().numpy())


if __name__ == '__main__':
    import argparse
    import sys
    import traceback
    try:
        def str2bool(value):
            """Converts string to bool."""
            value = value.lower()
            if value in ('yes', 'true', 't', '1'):
                return True
            if value in ('no', 'false', 'f', '0'):
                return False
            raise ValueError('Boolean argument needs to be true or false. '
                             'Instead, it is %s.' % value)
        parser = argparse.ArgumentParser(description='Learn perturbation mask')
        parser.register('type', 'bool', str2bool)
        parser.add_argument('--image', default='images/flute.jpg',
                            help='path of input image')
        parser.add_argument('--architecture', default='alexnet',
                            help='name of CNN architecture (choose from '
                                 'PyTorch pretrained networks')
        parser.add_argument('--input_size', type=int, default=227,
                            help='CNN image input size')
        parser.add_argument('--learning_rate', type=float, default=1e-1,
                            help='learning rate (for Adam optimization)')
        parser.add_argument('--epochs', type=int, default=300,
                            help='number of iterations for which to train.')
        parser.add_argument('--l1_lambda', type=float, default=1e0,
                            help='L1 regularization lambda coefficient term')
        parser.add_argument('--tv_lambda', type=float, default=1e1,
                            help='TV regularization lambda coefficient term')
        parser.add_argument('--tv_beta', type=float, default=3,
                            help='TV beta hyper-parameter')
        parser.add_argument('--area_lambda', type=float, default=1e1,
                            help='Area lambda coefficient term')
        parser.add_argument('--area', type=float, default=0.975)
        parser.add_argument('--blur_size', type=int, default=11,
                            help='Gaussian kernel blur size')
        parser.add_argument('--blur_sigma', type=int, default=10,
                            help='Gaussian blur sigma')
        parser.add_argument('--mask_size', type=int, default=28,
                            help='Learned mask size')
        parser.add_argument('--noise', type=float, default=0,
                            help='Amount of random Gaussian noise to add')
        parser.add_argument('--ease_in_area', type='bool', default=False,
                            help='Should area loss be linearly eased in.')
        parser.add_argument('--ease_rate', type=float, default=2.,
                            help='Rate at which area loss should be eased in.')
        parser.add_argument('--gpu', type=int, nargs='*', default=None)

        args = parser.parse_args()
        set_gpu(args.gpu)
        perturb_explanation(image=args.image,
                            arch=args.architecture,
                            input_size=args.input_size,
                            learning_rate=args.learning_rate,
                            epochs=args.epochs,
                            l1_lambda=args.l1_lambda,
                            tv_lambda=args.tv_lambda,
                            tv_beta=args.tv_beta,
                            mask_size=args.mask_size,
                            noise_std=args.noise,
                            area_lambda=args.area_lambda,
                            area=args.area,
                            ease_in_area=args.ease_in_area,
                            ease_rate=args.ease_rate)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
