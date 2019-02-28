"""
utils.py

Contains utility functions.
"""
import math
import os

from skimage.transform import resize
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')


def get_linear_scheduler(optimizer, num_epochs, last_epoch=-1):
    """Return LambdaLR scheduler that follows a linear decay schedule."""
    def lr_lambda_func(epoch):
        """Linear decay function."""
        return 1 - epoch / num_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=lr_lambda_func,
                                                  last_epoch=last_epoch)
    return scheduler


def create_dir_if_necessary(path, is_dir=False):
    """Create directory to path if necessary."""
    parent_dir = get_parent_dir(path) if not is_dir else path
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)


def get_parent_dir(path):
    """Return parent directory of path."""
    return os.path.abspath(os.path.join(path, os.pardir))


def get_device():
    """Return torch.device based on if cuda is available."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return device


def blur_input_tensor(tensor, kernel_size=11, sigma=5.0):
    """Blur tensor with a 2D gaussian blur.

    Args:
        tensor: torch.Tensor, 3 or 4D tensor to blur.
        kernel_size: int, size of 2D kernel.
        sigma: float, standard deviation of gaussian kernel.

    Returns:
        4D torch.Tensor that has been smoothed with gaussian kernel.
    """
    ndim = len(tensor.shape)
    if ndim == 3:
        tensor = tensor.unsqueeze(0)
    assert ndim == 4
    num_channels = tensor.shape[1]
    device = tensor.device

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) * torch.exp(
        -1*torch.sum((xy_grid - mean)**2., dim=-1) /
        (2.*variance)
    )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(num_channels, 1, 1, 1)

    padding = nn.ReflectionPad2d(int(mean))
    gaussian_filter = nn.Conv2d(in_channels=num_channels,
                                out_channels=num_channels,
                                kernel_size=kernel_size,
                                groups=num_channels,
                                bias=False)
    padding.to(device)
    gaussian_filter.to(device)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    smoothed_tensor = gaussian_filter(padding(tensor))

    return smoothed_tensor


def get_short_class_name(label_i,
                         labels_desc=np.loadtxt('synset_words.txt',
                                                str, delimiter='\t')):
    """Get short ImageNet class name from class index."""
    return ' '.join(labels_desc[label_i].split(',')[0].split()[1:])


def tv_norm(tensor, beta=2.):
    """Compute total variation (TV) norm."""
    assert tensor.size(0) == 1
    img = tensor[0]
    d_y = -img[:, :-1, :] + img[:, 1:, :]
    d_x = torch.transpose(-img[:, :, :-1] + img[:, :, 1:], 1, 2)
    return ((d_x.pow(2) + d_y.pow(2)).pow(beta/2.)).mean()


def create_area_target(mask, area):
    """Create target label for area norm loss."""
    size = mask.numel()
    target = torch.ones(size)
    target[:int(size * (1-area))] = 0
    return target


def area_norm(tensor, target):
    """Compute area norm."""
    sorted_x, _ = tensor.reshape(-1).sort()
    return ((sorted_x - target)**2).mean()


def save(mask, orig_img, title=None):
    """Save heatmap image of mask overlaid on the original image."""
    orig_w, orig_h = orig_img.size
    mask_np = mask.cpu().data.squeeze().numpy()

    _, ax = plt.subplots(1, 1)
    ax.imshow(orig_img)
    ax.imshow(resize(mask_np, (orig_h, orig_w)), cmap='jet', alpha=0.5,
              vmin=0.0, vmax=1.0)
    if title is not None:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig('heatmap.png')
    plt.close()


def plot_curve(mask, area_target, suffix=''):
    """Plot curve of sorted mask values and target mask area."""
    if not os.path.exists('area_curves'):
        os.makedirs('area_curves')
    _, ax = plt.subplots(1, 1)
    z = mask.cpu().reshape(-1).sort()[0].data.numpy()
    ax.plot(range(mask.numel()), z)
    ax.plot(range(mask.numel()), area_target.cpu().data.numpy())
    plt.savefig('area_curves/sorted_mask_%s.png' % suffix)
    plt.close()


def set_gpu(gpu=None, framework='pytorch'):
    """Set visible gpu(s). This function should be called once at beginning.

    Args:
        gpu (NoneType, int, or list of ints): the gpu(s) (zero-indexed) to use;
            None if no gpus should be used.
        framework (str): deep learning framework being used; the following
            frameworks are supported: 'tensorflow', 'pytorch'.
    Return:
        bool: True if using at least 1 gpu; otherwise False.
    """
    # Check type of gpu.
    if isinstance(gpu, list):
        if gpu:
            for gpu_i in gpu:
                if not isinstance(gpu_i, int):
                    raise ValueError('gpu should be of type NoneType, int, or '
                                     'list of ints. Instead, gpu[%d] is of '
                                     'type %s.' % type(gpu_i))
    elif isinstance(gpu, int):
        pass
    elif gpu is None:
        pass
    else:
        raise ValueError('gpu should be of type NoneType, int, or list of '
                         'ints. Instead, gpu is of type %s.' % type(gpu))

    # Set if gpu usage (i.e., cuda) is enabled.
    if gpu is None:
        cuda = False
    elif isinstance(gpu, list) and not gpu:
        cuda = False
    else:
        cuda = True

    # Set CUDA_VISIBLE_DEVICES environmental variable.
    gpu_params = ''
    if cuda:
        if isinstance(gpu, list):
            gpu_params = str(gpu).strip('[').strip(']')
        else:
            gpu_params = str(gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_params

    # Check type of framework.
    if framework == 'tensorflow':
        from tensorflow.python.client import device_lib
        num_visible_gpus = len([x.name for x in device_lib.list_local_devices()
                                if x.device_type == 'GPU'])
    elif framework == 'pytorch':
        num_visible_gpus = torch.cuda.device_count()
    else:
        if not isinstance(framework, str):
            raise TypeError('framework should be of type str; instead, '
                            'its type is %s.' % type(framework))
        raise ValueError('framework should be "tensorflow" or "pytorch"; '
                         'instead, framework = %s' % framework)

    # Check number of visible gpus.
    if isinstance(gpu, list):
        if num_visible_gpus != len(gpu):
            raise ValueError('The following %d gpu(s) should be visible: %s; '
                             'instead, %d gpu(s) are visible.'
                             % (len(gpu), str(gpu), num_visible_gpus))
    elif gpu is None:
        if num_visible_gpus != 0:
            raise ValueError('0 gpus should be visible; instead, %d gpu(s) '
                             'are visible.' % num_visible_gpus)
    else:
        if num_visible_gpus != 1:
            raise ValueError('1 gpu should be visible; instead %d gpu(s) '
                             'are visible.' % num_visible_gpus)
        assert num_visible_gpus == 1

    print("%d GPU(s) being used at the following index(es): %s" % (
        num_visible_gpus, gpu_params))
    return cuda
