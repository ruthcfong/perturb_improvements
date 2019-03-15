"""
explain.py

Script to learn explanatory perturbation mask.
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
import torchvision.utils as vutils
import numpy as np
from PIL import Image
import visdom

from utils import (get_device, set_gpu, blur_input_tensor, create_area_target,
                   get_short_class_name, tv_norm, area_norm, plot_curve, save,
                   hook_get_acts, get_first_module_name, get_pytorch_module,
                   replace_module)


class PerturbationLayer(nn.Module):
    def __init__(self,
                 module,
                 perturb_after=True,
                 perturb_batchwise=True,
                 interpolate=True):
        """
        Args:
            module: torch.nn.Module, module to wrap around.
            perturb_after: Boolean, if True, perturb after module module;
                otherwise, perturb before applying module.
            interpolate_batchwise: Boolean, if True, interpolate activations
                batch-wise.
        """
        super(PerturbationLayer, self).__init__()

        # Verify inputs.
        assert isinstance(module, nn.Module)
        assert isinstance(perturb_after, bool)
        assert isinstance(perturb_batchwise, bool)
        assert isinstance(interpolate, bool)

        self.module = module
        self.perturb_after = perturb_after
        self.mask = None
        self.perturbed_x = None
        self.perturb_batchwise = perturb_batchwise
        self.interpolate = interpolate
        if self.interpolate:
            assert self.perturb_batchwise


    def perturb(self, x, mask):
        """Either directly perturb or linearly interpolate activations
           batch-wise using the mask.


        Args:
            x: torch.Tensor, 4D tensor.
            mask: torch.Tensor, 4D tensor.

        Returns:
            x_interpolated: torch.Tensor, 4D tensor with batch size = 1.
        """
        assert isinstance(mask, torch.Tensor)
        assert mask.shape[0] == 1

        if not self.interpolate:

            if self.perturb_batchwise:
                assert(x.shape[0] == 2)
                expanded_mask = self.mask.expand(1, *x.shape[1:])
                return (x[0].unsqueeze(0) * (1 - expanded_mask)
                        + x[1].unsqueeze(0) * expanded_mask)

            expanded_mask = self.mask.expand(*x.shape)
            return x * expanded_mask

        # Verify activation shape.
        assert len(x.shape) == 4
        assert x.shape[0] > 1

        num_perturbations = x.shape[0]

        # Scale mask to be continuous indices into the perturbed activations.
        mask_idx = mask * (num_perturbations - 1)

        # Get the floor of the mask indices.
        mask_idx_f = torch.floor(mask_idx)

        # Get teh difference between the floor and the continuous indices.
        mask_idx_diff = mask_idx - mask_idx_f

        # Cast floor indices to make them discrete.
        mask_idx_f = mask_idx_f.long()

        # Prepare for linear interpolation.
        mask_idx_diff = torch.cat((1 - mask_idx_diff, mask_idx_diff), dim=0)

        # Get ceiling indices.
        mask_idx_c = torch.clamp(mask_idx_f + 1, max=num_perturbations-1)

        # Concatenate floor and ceiling indices.
        mask_discrete_idx = torch.cat((mask_idx_f, mask_idx_c), dim=0)

        # Get activations at discrete indices
        mask_discrete_idx = mask_discrete_idx.expand(2, *x.shape[1:])
        x_discrete = torch.gather(x, 0, mask_discrete_idx)
        mask_idx_diff = mask_idx_diff.expand(*x_discrete.shape)

        # Linearly interpolate activations.
        x_interpolated = torch.sum(x_discrete * mask_idx_diff, dim=0, keepdim=True)

        return x_interpolated

    def perturb_dual(self, x):
        assert isinstance(self.mask, torch.Tensor)
        mask_bs = self.mask.shape[0]
        assert mask_bs == 1 or mask_bs == 2
        if mask_bs == 2:
            x0 = self.perturb(x, self.mask[0].unsqueeze(0))
            x1 = self.perturb(x, self.mask[1].unsqueeze(0))
            x = torch.cat((x0, x1), dim=0)
        else:
            x = self.perturb(x, self.mask)
        assert x.shape[0] == mask_bs
        self.perturbed_x = x
        return x

    def forward(self, x):
        # Apply perturbation either before or after applying the module.
        if not self.perturb_after:
            x = self.perturb_dual(x)

        x = self.module(x)

        if self.perturb_after:
            x = self.perturb_dual(x)

        # Verify final shape.
        assert(len(x.shape) == 4)

        return x


def perturb_explanation(image='images/flute.jpg',
                        arch='alexnet',
                        layer='input',
                        perturbation='blur',
                        perturbation_dimension='spatial',
                        interpolate=False,
                        num_perturbations=-1,
                        input_size=227,
                        learning_rate=1e-1,
                        epochs=300,
                        l1_lambda=1e-2,
                        tv_lambda=1e-4,
                        tv_beta=3.,
                        blur_size=11,
                        blur_sigma=10.,
                        mask_size=None,
                        noise_std=0.,
                        area_lambda=0.,
                        area=1.,
                        ease_in_area=False,
                        ease_rate=1.,
                        area_delay=0,
                        use_softmax=True,
                        class_loss_type='deletion',
                        print_iter=25,
                        debug=False):
    """Generate a perturbation mask for a given image and CNN.

    Args:
        image: String, path to an image file.
        arch: String, name of PyTorch CNN architecture.
        layer: String, name of module in architecture.
        perturbation: String, either "blur" or "intensity".
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
    global vis
    if debug:
        vis = visdom.Visdom(env='perturb_improvements')
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

    if interpolate:
        assert num_perturbations > 1
        inputs = []
        alphas = np.linspace(0, 1, num_perturbations)
        for alpha in alphas:
            if perturbation == 'blur':
                if alpha == 1.0:
                    inputs.append(img)
                    continue
                perturbed_img = blur_input_tensor(img,
                                                  kernel_size=blur_size,
                                                  sigma=(1. - alpha)*blur_sigma)
            elif perturbation == 'intensity':
                perturbed_img = alpha * img
            inputs.append(perturbed_img)
        input = torch.cat(inputs, dim=0)
    else:
        if perturbation == 'blur':
            blurred_img = blur_input_tensor(
                img, kernel_size=blur_size, sigma=blur_sigma)
            input = torch.cat((blurred_img, img), 0)
        elif perturbation == 'intensity':
            input = img
        else:
            assert False

    img = img.to(device)
    input = input.to(device)

    target = (model(img))
    sm_target = nn.Softmax(1)(target)
    category = np.argmax(target.cpu().data.numpy())
    print("Category with highest score: %s (logits: %.4f, softmax: %.4f)"
          % (get_short_class_name(category),
             target.cpu().data.numpy()[0][category],
             sm_target.cpu().data.numpy()[0][category]))

    upsample = torch.nn.Upsample(size=(input_size, input_size), mode='bilinear').to(device)
    should_upsample = False
    if perturbation_dimension == 'spatial':
        # TODO(ruthfong): Handle this correctly.
        mask = torch.ones((1, 1, mask_size, mask_size),
                          requires_grad=True,
                          device=device)
        if mask_size is not None:
            should_upsample = True
    elif perturbation_dimension == 'channel':
        assert layer != 'input'
        acts = hook_get_acts(model, [layer], img)[0]
        assert len(acts.shape) == 4
        mask = torch.ones((1, acts.shape[1], 1, 1),
                          requires_grad=True,
                          device=device)
    else:
        assert(False)
    mask.data.fill_(0.5)

    area_target = create_area_target(mask, area).to(device)

    if layer == 'input':
        perturb_after = False
        layer = get_first_module_name(model)
    else:
        perturb_after = True

    if interpolate:
        perturb_batchwise = True
    else:
        perturb_batchwise = perturbation == 'blur'

    layer_module = get_pytorch_module(model, layer)
    print(f'Adding perturbation layer to {layer}.')
    perturb_layer = PerturbationLayer(module=layer_module,
                                      perturb_after=perturb_after,
                                      perturb_batchwise=perturb_batchwise,
                                      interpolate=interpolate)
    perturb_layer = perturb_layer.to(device)
    model = replace_module(model, layer.split('.'), perturb_layer)

    if use_softmax:
        softmax = torch.nn.Softmax(dim=1).to(device)
        model = nn.Sequential(model, softmax)

    optimizer = torch.optim.Adam([mask], lr=learning_rate)
    print("Optimizing...")

    class_losses = []
    area_losses = []
    area_lambdas = []
    for i in range(epochs):
        # TODO(ruthfong): Add transformations (i.e., rotation, jitter).

        # Upsample mask if necessary.
        if should_upsample:
            upsampled_mask = upsample(mask)
        else:
            upsampled_mask = mask

        # Set up dual mask if necessary.
        if class_loss_type == 'dual':
            dual_mask = torch.cat((upsampled_mask, 1-upsampled_mask), dim=0)
        else:
            dual_mask = upsampled_mask

        # Set perturbation mask.
        perturb_layer.mask = dual_mask

        noise = noise_std * torch.randn((1, 3, input_size, input_size),
                                        device=device)
        noisy_input = input + noise

        outputs = model(noisy_input)
        l1_loss = l1_lambda*torch.mean(torch.abs(1-mask))
        if perturbation_dimension == 'spatial':
            tv_loss = tv_lambda*tv_norm(mask, tv_beta)
        else:
            tv_loss = torch.tensor(0, device=device)
        if i < area_delay:
            curr_area_lambda = 0
        else:
            if ease_in_area:
                curr_area_lambda = area_lambda * min(
                    ease_rate*(i-area_delay) / float(epochs-area_delay), 1.)
            else:
                curr_area_lambda = area_lambda
        area_loss = curr_area_lambda*area_norm(mask, area_target)
        if class_loss_type == 'deletion':
            class_loss = outputs[0, category]
        elif class_loss_type == 'preservation':
            class_loss = -1 * outputs[0, category]
        elif class_loss_type == 'dual':
            assert outputs.shape[0] == 2
            class_loss = -1 * (outputs[0, category] - outputs[1, category])
        else:
            assert False
        tot_loss = l1_loss + tv_loss + class_loss + area_loss

        optimizer.zero_grad()
        tot_loss.backward()
        optimizer.step()

        mask.data.clamp_(0, 1)

        class_losses.append(class_loss.cpu().data.numpy())
        area_losses.append(area_loss.cpu().data.numpy())
        area_lambdas.append(curr_area_lambda)

        if i % print_iter == 0:
            print('Epoch %d\tL1 Loss %f\tTV Loss %f\tArea Loss %f\t'
                  'Class Loss %f\tTot Loss %f'
                  % (i, l1_loss.item(), tv_loss.item(), area_loss.item(),
                     class_loss.item(), tot_loss.item()))
            plot_curve(mask, area_target, suffix='_%d' % i)
            if debug:
                vis.image(vutils.make_grid(input, normalize=True), win=0)
                if perturbation_dimension == 'spatial':
                    vis.image(vutils.make_grid(mask), win=1)
                vis.line(np.sort(mask.cpu().data.numpy().flatten()), win=2)
                errs = np.vstack((class_losses, area_losses)).T
                vis.line(errs, win=3)
                vis.line(np.array(area_lambdas), win=4)
                if perturbation_dimension == 'spatial':
                    vis.image(vutils.make_grid(upsample(mask.grad), normalize=True), win=5)
                    vis.image(vutils.make_grid(upsampled_mask, normalize=True), win=6)
                elif perturbation_dimension == 'channel':
                    perturbed_x = perturb_layer.perturbed_x
                    if perturbed_x.shape[0] == 1:
                        y = upsample(torch.sum(perturbed_x, 1, keepdim=True)).squeeze()
                        vis.heatmap(y.flip(0), win=5)
                    elif perturbed_x.shape[0] == 2:
                        y0 = upsample(torch.sum(perturbed_x[0].unsqueeze(0), 1, keepdim=True)).squeeze()
                        y1 = upsample(torch.sum(perturbed_x[1].unsqueeze(0), 1, keepdim=True)).squeeze()
                        vis.heatmap(y0.flip(0), win=5)
                        vis.heatmap(y1.flip(0), win=6)
                    else:
                        assert(False)

    # TODO(ruthfong/mandelapatrik): Save mask / visualization.
    return mask.data.cpu().numpy()


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
        parser.add_argument('--mask_size', type=int, default=None,
                            help='Learned mask size')
        parser.add_argument('--noise', type=float, default=0,
                            help='Amount of random Gaussian noise to add')
        parser.add_argument('--ease_in_area', type='bool', default=False,
                            help='Should area loss be linearly eased in.')
        parser.add_argument('--ease_rate', type=float, default=2.,
                            help='Rate at which area loss should be eased in.')
        parser.add_argument('--gpu', type=int, nargs='*', default=None)
        parser.add_argument('--layer', type=str, default='input')
        parser.add_argument('--perturbation', choices=['blur', 'intensity'],
                            default='blur')
        parser.add_argument('--dimension', choices=['spatial', 'channel'],
                            default='spatial')
        parser.add_argument('--num_perturbations', type=int, default=10)
        parser.add_argument('--use_softmax', type='bool', default=True)
        parser.add_argument('--interpolate', type='bool', default=False)
        parser.add_argument('--loss',
                            choices=['deletion', 'preservation', 'dual'],
                            default='deletion')
        parser.add_argument('--area_delay', type=int, default=0)
        parser.add_argument('--debug', type='bool', default=False)


        args = parser.parse_args()
        set_gpu(args.gpu)
        perturb_explanation(image=args.image,
                            arch=args.architecture,
                            layer=args.layer,
                            perturbation=args.perturbation,
                            perturbation_dimension=args.dimension,
                            num_perturbations=args.num_perturbations,
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
                            ease_rate=args.ease_rate,
                            area_delay=args.area_delay,
                            interpolate=args.interpolate,
                            use_softmax=args.use_softmax,
                            class_loss_type=args.loss,
                            debug=args.debug)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
