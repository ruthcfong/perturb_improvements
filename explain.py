import os
import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


def get_short_class_name(label_i, labels_desc = np.loadtxt('synset_words.txt',
    str, delimiter='\t')):
    return ' '.join(labels_desc[label_i].split(',')[0].split()[1:])


def tv_norm(x, beta=2.):
    assert(x.size(0) == 1)
    img = x[0]
    dy = -img[:,:-1,:] + img[:,1:,:]
    dx = torch.transpose(-img[:,:,:-1] + img[:,:,1:], 1, 2)
    return ((dx.pow(2) + dy.pow(2)).pow(beta/2.)).mean()


def create_area_target(mask, area):
  size = mask.numel()
  target = torch.ones(size)
  target[:int(size * (1-area))] = 0
  return target


def area_norm(x, target):
  sorted_x, _ = x.reshape(-1).sort()
  return ((sorted_x - target)**2).mean()


def preprocess_image(img):
  means=[0.485, 0.456, 0.406]
  stds=[0.229, 0.224, 0.225]

  preprocessed_img = img.copy()[: , :, ::-1]
  for i in range(3):
    preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
    preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
  preprocessed_img = \
      np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))

  if use_cuda:
      preprocessed_img_tensor = torch.from_numpy(preprocessed_img).cuda()
  else:
      preprocessed_img_tensor = torch.from_numpy(preprocessed_img)

  preprocessed_img_tensor.unsqueeze_(0)
  return Variable(preprocessed_img_tensor, requires_grad = False)


def save(mask, img, blurred):
    mask = mask.cpu().data.numpy()[0]
    mask = np.transpose(mask, (1, 2, 0))

    mask = (mask - np.min(mask)) / np.max(mask)
    mask = 1 - mask
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    
    heatmap = np.float32(heatmap) / 255
    cam = 1.0*heatmap + np.float32(img)/255
    cam = cam / np.max(cam)

    img = np.float32(img) / 255
    perturbated = np.multiply(1 - mask, img) + np.multiply(mask, blurred)   

    cv2.imwrite("perturbated.png", np.uint8(255*perturbated))
    cv2.imwrite("heatmap.png", np.uint8(255*heatmap))
    cv2.imwrite("mask.png", np.uint8(255*mask))
    cv2.imwrite("cam.png", np.uint8(255*cam))


def numpy_to_torch(img, requires_grad=True):
  if len(img.shape) < 3:
    output = np.float32([img])
  else:
    output = np.transpose(img, (2, 0, 1))

  output = torch.from_numpy(output)
  if use_cuda:
    output = output.cuda()

  output.unsqueeze_(0)
  v = Variable(output, requires_grad = requires_grad)
  return v


def plot_curve(mask, area_target, suffix=''):
  if not os.path.exists('area_curves'):
    os.makedirs('area_curves')
  f, ax = plt.subplots(1, 1)
  z = mask.cpu().reshape(-1).sort()[0].data.numpy()
  ax.plot(range(mask.numel()), z)
  ax.plot(range(mask.numel()), area_target.cpu().data.numpy())
  plt.savefig('area_curves/sorted_mask_%s.png' % suffix)
  plt.close()


def perturb_explanation(image='images/flute.jpg', arch='alexnet',
        input_size=227, learning_rate=1e-1, epochs=300,
        l1_lambda=1e-2, tv_lambda=1e-4, tv_beta=3, blur_size=11, blur_sigma=10, 
        mask_size=28, noise_std=0, area_lambda=0, area=1., ease_in_area=False,
        ease_rate=1., print_iter=25):
  # Load pre-trained model.
  model = models.__dict__[arch](pretrained=True)
  if use_cuda:
    model.cuda()

  original_img = cv2.imread(image, 1)
  original_img = cv2.resize(original_img, (input_size, input_size))
  img = np.float32(original_img) / 255
  blurred_img_numpy = cv2.GaussianBlur(img, (blur_size, blur_size), blur_sigma)
  mask_init = 0.5*np.ones((mask_size, mask_size), dtype = np.float32)

  # Convert to torch variables
  img = preprocess_image(img)
  blurred_img = preprocess_image(blurred_img_numpy)
  mask = numpy_to_torch(mask_init)
  area_target = create_area_target(mask, area)
  if use_cuda:
    area_target = area_target.cuda()

  if use_cuda:
    upsample = torch.nn.Upsample(size=(input_size, input_size), mode='bilinear').cuda()
    softmax = torch.nn.Softmax(dim=1).cuda()
  else:
    upsample = torch.nn.Upsample(size=(input_size, input_size), mode='bilinear')
    softmax = torch.nn.Softmax(dim=1)

  optimizer = torch.optim.Adam([mask], lr=learning_rate)

  target = softmax(model(img))
  category = np.argmax(target.cpu().data.numpy())
  print("Category with highest probability: %s (%.4f)" % (get_short_class_name(category),
          target.cpu().data.numpy()[0][category]))
  print("Optimizing...")

  for i in range(epochs):
    upsampled_mask = upsample(mask)
    upsampled_mask = upsampled_mask.expand(1, 3, upsampled_mask.size(2),
                                           upsampled_mask.size(3))

    # Use the mask to perturbated the input image.
    perturbated_input = img.mul(upsampled_mask) + blurred_img.mul(1-upsampled_mask)

    noise = np.zeros((input_size, input_size, 3), dtype = np.float32)
    if noise_std != 0:
        noise = noise + cv2.randn(noise, 0, noise_std)
    noise = numpy_to_torch(noise)
    perturbated_input = perturbated_input + noise

    outputs = softmax(model(perturbated_input))
    l1_loss = l1_lambda*torch.mean(torch.abs(1-mask))
    tv_loss = tv_lambda*tv_norm(mask, tv_beta)
    if ease_in_area:
      curr_area_lambda = area_lambda * min(ease_rate*i / float(epochs), 1.)
    else:
      curr_area_lambda
    area_loss = curr_area_lambda*area_norm(mask, area_target)
    class_loss = outputs[0, category]
    tot_loss = l1_loss + tv_loss + class_loss + area_loss

    optimizer.zero_grad()
    tot_loss.backward()
    optimizer.step()

    mask.data.clamp_(0, 1)

    if i % print_iter == 0:
        print('Epoch %d\tL1 Loss %f\tTV Loss %f\tArea Loss %f\tClass Loss %f\tTot Loss %f'
              % (i, l1_loss.item(), tv_loss.item(), area_loss.item(),
                 class_loss.item(), tot_loss.item()))
        plot_curve(mask, area_target, suffix='_%d' % i)

  upsampled_mask = upsample(mask)
  save(upsampled_mask, original_img, blurred_img_numpy)
  return np.squeeze(upsampled_mask.data.cpu().numpy())

if __name__ == '__main__':
  import argparse
  import sys
  import traceback
  try:
    parser = argparse.ArgumentParser(description='Learn perturbation mask')

    def str2bool(v):
      v = v.lower()
      if v in ('yes', 'true', 't', '1'):
        return True
      elif v in ('no', 'false', 'f', '0'):
        return False
      raise ValueError('Boolean argument needs to be true or false. '
                       'Instead, it is %s.' % v)
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--image', default='images/flute.jpg',
            help='path of input image')
    parser.add_argument('--architecture', default='alexnet',
            help='name of CNN architecture (choose from PyTorch pretrained networks')
    parser.add_argument('--input_size', type=int, default=227,
            help='CNN image input size')
    parser.add_argument('--learning_rate', type=float, default=1e-1,
            help='learning rate (for Adam optimization)')
    parser.add_argument('--epochs', type=int, default=300,
            help='number of iterations to run optimization for')
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

    args = parser.parse_args()
    perturb_explanation(image=args.image, arch=args.architecture,
            input_size=args.input_size, learning_rate=args.learning_rate,
            epochs=args.epochs, l1_lambda=args.l1_lambda,
            tv_lambda=args.tv_lambda, tv_beta=args.tv_beta,
            mask_size=args.mask_size, noise_std=args.noise,
            area_lambda=args.area_lambda, area=args.area,
            ease_in_area=args.ease_in_area, ease_rate=args.ease_rate)
  except:
      traceback.print_exc(file=sys.stdout)
      sys.exit(1)
