import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms
from model import Discriminator, Generator, initialize_weights
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NUM_CLASSES = 10
GEN_EMBEDDING = 100
Z_DIM = 100
NUM_EPOCHS = 1  # 3  # 100
FEATURES_CRITIC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10


# load the model
saved_model_path = 'cond_wgan_gp_epoch_0.pth.tar'
gen_model = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES,
                      IMAGE_SIZE, GEN_EMBEDDING).to(device)
initialize_weights(gen_model)
gen_model.load_state_dict(torch.load(saved_model_path)['gen_model_state_dict'])
gen_model.eval()

noise = torch.randn(32, Z_DIM, 1, 1).to(device)


# convert text to number to tensor
Fashion_MNIST_classes = {
    "t_shirt": 0,
    "top": 0,
    "trouser": 1,
    "pullover": 2,
    "dress": 3,
    "coat": 4,
    "sandal": 5,
    "sandals": 5,
    "shirt": 6,
    "sneaker": 7,
    "sneakers": 7,
    "bag": 8,
    "ankle_boot": 9,
}


def user_input_to_tensor(user_input):
    try:
        class_id = Fashion_MNIST_classes[str(user_input)]
        class_id = [class_id for i in range(32)]
        return torch.tensor(class_id)
        # return (class_id)
    except (KeyError, ValueError, TypeError, NameError) as err:
        print('User input invalid! \nOr...\n', err)
    # return class_id

# displays and save images generated from tensors


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.detach().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.savefig('./generated_images/' + title + '.png')


# predict and generate image in form of tensor
def get_prediction(user_input):
    input = user_input_to_tensor(user_input)
    noise = torch.randn(32, Z_DIM, 1, 1).to(device)
    fake = gen_model(noise, input)
    img_grid_fake = torchvision.utils.make_grid(fake)
    return img_grid_fake

    # img = transforms.ToPILImage()(img_grid_fake)
    # display(img)
    # return img

# convert tensor result to image
