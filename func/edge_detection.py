import os
import torch
import skimage

from tqdm import tqdm
from torch.nn import functional as F
from matplotlib import image

def read_data(path):
    image = skimage.io.imread(path)
    image = skimage.color.rgb2gray(image[...,:3])
    return torch.Tensor(image)

def binarize(input_values, threshold):
    return torch.where(input_values>threshold, 1.0, 0.0)

def get_kernel_from_rule(rule):
    kernel_matrix = torch.zeros(5, 5)
    binary_rule = format(rule, "b")
    binary_rule = "0"*(25-len(binary_rule)) + binary_rule

    direction, indexer, step = ("right", 0, 1)
    cur_row, cur_col = 2, 2
    for binary in binary_rule[::-1]:
        if int(binary) == 1 and cur_row <= 4 and cur_row <= 4:
            kernel_matrix[cur_row, cur_col] = 1
        indexer += 1
        
        if direction == "right":
            cur_col += 1
            direction, indexer = ("down", 0) if indexer == step else ("right", indexer)
        elif direction == "down":
            cur_row += 1
            direction, indexer, step = ("left", 0, step+1) if indexer == step else ("down", indexer, step)
        elif direction == "left":
            cur_col -= 1
            direction, indexer = ("up", 0) if indexer == step else ("left", indexer)
        elif direction == "up":
            cur_row -= 1
            direction, indexer, step = ("right", 0, step+1) if indexer == step else ("up", indexer, step)
    return kernel_matrix

def run_ca(path, binary_threshold, rule, epoch, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir+"/binarized", exist_ok=True)
    os.makedirs(out_dir+f"/ca/rule_{rule}", exist_ok=True)
    _, file_name = os.path.split(path)

    input_matrix = read_data(path)
    input_matrix = binarize(input_matrix, binary_threshold)
    binarized_data = input_matrix.clone().detach()

    kernel_matrix = get_kernel_from_rule(rule)
    kernel_size = kernel_matrix.shape

    input_matrix = input_matrix.to(device)
    kernel_matrix = kernel_matrix.to(device)
    with tqdm(total=epoch) as pbar:
        for i in range(epoch):
            pbar.set_description(f"running ca with rule {rule} at epoch {i+1}")
            padded_input_matrix = F.pad(input_matrix, (2,2,2,2), value=0.0)

            unfolded_input = F.unfold(padded_input_matrix.unsqueeze(0).unsqueeze(0), kernel_size).squeeze(0).T
            flattened_kernel = kernel_matrix.flatten()

            convolved = unfolded_input@flattened_kernel
            input_matrix = convolved.reshape(input_matrix.shape[0], input_matrix.shape[1])
            input_matrix = torch.where(input_matrix%2==0, 0.0, 1.0)

            pbar.update(1)

    binarized_path = f'{out_dir}/binarized/{file_name}'
    ca_path = f'{out_dir}/ca/rule_{rule}/{file_name}'

    image.imsave(binarized_path, binarized_data.cpu().detach().numpy())
    image.imsave(ca_path, input_matrix.cpu().detach().numpy())


if __name__ == "__main__":
    path = "./data/testing.png"
    binary_threshold = 0.3
    rule = 1025
    epoch = 1
    device = torch.device("cuda")
    out_dir = "./data/testing"
    run_ca(path, binary_threshold, rule, epoch, device, out_dir)