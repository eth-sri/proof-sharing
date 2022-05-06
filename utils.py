import os
import pickle
import torch
from torchvision import datasets, transforms
import numpy as np
from scipy.io import loadmat
import itertools
from time import time
from tqdm import tqdm
import re
import logging
import logging.handlers
import datetime

from relaxations import Zonotope_Net, Box_Net, Zonotope  # noqa: F402
from networks import Network, Normalization  # noqa: F402

PATH_EXAMPLES = 'examples/'
DEVICE = 'cpu'
INPUT_SIZE = (1, 28, 28)
DOWNLOAD_DATA = True


def runRepl(arg, repl):
    for a in repl:
        arg = arg.replace(a+"=", "'"+a+"':")
    return eval("{"+arg+"}")


def parseVec(lines):
    return torch.Tensor(eval(lines.readline()[:-1]))


def extract_mean(text):

    m = re.search('mean=\[(.+?)\]', text)  # noqa: W605

    if m:
        means = m.group(1)
    mean_str = means.split(',')
    num_means = len(mean_str)
    mean_array = np.zeros(num_means)
    for i in range(num_means):
        mean_array[i] = np.float64(mean_str[i])
    return mean_array


def extract_std(text):

    m = re.search('std=\[(.+?)\]', text)  # noqa: W605
    if m:
        stds = m.group(1)
    std_str = stds.split(',')
    num_std = len(std_str)
    std_array = np.zeros(num_std)
    for i in range(num_std):
        std_array[i] = np.float64(std_str[i])
    return std_array


def save_net_self_trained(net, path):
    layer_list = []
    for layer in net.layers:
        current_layer = {}
        if isinstance(layer, torch.nn.Linear):
            current_layer['type'] = 'Linear'
            current_layer['parameters'] = [
                layer.in_features, layer.out_features]
            current_layer['weights'] = (
                layer.weight.data, layer.bias.data)

        elif isinstance(layer, torch.nn.Conv2d):
            current_layer['type'] = 'Conv'
            current_layer['parameters'] = [
                layer.in_channels, layer.out_channels, layer.kernel_size, layer.stride, layer.padding]
            current_layer['weights'] = (
                layer.weight.data, layer.bias.data)
        elif isinstance(layer, Normalization):
            current_layer['type'] = 'Normalization'
            current_layer['weights'] = (
                layer.mean.data, layer.sigma.data)
        elif isinstance(layer, torch.nn.ReLU):
            current_layer['type'] = 'ReLU'
        elif isinstance(layer, torch.nn.MaxPool2d):
            current_layer['type'] = 'MaxPool'
            current_layer['parameters'] = [layer.kernel_size]
        elif isinstance(layer, torch.nn.AvgPool2d):
            current_layer['type'] = 'AvgPool'
            current_layer['parameters'] = [layer.kernel_size]
        elif isinstance(layer, torch.nn.Flatten):
            current_layer['type'] = 'Flatten'
        else:
            print('Unknown layer:', layer)
        layer_list.append(current_layer)

        # print(current_layer)

    pickle.dump(layer_list, open(path, 'wb'))
    print('Net saved to {}'.format(path))


def load_net_self_trained(path):
    layers = pickle.load(open(path, 'rb'))

    conv_layers = []
    fc_layers = []
    weights = []
    biases = []
    use_normalization = False
    mean = 0.1307
    sigma = 0.3081
    nonlinearity_after_conv = 'relu'

    input_size = get_input_size_from_file(path)

    for layer in layers:
        if layer['type'] == 'Linear':
            weights.append(layer['weights'][0])
            biases.append(layer['weights'][1])
            fc_layers.append(layer['parameters'][1])

        elif layer['type'] == 'Conv':
            weights.append(layer['weights'][0])
            biases.append(layer['weights'][1])
            params = layer['parameters'][1:]
            param_list = []
            for param in params:
                if isinstance(param, tuple):
                    param = param[0]
                param_list.append(param)

            conv_layers.append(tuple(param_list))

        elif layer['type'] == 'Normalization':
            use_normalization = True
            mean = layer['weights'][0]
            sigma = layer['weights'][1]

        elif layer['type'] == 'ReLU':
            pass
        elif layer['type'] == 'MaxPool':
            nonlinearity_after_conv = 'max'
            print(conv_layers[-1], (layer.kernel_size,))
            conv_layers[-1] = conv_layers[-1] + (layer.kernel_size,)
        elif layer['type'] == 'AvgPool':
            nonlinearity_after_conv = 'average'
            conv_layers[-1] = conv_layers[-1] + (layer.kernel_size,)
        elif layer['type'] == 'Flatten':
            pass
        else:
            print('Unknown layer:', layer)

    net = Network(DEVICE, input_size, conv_layers, fc_layers,
                  10, use_normalization, nonlinearity_after_conv, mean=mean, sigma=sigma)

    for layer in net.layers:
        if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d)):
            layer.weight = torch.nn.Parameter(weights.pop(0), False)
            layer.bias = torch.nn.Parameter(biases.pop(0), False)
    net.update_bias_free_layers()

    return net


def load_net_from_eran_examples(file_name):

    conv_layers = []
    fc_layers = []
    weights = []
    biases = []
    use_normalization = False
    mean = 0.1307
    sigma = 0.3081

    input_size = get_input_size_from_file(file_name)

    lines = open(file_name, 'r')
    while True:
        curr_line = lines.readline()[:-1]
        if 'Normalize' in curr_line:
            use_normalization = True

            mean = extract_mean(curr_line)
            sigma = extract_std(curr_line)

        elif curr_line in ['ReLU', 'Affine']:
            W = parseVec(lines)
            b = parseVec(lines)

            weights.append(W)
            biases.append(b)
            fc_layers.append(b.numel())

        elif curr_line == 'Conv2D':
            line = lines.readline()

            start = 0
            if('ReLU' in line):
                start = 5
            elif('Sigmoid' in line):
                start = 8
            elif('Tanh' in line):
                start = 5
            elif('Affine' in line):
                start = 7
            if 'padding' in line:
                args = runRepl(
                    line[start:-1], ['filters', 'input_shape', 'kernel_size', 'stride', 'padding'])
            else:
                args = runRepl(
                    line[start:-1], ['filters', 'input_shape', 'kernel_size'])

            W = parseVec(lines).permute(3, 2, 0, 1)
            b = parseVec(lines)

            weights.append(W)
            biases.append(b)

            conv_layers.append(
                (args['filters'], args['kernel_size'][0], args['stride'][0], args['padding']))

        elif curr_line == '':
            break
        else:
            raise Exception('Unsupported Operation: ')
            pass
    net = Network(DEVICE, input_size, conv_layers, fc_layers,
                  10, use_normalization, mean=mean, sigma=sigma)

    for layer in net.layers:
        if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d)):
            layer.weight = torch.nn.Parameter(weights.pop(0), False)
            layer.bias = torch.nn.Parameter(biases.pop(0), False)
    net.update_bias_free_layers()

    return net


def load_net_from_bridging_the_gap(file_name):
    state_dict_load = torch.load(file_name,
                                 map_location=torch.device(DEVICE))

    keys = list(state_dict_load.keys())
    conv_layers = []
    fc_layers = []
    weights = []
    biases = []

    input_size = get_input_size_from_file(file_name)

    for key in keys:
        if 'conv.weight' in key:
            layer_idx = int(key.split('.')[2])
            shape = state_dict_load[key].shape

            filters = shape[0]
            kernel_size = shape[2]
            # if layer_idx == 1:
            #     stride = 2
            #     padding = 2
            # elif layer_idx == 3:
            #     stride == 2
            #     padding = 1
            if layer_idx == 1:
                stride = 1
                padding = 1
            elif layer_idx in [3, 5]:
                stride = 2
                padding = 1
            else:
                raise NotImplementedError

            conv_layers.append((filters, kernel_size, stride, padding))
            weights.append(state_dict_load[key])

        elif 'linear.weight' in key:
            shape = state_dict_load[key].shape
            fc_layers.append(shape[0])
            weights.append(state_dict_load[key])

        elif 'bias' in key:
            biases.append(state_dict_load[key])

    net = Network(DEVICE, input_size, conv_layers, fc_layers,
                  10)

    for layer in net.layers:
        if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d)):

            layer.weight = torch.nn.Parameter(weights.pop(0), False)
            layer.bias = torch.nn.Parameter(biases.pop(0), False)
    net.update_bias_free_layers()
    return net


def load_net_from_acasxu(file_name):

    use_normalization = True

    matfile = loadmat(file_name)

    weights = list(matfile['W'][0])
    biases = list(matfile['b'][0])

    fc_layers = matfile['layer_sizes'][0][1:]
    input_size = (1, matfile['layer_sizes'][0][0])

    mean = matfile['means_for_scaling'][0][:5]
    sigma = matfile['range_for_scaling'][0][:5]

    net = Network(DEVICE, input_size, [], fc_layers,
                  fc_layers[-1], use_normalization, mean=mean, sigma=sigma)

    for layer in net.layers:
        if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d)):
            layer.weight = torch.nn.Parameter(
                torch.Tensor(weights.pop(0)), False)
            layer.bias = torch.nn.Parameter(
                torch.Tensor(biases.pop(0).T), False)
    net.update_bias_free_layers()
    # mean = torch.Tensor(mean).view(1, 1, 1, 5)
    # print(net(mean))

    return net


def load_net_from_patch_attacks(file_name):

    load_dict = torch.load(file_name, map_location=torch.device(DEVICE))
    state_dict_load = load_dict['state_dict']
    layers = load_dict['model_layers']

    conv_layers = []
    fc_layers = []
    weights = []
    biases = []
    use_normalization = False
    mean = 0.1307
    sigma = 0.3081
    nonlinearity_after_conv = 'relu'

    input_size = get_input_size_from_file(file_name)

    for layer in layers:
        if layer['type'] == 'Linear':
            fc_layers.append(layer['parameters'][1])

        elif layer['type'] == 'Conv':
            params = layer['parameters'][1:]
            param_list = []
            for param in params:
                if isinstance(param, tuple):
                    param = param[0]
                param_list.append(param)

            conv_layers.append(tuple(param_list))
        elif layer['type'] == 'Normalization':
            use_normalization = True
            mean = layer['weights'][0]
            sigma = layer['weights'][1]
        elif layer['type'] == 'ReLU':
            pass
        elif layer['type'] == 'MaxPool':
            nonlinearity_after_conv = 'max'
            print(conv_layers[-1], (layer.kernel_size,))
            conv_layers[-1] = conv_layers[-1] + (layer.kernel_size,)
        elif layer['type'] == 'AvgPool':
            nonlinearity_after_conv = 'average'
            conv_layers[-1] = conv_layers[-1] + (layer.kernel_size,)
        elif layer['type'] == 'Flatten':
            pass
        else:
            print('Unknown layer:', layer)

    for key in state_dict_load.keys():

        if '.weight' in key:
            weights.append(state_dict_load[key])
        elif '.bias' in key:
            biases.append(state_dict_load[key])
        else:
            print('Unknown layer type: {}'.format(key))

    net = Network(DEVICE, input_size, conv_layers, fc_layers,
                  10, use_normalization, nonlinearity_after_conv, mean=mean, sigma=sigma)

    for layer in net.layers:
        if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d)):

            layer.weight = torch.nn.Parameter(weights.pop(0), False)
            layer.bias = torch.nn.Parameter(biases.pop(0), False)

    net.update_bias_free_layers()
    return net


def get_input_size_from_file(file_name):

    if 'mnist' in file_name:
        input_size = (1, 28, 28)
    elif 'cifar' in file_name:
        input_size = (3, 32, 32)
    else:
        print('Unknown Dataset for file {}'.format(file_name))
        raise NotImplementedError

    return input_size


def get_acaxsu_property(property_number):
    # outer list belongs to set of OR-conditions
    # inner list belongs to set of AND-conditions
    # inner most tuples consist of weights for halfspace conditon
    # weights are for the outputs [COC, weak left, weak right, strong left, strong right]

    # todo properties 2,3,4,7,8 need to be LPs

    property_conditions = None
    lower_bound_input = torch.Tensor([0, -3.141593, -3.141593, 100, 0])
    upper_bound_input = torch.Tensor([60760, 3.141593, 3.141593, 1200, 1200])

    if property_number == 1:
        # COC <= 1500 (with norming 3.991125)
        property_conditions = ([-1, 0, 0, 0, 0], -3.991125)

        lower_bound_input[0] = 55947.691
        lower_bound_input[3] = 1145
        upper_bound_input[4] = 60

    elif property_number == 2:
        # COC is not maximal
        property_conditions = ['OR',
                               ([1, -1, 0, 0, 0], 0),
                               ([1, 0, -1, 0, 0], 0),
                               ([1, 0, 0, -1, 0], 0),
                               ([1, 0, 0, 0, -1], 0)]

    elif property_number in [3, 4]:
        # COC is not minimal
        property_conditions = ['OR',
                               ([-1, 1, 0, 0, 0], 0),
                               ([-1, 0, 1, 0, 0], 0),
                               ([-1, 0, 0, 1, 0], 0),
                               ([-1, 0, 0, 0, 1], 0)]
        if property_number == 4:
            lower_bound_input = torch.Tensor([1500, -0.06, 0, 1000, 700])
            upper_bound_input = torch.Tensor(
                [1800, 0.06, 0, 1200, 800])

    elif property_number == 5:
        # strong right is minimal
        property_conditions = ['AND',
                               ([1, 0, 0, 0, -1], 0),
                               ([0, 1, 0, 0, -1], 0),
                               ([0, 0, 1, 0, -1], 0),
                               ([0, 0, 0, 1, -1], 0)]

    elif property_number in [6, 10]:
        # COC is minimal
        property_conditions = ['AND',
                               ([-1, 1, 0, 0, 0], 0),
                               ([-1, 0, 1, 0, 0], 0),
                               ([-1, 0, 0, 1, 0], 0),
                               ([-1, 0, 0, 0, 1], 0)]

    elif property_number == 7:
        # neither strong left nor strong right are the minimal score
        property_conditions = ['AND',
                               ['OR',
                                ([-1, 0, 0, 1, 0], 0),
                                   ([0, -1, 0, 1, 0], 0),
                                   ([0, 0, -1, 1, 0], 0)],
                               ['OR',
                                   ([-1, 0, 0, 0, 1], 0),
                                   ([0, -1, 0, 0, 1], 0),
                                   ([0, 0, -1, 0, 1], 0)]]

    elif property_number == 8:
        # COC is minimal or weak left is minimal
        property_conditions = ['OR',
                               ['AND',
                                ([-1, 0, 1, 0, 0], 0),
                                   ([-1, 0, 0, 1, 0], 0),
                                   ([-1, 0, 0, 0, 1], 0)],
                               ['AND',
                                   ([0, -1, 1, 0, 0], 0),
                                   ([0, -1, 0, 1, 0], 0),
                                   ([0, -1, 0, 0, 1], 0)]]

    elif property_number == 9:
        # strong left is minimal
        property_conditions = ['AND',
                               ([1, 0, 0, -1, 0], 0),
                               ([0, 1, 0, -1, 0], 0),
                               ([0, 0, 1, -1, 0], 0),
                               ([0, 0, 0, -1, 1], 0)]

        lower_bound_input = torch.Tensor([2000, -0.4, -3.141592, 100, 0])
        upper_bound_input = torch.Tensor(
            [7000, -0.14, -3.141592 + 0.01, 150, 150])

    lower_bound_input = lower_bound_input.view([1, 1, 1, 5])
    upper_bound_input = upper_bound_input.view([1, 1, 1, 5])
    return property_conditions, (lower_bound_input, upper_bound_input)


def acasxu_initialize_zonotope(property_number, requires_grad=False):
    _, (lower_bound, upper_bound) = get_acaxsu_property(property_number)

    lower_bound.requires_grad_(requires_grad)
    upper_bound.requires_grad_(requires_grad)

    # a0 = ((lower_bound + upper_bound) / 2).view(1, 1, 1, 5)
    # A = torch.diag((upper_bound - lower_bound)/2).view(5, 1, 1, 5)
    a0 = (lower_bound + upper_bound) / 2
    A = torch.diag((
        (upper_bound - lower_bound)/2).view(-1)).view(5, 1, 1, 5)
    z = Zonotope(a0, A)

    return z


def acasxu_forward_pass(net, lower_bound, upper_bound):

    # a0 = ((lower_bound + upper_bound) / 2).view(1, 1, 1, 5)
    # A = torch.diag((upper_bound - lower_bound)/2).view(5, 1, 1, 5)

    a0 = (lower_bound + upper_bound) / 2
    A = torch.diag((
        (upper_bound - lower_bound)/2).view(-1)).view(5, 1, 1, 5)

    z = Zonotope(a0, A)

    z_net = Zonotope_Net(net)
    z_net.relaxation_at_layers.append(z)

    z_net.forward_pass()
    # z = z_net.relaxation_at_layers[-1]

    return z_net


def acasxu_compute_gradients(y, property_number):
    if property_number == 9:
        true_label = 3

        loss = (-y + y[0, true_label]).clamp_min(0).sum()
        loss.backward()

    else:
        raise NotImplementedError


def check_acasxu_property(z, property_number):
    if property_number == 9:
        true_label = 3

        A_diff = z.A - z.A[:, [true_label]]
        A_diff_abs = torch.sum(A_diff.abs_(), 0, keepdims=True)

        y = z.a0 - A_diff_abs

        isVerified = (torch.argmin(y) == true_label)

        return isVerified, y
    else:
        raise NotImplementedError


def acasxu_get_splits(upper_bound, lower_bound):
    gradients = torch.max(lower_bound.grad.abs(),
                          upper_bound.grad.abs()).view(-1)

    num_dimensions = gradients.numel()
    lower_bound = lower_bound.requires_grad_(False).detach().view(-1)
    upper_bound = upper_bound.requires_grad_(False).detach().view(-1)

    # bounds_after_normalization = z_net.relaxation_at_layers[1].get_bounds()
    # gap_after_normalization = (bounds_after_normalization[1] - \
    #     bounds_after_normalization[0]).squeeze()

    # print(gap_after_normalization.squeeze().shape, lower_bound.shape)

    gap = upper_bound - lower_bound

    smears = gradients * gap

    split_multiple = 20 / smears.sum()
    num_splits = [int(torch.ceil(smear * split_multiple)) for smear in smears]
    cuts = [torch.linspace(lower_bound[i], upper_bound[i], num_splits[i] + 1).tolist()
            for i in range(num_dimensions)]
    # step_size = [gap[i] / num_splits[i] for i in range(5)]
    # print(gradients, smears, num_splits)

    split_enumeration = [range(x) for x in num_splits]
    splits = []

    for split_numbers in itertools.product(*split_enumeration):

        lower_bound_list = [cuts[i][split_numbers[i]]
                            for i in range(num_dimensions)]
        upper_bound_list = [cuts[i][split_numbers[i] + 1]
                            for i in range(num_dimensions)]

        splits.append((torch.Tensor(lower_bound_list),
                       torch.Tensor(upper_bound_list)))

        # lower_bound = torch.Tensor([cuts[0][split_numbers[0]],
        #                             cuts[1][split_numbers[1]],
        #                             cuts[2][split_numbers[2]],
        #                             cuts[3][split_numbers[3]],
        #                             cuts[4][split_numbers[4]]])

        # upper_bound = torch.Tensor([cuts[0][split_numbers[0] + 1],
        #                             cuts[1][split_numbers[1] + 1],
        #                             cuts[2][split_numbers[2] + 1],
        #                             cuts[3][split_numbers[3] + 1],
        #                             cuts[4][split_numbers[4] + 1]])

        # splits.append((lower_bound, upper_bound))

    # for idx0 in range(num_splits[0]):
    #     for idx1 in range(num_splits[1]):
    #         for idx2 in range(num_splits[2]):
    #             for idx3 in range(num_splits[3]):
    #                 for idx4 in range(num_splits[4]):

    #                     lower_bound = torch.Tensor([cuts[0][idx0],
    #                                                 cuts[1][idx1],
    #                                                 cuts[2][idx2],
    #                                                 cuts[3][idx3],
    #                                                 cuts[4][idx4]])

    #                     upper_bound = torch.Tensor([cuts[0][idx0 + 1],
    #                                                 cuts[1][idx1 + 1],
    #                                                 cuts[2][idx2 + 1],
    #                                                 cuts[3][idx3 + 1],
    #                                                 cuts[4][idx4 + 1]])

    #                     splits.append((lower_bound, upper_bound))

    return splits


def acasxu_sort_splits(splits, dimension=0):
    splits.sort(key=lambda x: x[0][dimension].item())


def acasxu_split(lower_bound, upper_bound, y, property_number):
    acasxu_compute_gradients(y, property_number)

    gradients = torch.max(lower_bound.grad.abs(),
                          upper_bound.grad.abs()) + 1E-10

    lower_bound.requires_grad_(False).detach_()
    upper_bound.requires_grad_(False).detach_()

    gap = upper_bound - lower_bound
    smears = gradients * gap

    cut_index = smears.argmax()
    center_cut = (upper_bound[cut_index] + lower_bound[cut_index]) / 2

    lower_bound_new1 = lower_bound.clone()
    upper_bound_new1 = upper_bound.clone()
    lower_bound_new2 = lower_bound.clone()
    upper_bound_new2 = upper_bound.clone()

    upper_bound_new1[cut_index] = center_cut
    lower_bound_new2[cut_index] = center_cut

    return (lower_bound_new1, upper_bound_new1), (lower_bound_new2, upper_bound_new2)


def get_activation_signs(model, selected_labels, num_elements, test_set, eps):
    selected_layers = []
    activation_signs = {}
    activation_sign_bins = {}
    for idx_layer, layer in enumerate(model.net.layers):
        if isinstance(layer, torch.nn.ReLU):
            # Add indices of Affine layers (shifted by 1)
            selected_layers.append(idx_layer)
            activation_signs[idx_layer] = []
            activation_sign_bins[idx_layer] = []

    if selected_labels is None:
        real_selected_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        real_selected_labels = selected_labels

    for label in real_selected_labels:

        dataset = load_dataset_selected_labels_only(
            'mnist', [label], num_elements // len(real_selected_labels), test_set)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                  shuffle=False, num_workers=1)

        for idx_sample, (inputs, labels) in enumerate(tqdm(data_loader)):

            relaxation = Zonotope_Net(model.net)
            relaxation.process_input_once(inputs, eps, labels)

            for idx_layer in selected_layers:

                signs, bins = get_activation_signs_at_layer(
                    relaxation, idx_layer)

                activation_signs[idx_layer].append(signs)
                activation_sign_bins[idx_layer].append(bins)

    return activation_signs, activation_sign_bins, selected_layers


def get_activation_signs_at_layer(relaxation, idx_layer):
    z = relaxation.relaxation_at_layers[idx_layer]
    lower_bound, upper_bound = z.get_bounds()

    # Set positive signs to 1, negative to -1 and the rest 0
    negative_activations = (upper_bound <= 0).int()
    positive_activations = (lower_bound > 0).int()

    activation_summary = - negative_activations + positive_activations

    num_positive = positive_activations.sum().item()
    num_negative = negative_activations.sum().item()
    num_mixed = lower_bound.numel() - num_negative - num_positive

    bins = [num_negative, num_mixed, num_positive]

    return activation_summary, bins


def acasxu_get_activation_signs(net, property_number, depth):

    selected_layers = []
    activation_signs = {}
    activation_sign_bins = {}

    for idx_layer, layer in enumerate(net.layers):
        if isinstance(layer, torch.nn.ReLU):
            # Add indices of Affine layers (shifted by 1)
            selected_layers.append(idx_layer)
            activation_signs[idx_layer] = []
            activation_sign_bins[idx_layer] = []

    _, (lower_bound_input, upper_bound_input) = get_acaxsu_property(property_number)

    lower_bound_input.requires_grad_(True)
    upper_bound_input.requires_grad_(True)

    z_net = acasxu_forward_pass(net, lower_bound_input, upper_bound_input)

    _, y_worst_case = check_acasxu_property(
        z_net.relaxation_at_layers[-1], property_number)

    acasxu_compute_gradients(y_worst_case, property_number)

    splits = acasxu_get_splits(upper_bound_input, lower_bound_input)

    for lower_bound, upper_bound in splits:

        relaxation_nets = acasxu_split_recursive(
            lower_bound, upper_bound, net, property_number, 0, depth)

        for z_net in relaxation_nets:
            for idx_layer in selected_layers:
                signs, bins = get_activation_signs_at_layer(
                    z_net, idx_layer)

                activation_signs[idx_layer].append(signs)
                activation_sign_bins[idx_layer].append(bins)

    return activation_signs, activation_sign_bins, selected_layers


def acasxu_split_recursive(lower_bound, upper_bound, net, property_number, current_depth, depth):

    last_iteration = (current_depth == depth)

    lower_bound.requires_grad_(not last_iteration)
    upper_bound.requires_grad_(not last_iteration)

    z_net = acasxu_forward_pass(net, lower_bound, upper_bound)

    if last_iteration:
        return [z_net]

    else:
        _, y_worst_case = check_acasxu_property(
            z_net.relaxation_at_layers[-1], property_number)

        bounds_new1, bounds_new2 = acasxu_split(lower_bound, upper_bound,
                                                y_worst_case, property_number)

        relaxations1 = acasxu_split_recursive(
            bounds_new1[0], bounds_new1[1], net, property_number, current_depth+1, depth)
        relaxations2 = acasxu_split_recursive(
            bounds_new2[0], bounds_new2[1], net, property_number, current_depth+1, depth)

        relaxations1.extend(relaxations2)

        return relaxations1


def load_selected_labels_only(dataset, labels=None, num_elements=None, start_element=0):

    if not isinstance(dataset.targets, torch.Tensor):
        dataset.targets = torch.Tensor(dataset.targets).long()

    if labels is not None:

        if isinstance(labels, int):
            labels = [labels]

        idx = torch.zeros(len(dataset.targets)).bool()
        for label in labels:
            idx = idx | (dataset.targets == label)

        dataset.data = dataset.data[idx, :, :]
        dataset.targets = dataset.targets[idx]

    if num_elements is not None:
        dataset.data = dataset.data[start_element:num_elements +
                                    start_element, :, :]
        dataset.targets = dataset.targets[start_element: num_elements +
                                          start_element]

    return dataset


def load_dataset_selected_labels_only(dataset_name, labels=None, num_elements=None, test_set=True,
                                      start_element=0):

    if dataset_name == 'mnist':
        dataset = datasets.MNIST(PATH_EXAMPLES, train=not test_set, download=DOWNLOAD_DATA,
                                 transform=transforms.Compose([transforms.ToTensor()]))
    elif dataset_name == 'cifar':
        dataset = datasets.CIFAR10(PATH_EXAMPLES, train=not test_set, download=DOWNLOAD_DATA,
                                   transform=transforms.Compose([transforms.ToTensor()]))
    elif dataset_name == 'fashionmnist':
        dataset = datasets.FashionMNIST(PATH_EXAMPLES, train=not test_set, download=DOWNLOAD_DATA,
                                        transform=transforms.Compose([transforms.ToTensor()]))
    else:
        print('Unknown dataset:', dataset_name)
        raise RuntimeError

    return load_selected_labels_only(dataset, labels, num_elements, start_element)


def load_deepg_specs(idx, folder):
    specs_file = '{}deepg/{}/{}.csv'.format(PATH_EXAMPLES, folder, idx)

    with open(specs_file, 'r') as fin:
        lines = fin.readlines()

        lower_bounds_list = []
        upper_bounds_list = []
        interval_specs_list = []

        idx_line_of_spec = -1
        idx_interval = 0
        num_params = 1
        for idx_line_overall, line in enumerate(lines):

            if idx_line_overall == 0:
                # Read unperturbed input image
                values = torch.Tensor(list(map(float, line[:-1].split(','))))
                input = values[::2]
                continue
            elif idx_line_overall == 1:
                # Read image shape
                values = tuple(map(int, line[:-1].split(' ')))
                num_channels, num_rows, num_columns = values
                continue
            elif idx_line_overall == 2:
                # Read image shape
                values = int(line[:-1])
                num_params = values
                continue

            idx_line_of_spec += 1

            if idx_line_of_spec < num_params:
                # read specs for the parameters
                values = torch.Tensor(list(map(float, line[:-1].split(' '))))
                assert values.shape[0] == 2
                interval_specs_list.append(values)

            elif idx_line_of_spec == num_params:
                # read interval bounds for image pixels
                values = torch.Tensor(list(map(float, line[:-1].split(','))))
                lower_bounds_list.append(values[::2])
                upper_bounds_list.append(values[1::2])

            elif line == 'SPEC_FINISHED\n':
                idx_line_of_spec = -1
                idx_interval += 1
            else:
                # read polyhedra constraints for image pixels
                pass

    lower_bounds = torch.stack(lower_bounds_list, 0)
    upper_bounds = torch.stack(upper_bounds_list, 0)
    interval_specs = torch.stack(
        interval_specs_list, 0).view(-1, num_params, 2)

    lower_bounds = lower_bounds.view(
        [-1, num_rows, num_columns, num_channels]).permute(0, 3, 1, 2)
    upper_bounds = upper_bounds.view(
        [-1, num_rows, num_columns, num_channels]).permute(0, 3, 1, 2)
    input = input.view(
        [1, num_rows, num_columns, num_channels]).permute(0, 3, 1, 2)

    return input, lower_bounds, upper_bounds, interval_specs


def deepg_get_input_and_label(lower_bounds, upper_bounds, specs, net):

    lower_bound_min_index = specs[:, 0].abs().argmin()
    input = (lower_bounds[lower_bound_min_index] +
             upper_bounds[lower_bound_min_index]) * 0.5

    label = net(input)[0].argmax().item()
    return input, label


def print_layers_and_sizes(net):

    # x = torch.empty((1, 1, 28, 28))
    try:
        input_size = list(net.input_size)
    except Exception:
        input_size = [1, 28, 28]
    input_size.insert(0, 1)

    x = torch.empty(input_size)

    print('Network:')
    print('  Input:', list(x.shape))
    for idx_layer, layer in enumerate(net.layers):
        x = layer(x)
        print('  ({}):'.format(idx_layer), layer,
              '\n                -->', list(x.shape))
    print()


class IntermediateDataset(torch.utils.data.Dataset):

    def __init__(self, path, num_items=None, labels=None):
        self.path = path
        self.labels = labels if labels is not None else list(range(10))

        self.count_items()
        self.num_items = self.total_items
        if num_items is not None:
            self.num_items = min(num_items, self.num_items)

    def count_items(self):
        folder, prefix = tuple(self.path.rsplit('/', 1))
        self.items_count = {}

        for label in self.labels:
            prefix_ext = '{}_{}_'.format(prefix, label)
            num_items = len(
                [name for name in os.listdir(folder) if prefix_ext in name])
            self.items_count[label] = num_items

        self.total_items = sum(self.items_count.values())

    def __len__(self):
        return self.num_items

    def __getitem__(self, index):
        # Load data and get label
        label = None
        index_orig = index + 0
        for iter_label, num_items in self.items_count.items():
            if index < num_items:
                label = iter_label
                break
            else:
                index = index - num_items

        if label is None:
            print('Issue with loading dataset, Num_items: {}, idx_item: {}'.format(
                self.items_count, index_orig))
            raise EnvironmentError

        index = str(index).zfill(5)
        data = pickle.load(
            open('{}_{}_{}.pkl'.format(self.path, label, index), 'rb'))

        label = data['label']
        isVerified = data['isVerified']
        isPredicted = data['isPredicted']
        relaxations = data['intermediate_relaxations']

        relaxations = {x: Zonotope(relaxations[x][[0]], relaxations[x][1:])
                       for x in relaxations.keys()}

        return relaxations, label, isPredicted, isVerified


def custom_collate(batch):
    # print('batch: ', batch)

    return batch[0]


class TimeLogger:

    def __init__(self):
        self.timers = {}

    def add_timers(self, names):
        if not isinstance(names, list):
            names = [names]

        for name in names:
            if name in self.timers.keys():
                print('Timer {} already exists'.format(name))
            else:
                self.timers[name] = Timer()

    def start_timer(self, name):
        assert(name in self.timers.keys())
        self.timers[name].start()

    def stop_timer(self, name):
        assert(name in self.timers.keys())
        self.timers[name].stop()

    def print_summary(self, names=None):
        if names is None:
            names = self.timers.keys()

        print_strs = []
        longest_name_length = len(max(names, key=len))
        longest_name_length = int(1.5*longest_name_length)

        for name in names:
            timer = self.timers[name]
            print_str = 'Timer {}:'.format(name).ljust(longest_name_length)
            print_str += 'Counts: {},'.format(timer.count).ljust(15)

            print_str += 'Total Time: {:.3f},   Average Time: {:.5f}'.format(
                timer.cumulative_time, timer.average_time)

            print_strs.append(print_str)
        full_print_str = '\n'.join(print_strs)

        return full_print_str


class Timer:
    def __init__(self):
        self.cumulative_time = 0
        self.count = 0
        self.start_time = None
        self.average_time = 0

    def start(self):
        self.start_time = time()

    def stop(self):
        # assert(self.start_time is not None)

        self.count += 1
        if self.start_time is not None:
            self.cumulative_time += (time() - self.start_time)
            self.start_time = None
            self.average_time = self.get_average()

    def get_average(self):
        return self.cumulative_time / self.count


def initialize_logger():
    # Source: https://github.com/acschaefer/duallog/blob/master/duallog/duallog.py

    # Define the log rotation criteria.
    max_bytes = 1024**2
    backup_count = 100

    # file_msg_format = '%(asctime)s %(levelname)-8s: %(message)s'
    file_msg_format = '%(message)s'
    # console_msg_format = '%(levelname)s: %(message)s'
    console_msg_format = '%(message)s'

    file_name_format = '{year:04d}{month:02d}{day:02d}_'\
        '{hour:02d}{minute:02d}{second:02d}.txt'
    t = datetime.datetime.now()
    file_name = file_name_format.format(year=t.year, month=t.month, day=t.day,
                                        hour=t.hour, minute=t.minute, second=t.second)
    file_name = 'log/{}'.format(file_name)

    # Generating Logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Set up logging to the logfile.
    file_handler = logging.handlers.RotatingFileHandler(
        filename=file_name, maxBytes=max_bytes, backupCount=backup_count)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(file_msg_format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Set up logging to the console.
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter(console_msg_format)
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    # logger.info('Time: {}'.format(t))

    return logger


class Stream2Logger(object):
    # Source https://gist.github.com/Kerrigan29a/1c281e2fd6cf4b4de4f6

    def __init__(self, stream, logger, level):
        self._stream = stream
        self._logger = logger
        self._level = level
        self._buffer = []

    def write(self, message):
        self._buffer.append(message)
        if "\n" in message:
            self.flush()

    def flush(self):
        if self._buffer:
            message = "".join(self._buffer)
            if "\n" in message:
                self._logger.log(self._level, message.rstrip())
                self._buffer = []
            else:
                self._buffer = [message]
