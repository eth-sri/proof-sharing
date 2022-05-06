import numpy as np
import torch
import torch.nn as nn
import config


class Normalization(nn.Module):

    def __init__(self, device, mean=0.1307, sigma=0.3081):
        super(Normalization, self).__init__()
        # self.mean = torch.FloatTensor([0.1307]).view((1, 1, 1, 1)).to(device)
        # self.sigma = torch.FloatTensor([0.3081]).view((1, 1, 1, 1)).to(device)
        mean = np.array(mean) if isinstance(mean, list) else np.array([mean])
        sigma = np.array(sigma) if isinstance(sigma, list) else np.array([sigma])

        self.mean = nn.Parameter(torch.FloatTensor(
            mean).view((1, -1, 1, 1)), False)
        self.sigma = nn.Parameter(torch.FloatTensor(
            sigma).view((1, -1, 1, 1)), False)

    def forward(self, x):
        return (x - self.mean) / self.sigma


class Network(nn.Module):
    def __init__(self, device, input_size, conv_layers, fc_layers, n_class=10,
                 normalization=True, nonlinearity_after_conv='relu',
                 auxiliary_outputs_layers=None, mean=0.1307, sigma=0.3081):
        super(Network, self).__init__()
        self.input_size = input_size

        if auxiliary_outputs_layers is None:
            self.use_auxiliary_outputs = False
            self.auxiliary_outputs_layers = []
        else:
            self.use_auxiliary_outputs = True
            self.auxiliary_outputs_layers = auxiliary_outputs_layers

        self.create_architecture(
            input_size, n_class, device, conv_layers, fc_layers,
            normalization, mean, sigma, nonlinearity_after_conv)

        self.create_bias_free_duplicate()

        self.single_sign_weight_layers = None
        self.absolute_layers_without_bias = None

    def create_architecture(self, input_size, n_class, device, conv_layers,
                            fc_layers, normalization, mean, sigma, nonlinearity_after_conv):
        self.input_size = input_size
        self.n_class = n_class

        self.idx_layer_short = 0
        self.idx_layer_full = -1
        if self.use_auxiliary_outputs > 0:
            self.aux_outputs = {}

        if normalization:
            layers = [Normalization(device, mean, sigma)]
            self.idx_layer_full += 1
        else:
            layers = []

        num_channels = self.create_conv_layers(layers, conv_layers,
                                               nonlinearity_after_conv, input_size)
        layers.append(nn.Flatten())
        self.idx_layer_full += 1

        num_channels = self.create_fc_layers(layers, fc_layers, num_channels)

        assert(n_class == num_channels)

        self.layers = nn.Sequential(*layers)

    def create_conv_layers(self, layers, conv_layers, nonlinearity_after_conv, img_dim):

        img_dim = list(img_dim)

        for conv_layer in conv_layers:

            if len(conv_layer) == 4:
                n_channels, kernel_size, stride, padding = conv_layer
                kernel_size_mp = kernel_size
            else:
                n_channels, kernel_size, stride, padding, kernel_size_mp = conv_layer

            layers.append(nn.Conv2d(img_dim[0], n_channels, kernel_size,
                                    stride=stride, padding=padding))

            if nonlinearity_after_conv == 'max':
                layers.append(nn.MaxPool2d(kernel_size_mp))
            elif nonlinearity_after_conv == 'average':
                layers.append(nn.AvgPool2d(kernel_size_mp))
            elif nonlinearity_after_conv == 'relu':
                kernel_size_mp = 1
                layers.append(nn.ReLU())
            else:
                config.logger.error('Unknown nonlinearity layer')

            img_dim[0] = n_channels
            img_dim[1] = ((img_dim[1] + 2 * padding - kernel_size) //
                          stride) // kernel_size_mp + 1
            img_dim[2] = ((img_dim[2] + 2 * padding - kernel_size) //
                          stride) // kernel_size_mp + 1

            self.idx_layer_full += 2
            self.idx_layer_short += 1

            if self.idx_layer_short in self.auxiliary_outputs_layers:
                num_channels = n_channels * img_dim[1] * img_dim[2]

                self.aux_outputs[self.idx_layer_full] = \
                    nn.Sequential(nn.Flatten(),
                                  nn.Linear(num_channels, self.n_class))

        num_dim = 1
        for i in img_dim:
            num_dim *= i
        return num_dim

    def create_fc_layers(self, layers, fc_layers, num_channels):
        prev_fc_size = num_channels

        for i, fc_size in enumerate(fc_layers):
            layers.append(nn.Linear(prev_fc_size, fc_size))

            if i + 1 < len(fc_layers):
                layers.append(nn.ReLU())

            prev_fc_size = fc_size

            self.idx_layer_full += 2
            self.idx_layer_short += 1

            if self.idx_layer_short in self.auxiliary_outputs_layers:

                self.aux_outputs[self.idx_layer_full] = \
                    nn.Sequential(nn.Linear(fc_size, self.n_class))

        return prev_fc_size

    def create_bias_free_duplicate(self):
        self.bias_free_layers = self.duplicate_linear_layers()
        self.update_bias_free_layers()

    def update_bias_free_layers(self):
        for idx_layer, layer in self.bias_free_layers.items():
            layer.weight = self.layers[idx_layer].weight

    def duplicate_linear_layers(self, bias=False):
        duplicate_layers = {}

        for idx_layer, layer in enumerate(self.layers):
            if isinstance(layer, torch.nn.Linear):

                new_layer = layer.__class__(
                    layer.in_features, layer.out_features, bias=bias)

            elif isinstance(layer, torch.nn.Conv2d):
                new_layer = layer.__class__(
                    layer.in_channels, layer.out_channels, layer.kernel_size,
                    layer.stride, layer.padding, bias=bias)
            else:
                continue

            duplicate_layers[idx_layer] = new_layer

        return duplicate_layers

    def create_split_positive_negative_weights(self):
        self.single_sign_weight_layers = {}

        positive_weights_layers_without_bias = self.duplicate_linear_layers()
        negative_weights_layers_without_bias = self.duplicate_linear_layers()
        positive_weights_layers_with_bias = self.duplicate_linear_layers(True)
        negative_weights_layers_with_bias = self.duplicate_linear_layers(True)

        for key in positive_weights_layers_with_bias.keys():
            self.single_sign_weight_layers[key] = {
                'pos_with_bias': positive_weights_layers_with_bias[key],
                'neg_with_bias': negative_weights_layers_with_bias[key],
                'pos_without_bias': positive_weights_layers_without_bias[key],
                'neg_without_bias': negative_weights_layers_without_bias[key],
            }

        self.update_split_positive_negative_weights()

    def update_split_positive_negative_weights(self):
        for idx_layer, layer_dict in self.single_sign_weight_layers.items():
            pos_weight = self.layers[idx_layer].weight.data.clamp_min(0)
            neg_weight = self.layers[idx_layer].weight.data.clamp_max(0)
            bias = self.layers[idx_layer].bias

            layer_dict['pos_with_bias'].weight.data = pos_weight
            layer_dict['pos_without_bias'].weight.data = pos_weight
            layer_dict['pos_with_bias'].bias = bias

            layer_dict['neg_with_bias'].weight.data = neg_weight
            layer_dict['neg_without_bias'].weight.data = neg_weight
            layer_dict['neg_with_bias'].bias = bias

    def create_absolute_weights(self):
        self.absolute_layers_without_bias = self.duplicate_linear_layers()
        self.update_absolute_weight_layers()

    def update_absolute_weight_layers(self):
        for idx_layer, layer in self.absolute_layers_without_bias.items():
            layer.weight.data = self.layers[idx_layer].weight.data.abs()

    def remove_gradient(self):
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad_(False)

        for layer in self.bias_free_layers.values():
            for param in layer.parameters():
                param.requires_grad_(False)

        if self.absolute_layers_without_bias is not None:
            for layer in self.absolute_layers_without_bias.values():
                for param in layer.parameters():
                    param.requires_grad_(False)

        if self.single_sign_weight_layers is not None:
            for layer_dict in self.single_sign_weight_layers.values():
                for layer in layer_dict.values():
                    for param in layer.parameters():
                        param.requires_grad_(False)

    def add_gradient(self):
        for layer in self.layers:
            if isinstance(layer, Normalization):
                continue
            else:
                for param in layer.parameters():
                    param.requires_grad_(True)

    def forward(self, x, aux_outputs=False):
        if hasattr(self, 'use_auxiliary_outputs'):

            if self.use_auxiliary_outputs and \
                    (self.training or aux_outputs):
                outputs = []
                for idx_layer, layer in enumerate(self.layers):
                    x = layer(x)
                    if idx_layer in self.aux_outputs.keys():
                        y_aux = self.aux_outputs[idx_layer](x)
                        outputs.append(y_aux)

                outputs.insert(0, x)
                return outputs

        return self.layers(x)
