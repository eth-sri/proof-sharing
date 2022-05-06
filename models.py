import torch
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
from time import time
from relaxations import Zonotope, Zonotope_Net, Star, Box_Star, Star_Net, Box_Net  # noqa
import utils
# import config
import logging
import gurobipy as gp
from gurobipy import GRB

logger = logging.getLogger()


class Model():

    def __init__(self, device='cpu', verbosity=2):
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = device
        self.verification_templates = {}
        self.verification_counts = {}
        self.verbosity = verbosity
        self.nullspace_directions = {}
        self.milp_models = {}
        self.timelogger = None

    def set_net(self, net):
        self.net = net.to(self.device)

    def train(self, dataset, epochs=1, validation_set=None, robust=False, eps=2/255,
              auxiliary_outputs=False, auxiliary_weights=[]):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                                  shuffle=True, num_workers=2)
        disable_tqdm = self.verbosity < 2

        optimizer = optimizer = torch.optim.Adam(
            self.net.parameters(), lr=0.001)

        if robust:
            num_steps = 5
            bounds = [0.0, 1.0]

        for epoch in range(epochs):  # loop over the dataset multiple times
            self.net.train()

            running_loss = 0.0
            for idx_batch, (inputs, labels) in enumerate(tqdm(data_loader, disable=disable_tqdm)):

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if robust:
                    self.net.eval()
                    inputs = self.pgd_untargeted_attack(
                        inputs, labels, eps, num_steps, bounds)
                    self.net.train()

                optimizer.zero_grad()
                outputs = self.net(inputs)
                if auxiliary_outputs:
                    loss = self.criterion(outputs[0], labels)

                    for idx_output, output in enumerate(outputs[1:]):
                        loss += auxiliary_weights[idx_output] * \
                            self.criterion(output, labels)
                else:
                    loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                n = 10

            test_set = validation_set if validation_set is not None else dataset
            acc = self.get_accuracy(test_set)

            print_str = '[{}, {}] Loss: {:.3f}, Accuracy: {:.3f}'.format(
                epoch + 1, idx_batch + 1, running_loss / n, acc)
            if robust:
                rob_acc = self.get_accuracy(test_set, robust=True, eps=eps)
                print_str += ', Robust Accuracy: {:.3f}'.format(rob_acc)
            self.print(print_str, 1)

        self.print('Finished Training', 1)

    def get_accuracy(self, dataset, robust=False, eps=2/255):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                                  shuffle=False, num_workers=2)
        self.net.eval()
        if robust:
            num_steps = 10
            bounds = [0.0, 1.0]

        outputs = []
        num_true = []
        counts_true = 0
        counts_total = dataset.targets.shape[0]

        disable_tqdm = self.verbosity < 2

        for _, (inputs, labels) in enumerate(tqdm(data_loader, disable=disable_tqdm)):

            inputs, labels = inputs.to(self.device), labels.to(self.device)

            if robust:
                inputs = self.pgd_untargeted_attack(
                    inputs, labels, eps, num_steps, bounds)

            outputs = self.net(inputs)

            equal_outputs = torch.argmax(outputs, 1) == labels
            num_true = equal_outputs.long()
            sum_true = num_true.sum().item()

            counts_true += sum_true
        self.print('Predicted correctly: {}'.format(counts_true), 1)

        return counts_true / counts_total

    def verify_simple(self, dataset, eps, **params):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                  shuffle=False, num_workers=2)
        self.net.eval()

        counts_true = 0
        counts_total = dataset.targets.shape[0]

        disable_tqdm = self.verbosity < 2
        for _, (inputs, labels) in enumerate(tqdm(data_loader, disable=disable_tqdm)):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            if torch.argmax(self.net(inputs), 1) == labels:

                relaxation = params['relaxation_net_type'](
                    self.net, self.device, params['relu_transformer'])

                if params['lambdas_optimization_for_all_samples']:
                    counts_true += relaxation.process_input_iteratively(
                        inputs, eps, labels, num_steps=1,
                        step_duration=params['lambdas_optimization_num_iterations'])

                else:
                    counts_true += relaxation.process_input_once(
                        inputs, eps, labels)

        self.print('Verified: {}'.format(counts_true), 1)
        return counts_true / counts_total

    def verify_proof_transfer(self, dataset, eps, **params):
        self.net.eval()
        torch.manual_seed(12)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                  shuffle=True, num_workers=2)
        counts_true = 0
        counts_total = dataset.targets.shape[0]

        layer_indices = self.layer_indices_from_type(
            params['selected_layers'], params['num_skip_layers'], params['exclude_last_layer'])

        self.reset_counts(layer_indices)

        params['eps'] = eps
        params['selected_layers_indices'] = layer_indices
        params['submatching_only'] = False

        disable_tqdm = self.verbosity < 2
        for idx_sample, (inputs, labels) in enumerate(tqdm(data_loader, disable=disable_tqdm)):
            if idx_sample == params['submatching_only_after_num_examples']:
                params['submatching_only'] = True
                if params['submatching_only_after_num_examples'] > 0:
                    self.print(self.verification_counts, 1)

                    self.widen_templates(params['widening_methods'], **params)

            inputs, labels = inputs.to(self.device), labels.to(self.device)

            counts_true += self.apply_proof_transfer(
                inputs, labels, **params)

        self.print(self.verification_counts, 1)
        return counts_true / counts_total

    def apply_proof_transfer(self, inputs, labels, **params):

        # Wrong prediction
        if not torch.argmax(self.net(inputs), 1) == labels:
            self.verification_counts['false_prediction'] += 1
            self.print('False prediction: 0')
            return 0

        label_key = labels.item()

        if label_key not in self.verification_templates:
            self.verification_templates[label_key] = {}
        templates = self.verification_templates[label_key]

        z_net = params['relaxation_net_type'](self.net, self.device)
        z_net.initialize(inputs, params['eps'])
        z_net.is_lambda_optimizable = params['lambdas_optimization_for_all_samples']

        # Skip layers
        if params['num_skip_layers'] < 0:
            params['num_skip_layers'] += len(self.net.layers)

        if params['num_skip_layers'] > len(self.net.layers):
            logger.error('Too many layers skipped')
            raise RuntimeError

        for idx_layer in range(params['num_skip_layers']):
            z_net.apply_layer(idx_layer)

        for idx_layer in range(params['num_skip_layers'], len(self.net.layers) - 1):

            z_net.apply_layer(idx_layer)
            if idx_layer not in params['selected_layers_indices']:
                continue

            relaxation = z_net.relaxation_at_layers[-1]

            if idx_layer not in templates:
                templates[idx_layer] = []
            template_relaxation = templates[idx_layer]

            # Submatch
            isSubmatch, idx_submatch = self.try_submatching(
                template_relaxation, relaxation, **params)
            if isSubmatch:
                self.verification_counts['submatching'][idx_layer][label_key] += 1
                self.verification_counts['submatching_total'][idx_layer] += 1
                self.print('Submatching at layer {}: 1'.format(idx_layer))
                return 1

            if params['submatching_only']:
                continue

            # Union
            isUnionVerifiable = self.try_verify_union(
                template_relaxation, relaxation, labels, idx_layer, **params)

            if isUnionVerifiable:
                self.verification_counts['union'][idx_layer] += 1
                self.print('Union verified at layer {}: 1'.format(idx_layer))
                return 1

            # Create new template
            if len(template_relaxation) < params['num_templates_per_class']:
                isVerifiable = self.try_verify_as_template(
                    relaxation, labels, idx_layer, **params)

                if isVerifiable:
                    self.verification_counts['new_template'] += 1
                    self.print(
                        'New template created at layer {}: 1'.format(idx_layer))
                    return 1
            continue

        # When we do not succeed with submatching or verifying a union at earlier layers,
        # we pass it through the final layer and perform standard verification
        # z_net.truncate(-1)
        isVerified = z_net.process_from_layer(
            labels, len(self.net.layers) - 1,
            params['lambdas_optimization_for_all_samples'],
            step_duration=params['lambdas_optimization_num_iterations'])
        if isVerified:
            self.verification_counts['manual_positive'] += 1
        else:
            self.verification_counts['manual_negative'] += 1
        self.print('Manual verification: {}'.format(isVerified))

        return isVerified

    def try_submatching(self, template, relaxation, **params):
        for idx_z, z in enumerate(template):
            # isSubmatch = z.submatching(relaxation, params['submatching_method'])
            isSubmatch = z.submatching(relaxation)
            if isSubmatch:
                return True, idx_z
        return False, -1

    def try_verify_union(self, template, relaxation, labels, idx_layer,
                         **params):

        for idx_template, z in enumerate(template):
            z_union = z.union(
                relaxation, params['union_method'], **params['params_union'])

            isVerified = self.verify_and_update_relaxation(z_union, labels, idx_layer, idx_template,
                                                           **params)

            if isVerified:
                return True

        return False

    def try_verify_as_template(self, relaxation, labels, idx_layer,
                               **params):
        if params['params_union']['order_reduction']:
            relaxation = relaxation.order_reduction()

        idx_template = len(
            self.verification_templates[labels.item()][idx_layer])

        isVerified = self.verify_and_update_relaxation(relaxation, labels, idx_layer, idx_template,
                                                       **params)

        return isVerified

    def verify_and_update_relaxation(self, relaxation, labels, idx_layer, idx_template, **params):

        isVerified, relaxation_net = self.propagate_relaxation(relaxation,
                                                               labels, idx_layer, **params)
        if isVerified:
            templates = self.verification_templates[labels.item()]
            overwrite = len(templates[idx_layer]) > idx_template

            if params['propagate_templates']:
                for idy_layer in range(idx_layer, len(self.net.layers) - 1):
                    if idy_layer in self.verification_templates[labels.item()]:

                        new_relaxation = relaxation_net.relaxation_at_layers[idy_layer - idx_layer]

                        if overwrite:
                            templates[idy_layer][idx_template] = new_relaxation
                        else:
                            templates[idy_layer].append(new_relaxation)

            else:
                if overwrite:
                    templates[idx_layer][idx_template] = relaxation
                else:
                    templates[idx_layer].append(relaxation)

        return isVerified

    def propagate_relaxation(self, relaxation, labels, idx_layer, **params):

        relaxation_net = params['relaxation_net_type'](self.net, self.device)
        relaxation_net.relaxation_at_layers.append(relaxation)

        num_steps = params['lambdas_optimization_num_iterations'] // 20
        step_duration = params['lambdas_optimization_num_iterations'] % 20

        isVerified = relaxation_net.process_from_layer(
            labels, start_layer=idx_layer+1,
            lambda_optimization=params['lambdas_optimization_for_templates'],
            num_steps=num_steps,
            step_duration=step_duration)

        if not isVerified and params['template_verification_with_recursion'] and \
                (idx_layer < (len(self.net.layers) - 2)):
            a0_rot = relaxation.a0.matmul(relaxation.U_rot)
            A_rot = relaxation.A.matmul(
                relaxation.U_rot).abs().sum(0, keepdims=True)

            lower_bound = a0_rot - A_rot
            upper_bound = a0_rot + A_rot

            if isinstance(relaxation, Star):
                constraints = (relaxation.C, relaxation.d)
            else:
                constraints = None

            isVerified = relaxation_net.process_input_recursively(
                lower_bound, upper_bound, labels, split_max_depth=3, start_layer=idx_layer+1,
                U_rot_inv=relaxation.U_rot_inv, constraints=constraints)

        if not isVerified and \
            ((params['template_verification_with_milp'] and idx_layer < (len(self.net.layers) - 2))
             or isinstance(relaxation, Star)):

            isVerified, _ = self.template_verification_with_milp(
                relaxation, labels, idx_layer)

        return isVerified, relaxation_net, relaxation

    def submatch_points(self, dataset, **params):

        assert(len(self.verification_templates) > 0)
        selected_labels = self.verification_templates.keys()
        selected_layers = []
        for x in self.verification_templates.values():
            selected_layers.extend(list(x.keys()))
        selected_layers = list(set(selected_layers))

        self.print('{} {}'.format(selected_labels, selected_layers), 1)

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                                  shuffle=False, num_workers=2)
        self.net.eval()

        counts_submatches = 0
        counts_predicted = 0
        counts_total = dataset.targets.shape[0]

        disable_tqdm = self.verbosity < 2
        for _, (inputs, _) in enumerate(tqdm(data_loader, disable=disable_tqdm)):
            x = inputs
            isSubmatches_Total = torch.zeros(x.shape[0]).bool()

            for idx_layer, layer in enumerate(self.net.layers):
                x = layer(x)

                if idx_layer in selected_layers:

                    for idx_label, template_of_label in self.verification_templates.items():
                        templates = template_of_label[idx_layer]

                        for z in templates:
                            isSubmatches = z.submatching_single(x)
                            isSubmatches_Total += isSubmatches

            isPrediction = torch.zeros_like(isSubmatches_Total).bool()
            y = x.argmax(1)
            for idx_label in selected_labels:
                isPrediction += (y == idx_label)

            counts_submatches += isSubmatches_Total.int().sum().item()
            counts_predicted += isPrediction.int().sum().item()

        return counts_submatches, counts_predicted, counts_total

    def generate_templates_once(self, dataset, eps, **params):

        self.print('Generate templates:', 1)

        self.net.eval()
        torch.manual_seed(12)
        layer_indices = self.layer_indices_from_type(
            params['selected_layers'], params['num_skip_layers'], params['exclude_last_layer'])
        self.reset_counts(layer_indices)

        if params['union_method'] == 'nullspace':
            self.derive_nullspace_directions(layer_indices[0])

        relaxation_lists = {}
        for label in range(10):
            relaxation_lists[label] = {}
            for idx_layer in layer_indices:
                relaxation_lists[label][idx_layer] = []

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                  shuffle=False, num_workers=1)

        disable_tqdm = self.verbosity < 2

        available_labels = []

        for _, (inputs, labels) in enumerate(tqdm(data_loader, disable=disable_tqdm)):

            relaxation = params['relaxation_net_type'](self.net)
            isVerified = relaxation.process_input_once(inputs, eps, labels)

            if isVerified:
                available_labels.append(labels.item())
                for idx_layer in layer_indices:
                    relaxation_lists[labels.item()][idx_layer].append(
                        relaxation.relaxation_at_layers[idx_layer+1])

        self.print('Used relaxations: {}'.format(len(available_labels)), 1)

        available_labels = list(sorted(set(available_labels)))

        for label in tqdm(available_labels, disable=disable_tqdm):
            self.verification_templates[label] = {}
            for idx_layer in layer_indices:
                self.verification_templates[label][idx_layer] = []
                rel = relaxation_lists[label][idx_layer]
                if len(rel) > 0:
                    self.create_template(
                        relaxation_lists[label][idx_layer], label, idx_layer, **params)
                relaxation_lists[label][idx_layer] = []

        # self.template_reunion(**params)

        # self.print(self.verification_counts, 1)

    def create_template(self, relaxation_list, label, idx_layer, **params):

        # num_elements = len(relaxation_list)
        # num_clusters = params['num_templates_per_class']

        # self.print('Start clustering layer {} label {}'.format(idx_layer, label),2)
        # t = time()

        if len(relaxation_list) <= params['num_templates_per_class']:
            cluster_label = list(range(len(relaxation_list)))
        else:
            centroid, cluster_label = self.cluster_relaxations(
                relaxation_list, idx_layer, **params)
            # self.print(cluster_label,3)

        # self.print('End clustering layer {} label {}'.format(idx_layer, label),2)
        # self.print('Time: {}'.format(time() - t),2)

        true_num_clusters = len(set(cluster_label)) - \
            (1 if -1 in cluster_label else 0)

        for idx_cluster in range(true_num_clusters):

            assigned_relaxations = [z for (z, cluster_label) in zip(
                relaxation_list, cluster_label) if cluster_label == idx_cluster]

            z_template_list, _ = self.create_cluster_union(
                assigned_relaxations, label, idx_layer, **params)

            for z_template in z_template_list:

                if z_template is not None:
                    self.verification_templates[label][idx_layer].append(
                        z_template)
                    self.verification_counts['new_template'] += 1

    def cluster_relaxations(self, relaxation_list, idx_layer, **params):
        from sklearn import cluster, mixture  # noqa: F401
        # from scipy.cluster.vq import kmeans2
        import numpy as np
        np.random.seed(17)

        num_clusters = params['num_templates_per_class']
        method = params['cluster_method']

        if params['union_method'] == 'nullspace':
            nullspace_layer = idx_layer + 1
            while not isinstance(self.net.layers[nullspace_layer],
                                 (torch.nn.Linear, torch.nn.Conv2d)):
                nullspace_layer += 1

            U_direct = self.nullspace_directions[nullspace_layer][0][:, :10]

            a0_list = [z.a0_flat.matmul(U_direct)
                       for z in relaxation_list]
        else:

            # Using center of relaxations at current layer
            lower_bound, upper_bound = relaxation_list[0].get_bounds()
            for z in relaxation_list[1:]:
                bounds = z.get_bounds()

                lower_bound = torch.min(lower_bound, bounds[0])
                upper_bound = torch.max(upper_bound, bounds[1])

            active_neurons = upper_bound[0, :] > lower_bound[0, :]

            a0_list = [z.a0_flat[:, active_neurons]
                       for z in relaxation_list]

            # Using center of relaxation after next layer
            # a0_list = [self.net.layers[idx_layer+1](z.a0)
            #            for z in relaxation_list]

            # Using center of relaxation after last layer
            # a0_list = []
            # for z in relaxation_list:
            #     x = z.a0
            #     for layer in self.net.layers[idx_layer:]:
            #         x = layer(x)
            #     a0_list.append(x)

        a0_cat = torch.cat(a0_list, 0).numpy().astype(np.float64)

        if method == 'kmeans':
            alg = cluster.KMeans(num_clusters, init='k-means++')
            cluster_label = alg.fit_predict(a0_cat)
            centroid = torch.Tensor(alg.cluster_centers_)

        elif method == 'gaussianmixture':
            try:
                alg = mixture.GaussianMixture(num_clusters)
                cluster_label = alg.fit_predict(a0_cat)
                centroid = torch.Tensor(alg.means_)
            except Exception:
                self.print('Skip to kmeans', 2)
                alg = cluster.KMeans(num_clusters, init='k-means++')
                cluster_label = alg.fit_predict(a0_cat)
                centroid = torch.Tensor(alg.cluster_centers_)

        elif method == 'dbscan':
            alg = cluster.DBSCAN(eps=0.7, min_samples=5, metric='chebyshev')
            cluster_label = alg.fit_predict(a0_cat)
            centroid = None

        elif method == 'optics':
            alg = cluster.OPTICS(max_eps=10.0, min_samples=5, metric='chebyshev',
                                 cluster_method='dbscan', eps=0.7)
            cluster_label = alg.fit_predict(a0_cat)
            centroid = None

        elif method == 'hdbscan':
            import hdbscan

            alg = hdbscan.HDBSCAN(min_cluster_size=6, metric='chebyshev')
            cluster_label = alg.fit_predict(a0_cat)
            centroid = None

        else:
            self.print('Unknown cluster method: {}'.format(method), 0)
            raise RuntimeError

        return centroid, cluster_label

    def create_cluster_union(self, relaxation_list, label, idx_layer, **params):

        if params['union_method'] == 'nullspace':

            nullspace_layer = idx_layer + 1
            while not isinstance(self.net.layers[nullspace_layer],
                                 (torch.nn.Linear, torch.nn.Conv2d)):
                nullspace_layer += 1

            params['params_union']['U'] = self.nullspace_directions[nullspace_layer][0]
            params['params_union']['U_inv'] = self.nullspace_directions[nullspace_layer][1]

        z_union_all = relaxation_list[0].union(
            relaxation_list[1:], params['union_method'], **params['params_union'])

        isVerified, z_net, z_union_all = self.propagate_relaxation(
            z_union_all, label, idx_layer, **params)

        if isVerified and params['template_minimization_method'] != 'truncation_and_widening':
            self.print('Template creation (Label:{}, Layer:{}) relaxations used: {}/{}'.format(
                label, idx_layer, len(relaxation_list), len(relaxation_list)), 1)
            return [z_union_all], list(range(len(relaxation_list)))

        else:
            method = params['template_minimization_method']

            if method == 'iterative_union':

                template_list, unused_relaxations = self.iterative_union(
                    z_union_all, relaxation_list, idx_layer, label, **params)

            elif method == 'iterative_truncation':

                template_list, unused_relaxations = self.iterative_truncation(
                    z_union_all, relaxation_list, idx_layer, label, **params)

            elif method == 'truncation_and_widening':

                template_list, unused_relaxations = self.truncation_and_widening(
                    z_union_all, relaxation_list, idx_layer, label, **params)

            else:
                self.print(
                    'Unknown template generation method: {}'.format(method), 0)
                raise RuntimeError

            if len(unused_relaxations) > 1 and params['template_recursive_splitting']:

                if len(unused_relaxations) <= params['num_templates_per_class']:
                    cluster_label = list(range(len(unused_relaxations)))
                else:
                    _, cluster_label = self.cluster_relaxations(
                        unused_relaxations, idx_layer, **params)

                true_num_clusters = len(set(cluster_label)) - \
                    (1 if -1 in cluster_label else 0)

                for idx_cluster in range(true_num_clusters):

                    assigned_relaxations = [z for (z, cluster_label) in zip(
                        unused_relaxations, cluster_label) if cluster_label == idx_cluster]

                    if len(assigned_relaxations) >= params['template_minimal_size']:
                        z_template_list, _ = self.create_cluster_union(
                            assigned_relaxations, label, idx_layer, **params)

                        template_list.extend(z_template_list)

            return template_list, []

    def iterative_union(self, z_template, relaxation_list, idx_layer, label, **params):
        center_diff = [(z_template.a0 - z.a0).abs().sum()
                       for z in relaxation_list]
        indices_sorted = torch.Tensor(center_diff).argsort()

        z_union_list = []
        indices_used = []
        unused_relaxations = []
        z_union = None

        for idx_zono in indices_sorted:
            z = relaxation_list[idx_zono]

            if len(z_union_list) == 0:

                if params['union_method'] == 'nullspace':
                    U = params['params_union']['U']
                    U_inv = params['params_union']['U_inv']
                else:
                    U = None
                    U_inv = None
                z_union_new = z.order_reduction(U, U_inv)

            else:
                z_union_new = z.union(z_union_list, params['union_method'],
                                      **params['params_union'])

            isVerified, _, z_union_new = self.propagate_relaxation(
                z_union_new, label, idx_layer, **params)

            if isVerified:
                z_union_list.append(z)
                indices_used.append(idx_zono)
                z_union = z_union_new
            else:
                unused_relaxations.append(z)

        self.print('Template creation (Label:{}, Layer:{}) relaxations used: {}/{}'.format(
            label, idx_layer, len(z_union_list), len(relaxation_list)), 1)

        if z_union is None:
            template_list = []
        else:
            template_list = [z_union]

        return template_list, unused_relaxations

    def iterative_truncation(self, z_template, relaxation_list, idx_layer, label,
                             max_iterations=30, **params):

        if params['template_prewidening']:
            z_template = z_template.widening(
                params['widening_methods'], **params['params_widening'])

        num_relaxations = len(relaxation_list)

        if not isinstance(z_template, Box_Star):
            new_relaxation = Star(z_template)
        else:
            new_relaxation = z_template

        relaxation_net = Star_Net(self.net)
        relaxation_net.relaxation_at_layers = [new_relaxation]
        relaxation_net.use_overapproximation_only = False
        relaxation_net.use_general_zonotope = False

        isVerified = self.apply_iterative_truncation(
            relaxation_net, relaxation_list, idx_layer, label,
            max_iterations=max_iterations, **params)

        if isVerified:
            print_str = 'Template creation (Label:{}, Layer:{}) relaxations used: {}'.format(
                label, idx_layer, num_relaxations)
            print_str += ' with {} constraints'.format(
                0 if new_relaxation.d is None else new_relaxation.d.numel())
            self.print(print_str, 1)

            template_list = [new_relaxation]
            unused_relaxations = []

        else:
            print_str = 'Template creation (Label:{}, Layer:{}) Failed using {} relaxations'.format(
                label, idx_layer, num_relaxations)
            self.print(print_str, 1)

            template_list = []
            unused_relaxations = relaxation_list

        return template_list, unused_relaxations

    def apply_iterative_truncation(self, relaxation_net, relaxation_list, idx_layer, label,
                                   interpolation_weight=0.05, max_iterations=20, **params):

        method = params['truncation_method']
        max_iterations = 0

        if method == 'exact':
            return self.apply_iterative_truncation_exact(relaxation_net, relaxation_list, idx_layer, label,
                                                         interpolation_weight, max_iterations)
        elif method == 'pgd':
            return self.apply_iterative_truncation_pgd(relaxation_net, relaxation_list, idx_layer, label,
                                                       interpolation_weight, max_iterations)

        elif method == 'submatch':
            return self.apply_iterative_truncation_submatch(relaxation_net, relaxation_list, idx_layer, label,
                                                            interpolation_weight, max_iterations, **params)
        else:
            self.print('Unknown truncation method: {}'.format(method), 0)
            raise NotImplementedError

    def apply_iterative_truncation_exact(self, relaxation_net, relaxation_list, idx_layer, label,
                                         interpolation_weight, max_iterations):

        t = time()

        relaxation_net.use_overapproximation_only = False
        relaxation_net.use_general_zonotope = False
        relaxation_net.num_solutions = 10
        relaxation_net.early_stopping_objective = 1.0
        relaxation_net.early_stopping_bound = -1E-5
        relaxation_net.timelimit = 60*30
        relaxation_net.use_tighter_bounds = True
        relaxation_net.use_tighter_bounds_using_milp = True
        relaxation_net.use_lp = False
        relaxation_net.use_retightening = False
        relaxation_net.milp_neuron_ratio = 1.0
        # max_iterations = 50
        # print(relaxation_net.milp_neuron_ratio, max_iterations)

        if idx_layer in self.milp_models.keys():
            milp_model = self.milp_models[idx_layer]
        else:
            milp_model = None

        # if relaxation_net.milp_model is not None:
        #     # relaxation_net.milp_model.update()
        #     milp_variables = [
        #         v for v in relaxation_net.milp_model.getVars() if 'y_diff_max' in v.VarName]

        #     # Check if model was previously run
        #     if len(milp_variables) == 0:
        #         isVerified, violation = relaxation_net.run_optimization(label)

        #     else:
        #         relaxation_net.milp_model.optimize()
        #         isVerified, violation = relaxation_net.evaluate_optimization()
        # else:
        isVerified, violations = relaxation_net.process_from_layer(
            label, idx_layer + 1, return_violation=True, milp_model=milp_model)

        idx_iter = 0

        if violations is None:
            max_iterations = 0
        else:
            self.print('-1, loss: {:.4f}, d: None, time: {:.2f}'.format(
                relaxation_net.milp_model.objVal, time() - t), 2)
            if relaxation_net.milp_model.Status == 2:
                prev_objective = relaxation_net.milp_model.objVal
            else:
                prev_objective = 1E5

        while not isVerified and idx_iter < max_iterations:

            C, d = self.derive_hyperplane_intersection(
                relaxation_net.relaxation_at_layers[0], violations, relaxation_list,
                idx_layer, label, interpolation_weight, method='static_interpolation')

            t = time()
            isVerified, violations = relaxation_net.rerun_with_additional_constraint(
                C, d)

            if violations is None:
                break

            objective = relaxation_net.milp_model.objVal

            d_values = ', '.join(['{:.3f}'.format(x.item()) for x in d])
            self.print('{}, loss: {:.4f}, d: {}, time: {:.2f}'.format(
                idx_iter, objective, d_values, time() - t), 2)

            if (idx_iter > 10 and objective > 1.0) or (idx_iter > 20 and objective > 0.2):
                self.print('Loss too large, stop truncation', 2)
                break

            if relaxation_net.milp_model.Status == 2:
                if objective >= prev_objective:
                    break
                else:
                    prev_objective = objective
            idx_iter += 1

        return isVerified

    def apply_iterative_truncation_pgd(self, relaxation_net, relaxation_list, idx_layer, label,
                                       interpolation_weight, max_iterations):

        s = relaxation_net.relaxation_at_layers[0]

        num_points = 256
        num_points_initialization_pgd = 1024
        num_points_initialization_frank_wolfe = 1024
        num_iterations_pgd = 20
        num_iterations_frank_wolfe = 20
        learning_rate_pgd = 5E0
        max_offset_advarsarial_label = 0.2
        use_early_stop = True
        use_frank_wolfe = False

        num_eps = s.A.shape[0]
        num_dimensions = s.a0.numel()
        advarsarial_labels = [i for i in range(10) if i != label]

        print_str = ''
        for var in [num_points, learning_rate_pgd, use_early_stop, use_frank_wolfe]:
            varname = [k for k, v in locals().items() if v == var][0]
            print_str += '{}: {}, '.format(varname, var)

        self.print(print_str, 3)

        m = gp.Model()
        m.Params.OutputFlag = 0
        m.Params.FeasibilityTol = 1e-6
        eps_vars = []
        x_init_vars = []
        for i in range(num_eps):
            v = m.addVar(vtype=GRB.CONTINUOUS, lb=-1, ub=1)
            eps_vars.append(v)

        for i in range(num_dimensions):
            v = m.addVar(vtype=GRB.CONTINUOUS, lb=-
                         GRB.INFINITY, ub=GRB.INFINITY)
            x_init_vars.append(v)

        for i in range(num_dimensions):
            expr = gp.LinExpr()
            expr += -1*x_init_vars[i]

            for k in range(num_eps):
                expr.addTerms(s.A[k, i], eps_vars[k])
            expr.addConstant(s.a0[0, i])
            m.addConstr(expr, GRB.EQUAL, 0)

        def update_lp_model(C, d):
            if d is None:
                return
            num_constraints = d.numel()

            for i in range(num_constraints):
                expr = gp.LinExpr()
                for j in range(num_dimensions):
                    expr.addTerms(C[i, j], x_init_vars[j])

                m.addConstr(expr, GRB.LESS_EQUAL, d[i])

        update_lp_model(s.C, s.d)

        def project_point_on_star(points, s):

            points_inside = s.submatching_single(points)
            x_guess_candidates = sample_points_inside_star(
                s, num_points_initialization_frank_wolfe)

            for idx_point in range(num_points):
                if points_inside[idx_point]:
                    continue
                p = points[idx_point, :]

                if use_frank_wolfe:

                    distances = (x_guess_candidates - p).norm(p=2, dim=1)
                    min_distance_index = distances.argmin()
                    x_guess = x_guess_candidates[min_distance_index]

                    for idx_iter in range(num_iterations_frank_wolfe):
                        direction = p - x_guess

                        expr = gp.LinExpr()
                        for j in range(num_dimensions):
                            expr.addTerms(direction[j], x_init_vars[j])
                        m.setObjective(expr, GRB.MAXIMIZE)
                        m.optimize()

                        x_correction = [v.x for v in x_init_vars]
                        x_correction = torch.Tensor(x_correction).view_as(p)
                        x_guess_prev = x_guess

                        step_size = 2 / (2+idx_iter)
                        x_guess = x_guess + step_size * \
                            (x_correction - x_guess)

                        if (x_guess_prev - x_guess).abs().max() < 1E-3:
                            break

                else:

                    expr = gp.QuadExpr()
                    for j in range(num_dimensions):
                        expr.addTerms(1.0, x_init_vars[j], x_init_vars[j])
                        expr.addTerms(- 2.0 * p[j], x_init_vars[j])
                    m.setObjective(expr, GRB.MINIMIZE)
                    m.optimize()

                    x_guess = [v.x for v in x_init_vars]
                    x_guess = torch.Tensor(x_guess).view_as(p)

                points[idx_point, :] = x_guess

        def propagate_through_network_and_return_difference(points):

            points_iter = points

            for layer in self.net.layers[idx_layer+1:]:
                points_iter = layer(points_iter)

            points_diff = points_iter[:, advarsarial_labels] - \
                points_iter[:, [label]]

            return points_diff

        def sample_points_inside_star(s, num_points_init):

            num_points_valid = 0
            new_points = torch.Tensor([]).view(-1, s.a0.numel())
            num_points_sample = int(1.5*num_points_init)

            while num_points_valid < num_points_init:

                eps = torch.rand((num_points_sample, num_eps))*2 - 1
                points = s.a0.repeat((num_points_sample, 1)) + eps.matmul(s.A)

                points_inside = s.submatching_single(points)
                valid_points = points[points_inside]
                new_points = torch.cat([new_points, valid_points], 0)
                num_points_valid = new_points.shape[0]

            return new_points

        isVerified = False

        num_constraints_before = 0 if s.d is None else s.d.numel()

        for idx_truncation in range(max_iterations):
            t = time()

            points = sample_points_inside_star(
                s, num_points_initialization_pgd)

            points_diff = propagate_through_network_and_return_difference(
                points)
            points_diff_max = points_diff.max(1)[0]

            indices_largest = points_diff_max.argsort(
                descending=True)[:num_points]
            points = points[indices_largest]

            points.requires_grad_(True)

            optimizer = torch.optim.SGD([points], lr=learning_rate_pgd)

            for idx_pgd in range(num_iterations_pgd):

                points_init = points.clone().detach()

                points_diff = propagate_through_network_and_return_difference(
                    points)

                points_diff_max = points_diff.max(1, keepdims=True)[0]
                biggest_violation = points_diff_max.max().item()
                isVerified = biggest_violation < 0

                if use_early_stop and not isVerified:
                    break

                threshold_offset = max_offset_advarsarial_label * \
                    max((1-idx_pgd/(num_iterations_pgd - 5)), 1E-4)
                threshold_value = points_diff_max - threshold_offset

                labels_used = points_diff >= threshold_value
                points_diff_cleared = points_diff * labels_used

                loss = - points_diff_cleared.sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    project_point_on_star(points, s)

                # self.print(points_diff_max.max().item())

            num_constraints = 0 if s.d is None else s.d.numel()
            if num_constraints == num_constraints_before:
                d_values = 'None'
            else:
                num_new_constraints = num_constraints - num_constraints_before
                num_constraints_before = num_constraints
                d_values = ', '.join(['{:.3f}'.format(x.item())
                                      for x in s.d[-num_new_constraints:]])

            self.print('{}, loss: {:.4f}, d: {}, time: {:.2f}'.format(
                idx_truncation-1, biggest_violation, d_values, time() - t), 2)

            if isVerified:
                break
            elif idx_truncation+1 < max_iterations:
                violation_indices = (points_diff_max > 0).view(-1)
                violation_points = points_init[violation_indices, :]
                num_violations = violation_points.shape[0]

                violations = [violation_points[i, :]
                              for i in range(num_violations)]

                C, d = self.derive_hyperplane_intersection(
                    s, violations, relaxation_list, idx_layer, label,
                    interpolation_weight, method='static_interpolation')

                update_lp_model(C, d)
                s.add_linear_constraints(C, d)
            else:
                pass

        return isVerified

    def apply_iterative_truncation_submatch(self, relaxation_net, relaxation_list, idx_layer, label,
                                            interpolation_weight, max_iterations, **params):

        t = time()

        relaxation_net.use_overapproximation_only = False
        relaxation_net.use_general_zonotope = False
        relaxation_net.num_solutions = 10
        relaxation_net.early_stopping_objective = None
        relaxation_net.early_stopping_bound = -1E-5
        relaxation_net.timelimit = 15*60
        relaxation_net.use_tighter_bounds = False
        relaxation_net.use_tighter_bounds_using_milp = False
        relaxation_net.use_lp = False
        relaxation_net.use_retightening = False

        template_layer, submatch_template = params['template_match']
        interpolation_weight = 1E-3

        # Backpropagate constraints
        weight = self.net.layers[idx_layer+1].weight.data
        bias = self.net.layers[idx_layer+1].bias.data
        C_new = submatch_template.C.matmul(weight)
        d_new = submatch_template.d - C_new.matmul(bias)

        s = Star(constraints=(C_new, d_new))
        relaxation = relaxation_net.relaxation_at_layers[0]
        _, _, isIntersection = s.check_constraints(relaxation, True, True)
        C_new = C_new[isIntersection, :]
        d_new = d_new[isIntersection]
        self.print('Constraints used1: {}/{}'.format(isIntersection.int().sum().item(),
                                                     isIntersection.numel()), 2)

        s.C = C_new
        s.d = d_new
        fulfilled_constraints_summary = torch.ones_like(d_new).bool()
        for z in relaxation_list:
            isFulfilled, fulfilled_constraints = \
                s.check_constraints(z, return_fulfilled_constraints=True)

            if not isFulfilled:
                try:
                    fulfilled_constraints_summary *= fulfilled_constraints
                except Exception:
                    self.print('{} {}'.format(fulfilled_constraints.shape,
                                              fulfilled_constraints_summary.shape), 3)
                    raise RuntimeError

        C_new = C_new[fulfilled_constraints_summary, :]
        d_new = d_new[fulfilled_constraints_summary]

        self.print('Constraints used2: {}/{}'.format(fulfilled_constraints_summary.int().sum().item(),
                                                     fulfilled_constraints_summary.numel()), 2)
        relaxation.add_linear_constraints(C_new, d_new)

        relaxation_net.initialize_milp_model(idx_layer)
        relaxation_net.add_input_constraints()
        for i in range(idx_layer+1, template_layer+1):
            relaxation_net.apply_layer(i)

        # Encode template match as output
        m = relaxation_net.milp_model
        m.update()
        x_out = [v for v in m.getVars()
                 if 'x_{}'.format(template_layer) in v.VarName]
        num_dimensions = len(x_out)
        violations = []

        # Constraints
        if submatch_template.d is not None:
            num_constraints = submatch_template.d.numel()
            constraints = []

            for i in range(num_constraints):
                var_name = 'constr_{}'.format(i)
                var = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY,
                               ub=GRB.INFINITY, name=var_name)
                constraints.append(var)

            for i in range(num_constraints):
                expr = gp.LinExpr()
                expr += -1*constraints[i]
                for j in range(num_dimensions):
                    expr.addTerms(submatch_template.C[i, j], x_out[j])
                expr.addConstant(-1*submatch_template.d[i])
                m.addConstr(expr, GRB.EQUAL, 0)

            violations.extend(constraints)

        # Bounds
        lower_bounds = []
        upper_bounds = []

        a0, A = submatch_template.a0_flat, submatch_template.A_flat
        A_rotated_max = A.matmul(submatch_template.U_rot).abs_().sum(0)
        a0_rotated = a0.matmul(submatch_template.U_rot).squeeze_(0)
        lower_bound_rot = (a0_rotated - A_rotated_max).numpy()
        upper_bound_rot = (a0_rotated + A_rotated_max).numpy()
        U_rot = submatch_template.U_rot.numpy()

        for i in range(num_dimensions):
            var_name = 'lb_{}'.format(i)
            var = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY,
                           ub=GRB.INFINITY, name=var_name)
            lower_bounds.append(var)

            var_name = 'ub_{}'.format(i)
            var = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY,
                           ub=GRB.INFINITY, name=var_name)
            upper_bounds.append(var)

        for i in range(num_dimensions):
            # Lower bound
            expr = gp.LinExpr()
            expr += -1*lower_bounds[i]
            for k in range(num_dimensions):
                expr.addTerms(-U_rot[k, i], x_out[k])
            expr.addConstant(lower_bound_rot[i])
            m.addConstr(expr, GRB.EQUAL, 0)

            # Upper bound
            expr = gp.LinExpr()
            expr += -1*upper_bounds[i]
            for k in range(num_dimensions):
                expr.addTerms(U_rot[k, i], x_out[k])
            expr.addConstant(-1*upper_bound_rot[i])
            m.addConstr(expr, GRB.EQUAL, 0)

        violations.extend(lower_bounds)
        violations.extend(upper_bounds)

        # Add optimization criterion
        largest_violation = m.addVar(
            lb=-GRB.INFINITY, ub=GRB.INFINITY, name='largest_violation')

        m.addConstr(largest_violation == gp.max_(violations))
        m.setObjective(largest_violation, GRB.MAXIMIZE)

        relaxation_net.optimize_milp()
        isVerified, violations = relaxation_net.evaluate_optimization()

        # self.print('constraints: {}'.format(max([v.x for v in constraints])),3)
        # self.print('lower_bounds: {}'.format(max([v.x for v in lower_bounds])),3)
        # self.print('upper_bounds: {}'.format(max([v.x for v in upper_bounds])),3)

        idx_iter = 0
        max_iterations = num_constraints
        self.print('Max iterations: {}'.format(max_iterations), 3)

        if violations is None:
            max_iterations = 0
        else:
            self.print('-1, loss: {:.4f}, d: None, time: {:.2f}'.format(
                relaxation_net.milp_model.objVal, time() - t), 2)
            if relaxation_net.milp_model.Status == 2:
                prev_objective = relaxation_net.milp_model.objVal
            else:
                prev_objective = 1E5

        while not isVerified and idx_iter < max_iterations:
            if idx_iter == 20:
                interpolation_weight = 1E-4

            C, d = self.derive_hyperplane_intersection(
                relaxation_net.relaxation_at_layers[0], violations, relaxation_list,
                idx_layer, label, interpolation_weight, method='static_interpolation')

            t = time()
            isVerified, violations = relaxation_net.rerun_with_additional_constraint(
                C, d)

            if violations is None:
                break

            objective = relaxation_net.milp_model.objVal

            d_values = ', '.join(['{:.3f}'.format(x.item()) for x in d])
            self.print('{}, loss: {:.4f}, d: {}, time: {:.2f}'.format(
                idx_iter, objective, d_values, time() - t), 2)

            # if (idx_iter > 10 and objective > 1.0) or (idx_iter > 20 and objective > 0.2):
            #     self.print('Loss too large, stop truncation', 2)
            #     break

            if relaxation_net.milp_model.Status == 2:
                if objective >= prev_objective:
                    break
                else:
                    prev_objective = objective
            idx_iter += 1

        return isVerified

    def derive_hyperplane_intersection(self, relaxation, violations, relaxation_list,
                                       idx_layer, label, interpolation_weight=0.2,
                                       method='static_interpolation'):
        # method = 'sampling_based_distance'

        def forward_pass_and_loss(point):
            for layer in self.net.layers[idx_layer+1:]:
                point = layer(point)
            # point = point.squeeze()

            false_labels = list(range(10))
            false_labels.pop(label)

            difference = point[:, false_labels] - point[:, [label]]

            loss = difference.max(1)[0]

            return loss

        center = relaxation.a0
        center = center.view(1, -1)
        C_list = []
        d_list = []

        for violation in violations:

            violation = violation.view_as(center)

            isConstraintViolated = False
            for C_test, d_test in zip(C_list, d_list):
                if (C_test * violation).sum() > d_test:
                    isConstraintViolated = True
                    break

            if isConstraintViolated:
                continue

            violation_direction = violation - center

            if method == 'static_interpolation':
                # Static weighted interpolation
                distances = [z.largest_value_in_direction(violation_direction, local=False).item()
                             for z in relaxation_list]

                d_violation = (violation_direction *
                               violation).sum()

                C = violation_direction
                d_largest = torch.Tensor([max(distances)]).view(-1)

                d = d_largest + interpolation_weight * \
                    (d_violation - d_largest)

            elif method == 'dynamic_interpolation':
                # Dynamic weighted interpolation
                largest_values = \
                    [z.largest_value_in_direction(violation_direction, local=False, return_point=True)
                     for z in relaxation_list]
                d_largest, point_largest = max(
                    largest_values, key=lambda x: x[0].item())

                violation_loss = forward_pass_and_loss(violation)
                # center_loss = forward_pass_and_loss(center)
                point_largest_loss = forward_pass_and_loss(point_largest)

                d_violation = (violation_direction *
                               violation).sum()

                C = violation_direction

                weight = -point_largest_loss / \
                    (violation_loss - point_largest_loss)
                weight *= interpolation_weight
                weight.clamp_(0, 1)

                # print(violation_loss, point_largest_loss, weight)

                d = d_largest + weight * (d_violation - d_largest)

            elif 'sampling_based' in method:
                # Not working

                num_steps = 20
                step_length = violation_direction / num_steps
                loss_prev = forward_pass_and_loss(center)
                x_prev = center
                for i in range(num_steps):
                    x = center + (i+1) * step_length
                    loss = forward_pass_and_loss(x)

                    if loss > 0:
                        break
                    else:
                        loss_prev = loss
                        x_prev = x
                x = (x * loss_prev.abs() + x_prev *
                     loss.abs()) / (loss - loss_prev)

                lower_bound, upper_bound = relaxation[0].get_bounds()
                active_neurons = (upper_bound[0, :] > lower_bound[0, :]) & (
                    violation_direction[0].abs() > 0)
                active_neurons_idx = [x.item()
                                      for x in active_neurons.nonzero()]
                num_samples = 2 * len(active_neurons_idx) + 1

                samples = x.repeat(num_samples, 1)
                step_factor = 0.10
                for idx_dim, idx_neuron_overall in enumerate(active_neurons_idx):
                    step = step_factor * \
                        violation_direction[0, idx_neuron_overall]
                    samples[idx_dim + 1, idx_neuron_overall] += step
                    samples[idx_dim + (num_samples//2) + 1,
                            idx_neuron_overall] -= step

                losses = forward_pass_and_loss(samples)

                if method == 'sampling_based_direction':
                    labels = (losses <= 0).int()
                    self.print('\n\n\n')
                    self.print('Losses:\n{}'.format(losses))

                    from sklearn import svm
                    clf = svm.SVC(kernel='linear', C=1E5)
                    clf.fit(samples.numpy(), labels.numpy())
                    C = -torch.Tensor(clf.coef_[0]).view_as(center)
                    C[:, torch.bitwise_not(active_neurons)] = 0
                    d = torch.Tensor(clf.intercept_)

                    self.print('C:\n{}'.format(C[0, active_neurons]))
                    self.print('violation_direction:\n{}'.format(
                        violation_direction[0, active_neurons]))
                    predictions = samples.matmul(C.transpose(0, 1)).view(-1)
                    self.print('prediction values:\n{}'.format(predictions))
                    self.print('d: {}'.format(d))
                    self.print('labels:\n{}'.format(labels))
                    self.print('predictions:\n{}'.format(
                        (predictions < d).int()))

                    distances = [z.largest_value_in_direction(C, local=False).item()
                                 for z in relaxation_list]
                    d_violation = (C * violation).sum()
                    d_center = (C * center).sum()

                    self.print('d: center, largest, violation, svm:\n {} / {} / {}'.format(
                               d_center, torch.Tensor([max(distances)]), d_violation, d))
                    self.print('Angle: {}'.format(
                        (violation_direction * C).sum()))

                elif method == 'sampling_based_distance':
                    C = violation_direction
                    d_violation = (C * violation).sum()

                    d_largest = max([z.largest_value_in_direction(violation_direction, local=False).item()
                                     for z in relaxation_list])

                    d_lower_bound = d_largest + 0.05 * \
                        (d_violation - d_largest)

                    d_samples = samples.matmul(C.transpose(0, 1)).view(-1)
                    d_samples_violating = d_samples[losses > 0]

                    if d_samples_violating.numel() > 0:
                        d_samples_violating_min = d_samples_violating.min()

                        d = d_samples_violating_min - 0.3 * \
                            (d_violation * d_samples_violating_min)
                        d = d.clamp_min(d_lower_bound.item())
                    else:
                        d = d_largest + 0.5 * (d_violation - d_largest)

                else:
                    self.print(
                        'Unknown method for finding hyperplane coefficients: {}'.format(method), 0)
                    raise NotImplementedError

            else:
                self.print(
                    'Unknown method for finding hyperplane coefficients: {}'.format(method), 0)
                raise NotImplementedError

            C_list.append(C)
            d_list.append(d)

        C_all = torch.cat(C_list, 0)
        d_all = torch.Tensor(d_list)
        return C_all, d_all

    def truncation_and_widening(self, z_template, relaxation_list, idx_layer, label, **params):

        if params['template_prewidening']:
            z_template = z_template.widening(
                params['widening_methods'], **params['params_widening'])

        if not isinstance(z_template, Box_Star):
            z_template = Star(z_template)

            a0 = z_template.a0_flat.clone().detach_()
            A_rotated_max = z_template.A_rotated_bounds.clone().detach_()
            U = z_template.U_rot.clone().detach_()
            U_inv = z_template.U_rot_inv.clone().detach_()
            a0_rot = a0.matmul(U)

            lower_bound = a0_rot - A_rotated_max
            upper_bound = a0_rot + A_rotated_max
            C_new = z_template.C
            d_new = z_template.d
            A_rotated_max = A_rotated_max.squeeze_()
        else:
            lower_bound = z_template.lb.clone().detach()
            upper_bound = z_template.ub.clone().detach()
            C_new = z_template.C
            d_new = z_template.d
            A_rotated_max = ((upper_bound - lower_bound) / 2.0).squeeze_()
            U = None
            U_inv = None

        template_class = z_template.__class__

        lower_bound_new = lower_bound
        upper_bound_new = upper_bound

        t = time()
        num_iterations = 20
        isVerifiedOnce = False

        for idx_outer in range(num_iterations):

            if idx_outer == 0:
                interpolation_weight = 0.1
                max_iterations = 30
                # interpolation_weight = 0.3
            else:
                interpolation_weight = 0.4 + 0.02*idx_outer
                max_iterations = 10
                # interpolation_weight = 0.6
            # interpolation_weight = 0.2

            s = z_template.__class__()
            s.init_from_bounds(
                lower_bound_new, upper_bound_new, U, U_inv, (C_new, d_new))

            s_net = Star_Net(self.net)
            s_net.relaxation_at_layers = [s]

            isVerified = self.apply_iterative_truncation(
                s_net, relaxation_list, idx_layer, label,
                interpolation_weight, max_iterations, **params)

            num_constraints = 0 if s.d is None else s.d.numel()
            try:
                objective = '{:.4f}'.format(s_net.milp_model.objVal)
            except Exception:
                objective = 'NaN'
            self.print('T&W {} {} {} {:.4f}\n'.format(
                idx_outer, num_constraints, objective, time() - t), 2)
            t = time()

            if isVerified:
                isVerifiedOnce = True
                lower_bound = lower_bound_new
                upper_bound = upper_bound_new
                C_new = s_net.relaxation_at_layers[0].C
                d_new = s_net.relaxation_at_layers[0].d

                # lower_bound_new = lower_bound_new.clone()
                # upper_bound_new = upper_bound_new.clone()

                lower_bound_new = lower_bound_new - 0.05 * A_rotated_max
                upper_bound_new = upper_bound_new + 0.05 * A_rotated_max

                # x_rot_init = [v for v in s_net.milp_model.getVars()
                #               if 'x_rot_init' in v.VarName]

                # lower_bound_new = lower_bound_new.clone()
                # upper_bound_new = upper_bound_new.clone()

                # for i in range(len(x_rot_init)):
                #     widening_step = 0.05 * A_rotated_max[i].item()
                #     if widening_step > 0:

                #         # x_rot_init[i].lb -= widening_step
                #         # x_rot_init[i].ub += widening_step

                #         lower_bound_new[0, i] -= widening_step
                #         upper_bound_new[0, i] += widening_step

                # s_net.milp_model.update()

            else:
                break

        if isVerifiedOnce:
            s = template_class()
            s.init_from_bounds(
                lower_bound, upper_bound, U, U_inv, (C_new, d_new))
            # a0_new_rot = (upper_bound + lower_bound) / 2.0
            # A_rot = (upper_bound - lower_bound) / 2.0

            # a0_new = a0_new_rot.matmul(U_inv)
            # A_new = torch.diag(A_rot.squeeze(0)).matmul(U_inv)

            # z_template = Star(a0_new.detach_(),
            #                   A_new.detach_(), (C_new, d_new))
            # z_template.U_rot = U
            # z_template.U_rot_inv = U_inv
            # z_template.A_rotated_bounds = A_rot.detach_()
            s.reshape_as(z_template)

            print_str = 'Template creation (Label:{}, Layer:{}) relaxations used: {}/{}'.format(
                label, idx_layer, len(relaxation_list), len(relaxation_list))
            print_str += ' with {} constraints'.format(
                0 if s.d is None else s.d.numel())
            self.print(print_str, 1)

            return [s], []
        else:
            print_str = 'Template creation (Label:{}, Layer:{}) Failed using {} relaxations'.format(
                label, idx_layer, len(relaxation_list))
            self.print(print_str, 1)
            return [], relaxation_list

    def template_reunion(self, **params):
        disable_tqdm = self.verbosity < 2
        for label in tqdm(self.verification_templates.keys(), disable=disable_tqdm):
            for idx_layer in self.verification_templates[label].keys():

                templates = self.verification_templates[label][idx_layer]

                if len(templates) == 0:
                    continue

                params['params_union']['U'] = templates[0].U_rot
                params['params_union']['U_inv'] = templates[0].U_rot_inv

                for idx_template0 in range(len(templates)):
                    z0 = templates[idx_template0]
                    if z0 is None:
                        continue

                    for idx_template1 in range(idx_template0+1, len(templates)):
                        z1 = templates[idx_template1]
                        if z1 is None:
                            continue

                        z_union = z0.union(z1, params['union_method'],
                                           **params['params_union'])
                        isVerified, _, z_union = self.propagate_relaxation(
                            z_union, label, idx_layer, **params)

                        if isVerified:
                            templates[idx_template0] = z_union
                            templates[idx_template1] = None
                            self.print('Union templates (Label:{}, Layer:{}): {} & {}'.format(
                                label, idx_layer, idx_template0, idx_template1), 2)

                self.verification_templates[label][idx_layer] = list(
                    filter(None, templates))

                # self.verification_counts[idx_layer][label] = \
                #     [] * len(self.verification_templates[label][idx_layer])

    def widen_templates(self, methods, **params):

        if len(methods) == 0:
            return

        self.print('Widen templates:', 1)

        disable_tqdm = self.verbosity < 2

        for label in tqdm(self.verification_templates.keys(), disable=disable_tqdm):
            for idx_layer in self.verification_templates[label].keys():
                for idx_template in range(len(self.verification_templates[label][idx_layer])):

                    self.verification_templates[label][idx_layer][idx_template] = \
                        self.apply_template_widening(
                            self.verification_templates[label][idx_layer][idx_template],
                            label, idx_layer, methods, **params)

    def apply_template_widening(self, template, label, idx_layer, methods, **params):

        # Simple widening
        new_template = template.widening(
            methods, **params['params_widening'])

        if isinstance(template, Star):
            new_template.C = template.C
            new_template.d = template.d

        isVerified, _, new_template = self.propagate_relaxation(
            new_template, label, idx_layer, **params)

        if isVerified:
            template = new_template

        # Gradient based widening
        if 'gradient_based' in methods:

            template = self.apply_gradient_based_widening(
                template, label, idx_layer, **params)

        if 'gradient_based2' in methods:

            template = self.apply_gradient_based_widening2(
                template, label, idx_layer, **params)

        if 'milp_based' in methods:
            template = self.apply_milp_based_widening(
                template, label, idx_layer, **params)

        return template

    def apply_gradient_based_widening(self, template, label, idx_layer, **params):
        a0 = template.a0_flat.clone().detach_()
        A_rotated_max = template.A_rotated_bounds.clone().detach_()
        U = template.U_rot.clone().detach_()
        U_inv = template.U_rot_inv.clone().detach_()
        a0_rot = a0.matmul(U)

        lower_bound_prev = a0_rot - A_rotated_max
        upper_bound_prev = a0_rot + A_rotated_max

        lower_bound_add_prev = (params['params_widening']['stretch_factor']
                                - 1.0) / 4 * A_rotated_max
        upper_bound_add_prev = lower_bound_add_prev.clone()

        for idx_outer in range(50):

            lower_bound_add = lower_bound_add_prev * 2.0
            lower_bound_add.clamp_min_(1E-2)
            lower_bound_add.requires_grad_(True)

            upper_bound_add = upper_bound_add_prev * 2.0
            upper_bound_add.clamp_min_(1E-2)
            upper_bound_add.requires_grad_(True)

            optimizer = torch.optim.Adam(
                [lower_bound_add, upper_bound_add], lr=0.005)

            isVerified = False
            num_iterations = 200
            for idx_inner in range(num_iterations):

                z_net = params['relaxation_net_type'](self.net, self.device)

                lower_bound_new = lower_bound_prev - lower_bound_add
                upper_bound_new = upper_bound_prev + upper_bound_add

                a0_new_rot = (upper_bound_new + lower_bound_new) / 2.0
                A_rot = (upper_bound_new - lower_bound_new) / 2.0

                a0_new = a0_new_rot.matmul(U_inv)
                A_new = torch.diag(A_rot.squeeze(0)).matmul(U_inv)

                z_new = z_net.relaxation_type(a0_new, A_new)
                z_new.A_rotated_bounds = A_rot.detach_()
                z_new.reshape_as(template)
                z_net.relaxation_at_layers.append(z_new)

                z_net.process_from_layer(label, start_layer=idx_layer+1)

                loss = torch.sum(torch.nn.functional.relu(
                    z_net.y - z_net.y[0, label]))

                # print(idx_optim, loss.item(), scale.min(), scale.max())
                scale_lb = (lower_bound_add / A_rotated_max).detach()
                scale_ub = (upper_bound_add / A_rotated_max).detach()
                if loss <= 0:

                    self.print('{} {} {}, vol: {:.4f}, scale: {:.4f}/{:.4f}'.format(
                        idx_outer, idx_inner, idx_layer, z_new.get_approximate_volume(),
                        scale_lb.max().item(), scale_ub.max().item()), 3)

                    isVerified = True
                    break

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    lower_bound_add.clamp_min_(0)
                    upper_bound_add.clamp_min_(0)

                    if idx_inner > (num_iterations // 4) and (idx_inner % 20) == 0:
                        lower_bound_add[:10] = lower_bound_add[:10] / 2
                        upper_bound_add[:10] = upper_bound_add[:10] / 2

            if isVerified:
                A_rotated_max = A_rot.detach()
                lower_bound_prev = lower_bound_new.detach()
                upper_bound_prev = upper_bound_new.detach()
                lower_bound_add_prev = lower_bound_add.detach()
                upper_bound_add_prev = upper_bound_add.detach()

                if lower_bound_add.max() > 0 or lower_bound_add.max() > 0:
                    continue
                else:
                    break
            else:
                self.print('{} {} {}, vol: {:.4f}, scale: {:.4f}/{:.4f}, loss: {:.4f}'.format(
                    idx_outer, idx_inner, idx_layer, z_new.get_approximate_volume(),
                    scale_lb.max().item(), scale_ub.max().item(), loss.item()), 3)
                break

        a0_new_rot = (upper_bound_prev + lower_bound_prev) / 2.0
        A_rot = (upper_bound_prev - lower_bound_prev) / 2.0

        a0_new = a0_new_rot.matmul(U_inv)
        A_new = torch.diag(A_rot.squeeze(0)).matmul(U_inv)

        z_new.a0 = a0_new.detach()
        z_new.A = A_new.detach()
        z_new.U_rot = U
        z_new.U_rot_inv = U_inv
        z_new.A_rotated_bounds = A_rot.detach()
        z_new.reshape_as(template)

        self.print('Widening (Label:{}, Layer:{}) iterations: {}, volume: {:.0f}'.format(
            label, idx_layer, idx_outer, z_new.get_approximate_volume()), 2)

        return z_new

    def apply_gradient_based_widening2(self, template, label, idx_layer, num_iterations=50, **params):
        a0 = template.a0_flat.clone().detach_()
        A_rotated_max = template.A_rotated_bounds.clone().detach_()
        U = template.U_rot.clone().detach_()
        U_inv = template.U_rot_inv.clone().detach_()
        a0_rot = a0.matmul(U)

        lower_bound = a0_rot - A_rotated_max
        upper_bound = a0_rot + A_rotated_max

        lower_bound_verified = lower_bound.clone()
        upper_bound_verified = upper_bound.clone()
        loss_optimal = 1E10

        lower_bound_add = torch.ones_like(lower_bound) * 1E-2
        upper_bound_add = torch.ones_like(lower_bound) * 1E-2

        lower_bound_add.requires_grad_(True)
        upper_bound_add.requires_grad_(True)

        # optimizer = torch.optim.SGD(
        #     [lower_bound_add, upper_bound_add], lr=0.005, momentum=0.2)
        optimizer = torch.optim.Adam(
            [lower_bound_add, upper_bound_add], lr=0.005)

        idx_outer = -1
        for idx_outer in range(num_iterations):

            lower_bound_new = lower_bound - lower_bound_add
            upper_bound_new = upper_bound + upper_bound_add

            z_net = params['relaxation_net_type'](self.net, self.device)

            a0_new_rot = (upper_bound_new + lower_bound_new) / 2.0
            A_rot = (upper_bound_new - lower_bound_new) / 2.0

            a0_new = a0_new_rot.matmul(U_inv)
            A_new = torch.diag(A_rot.squeeze(0)).matmul(U_inv)

            z_new = z_net.relaxation_type(a0_new, A_new)
            z_new.A_rotated_bounds = A_rot.detach_()
            z_new.reshape_as(template)
            z_net.relaxation_at_layers.append(z_new)

            isVerified = z_net.process_from_layer(
                label, start_layer=idx_layer+1)

            verification_loss = z_net.get_verification_loss(label)
            widening_loss = lower_bound_add.sum() + upper_bound_add.sum()

            loss = verification_loss * 1E3 - widening_loss

            if isVerified and loss < loss_optimal:
                lower_bound_verified = lower_bound_new.detach()
                upper_bound_verified = upper_bound_new.detach()
                loss_optimal = loss.item()

            print_str = 'Verification loss: {:.3f}, Widening loss: {:.3f}, Total loss: {:.3f}'.format(
                verification_loss.item(), widening_loss.item(), loss.item())

            print_str += ', Volume: {:.0f}'.format(
                z_new.get_approximate_volume())
            self.print(print_str, 3)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                lower_bound_add.clamp_min_(0)
                upper_bound_add.clamp_min_(0)

        a0_new_rot = (upper_bound_verified + lower_bound_verified) / 2.0
        A_rot = (upper_bound_verified - lower_bound_verified) / 2.0

        a0_new = a0_new_rot.matmul(U_inv)
        A_new = torch.diag(A_rot.squeeze(0)).matmul(U_inv)

        z_new = Zonotope()
        z_new.a0 = a0_new.detach()
        z_new.A = A_new.detach()
        z_new.U_rot = U
        z_new.U_rot_inv = U_inv
        z_new.A_rotated_bounds = A_rot.detach()
        z_new.reshape_as(template)

        self.print('Widening (Label:{}, Layer:{}) iterations: {}, volume: {:.0f}'.format(
            label, idx_layer, idx_outer+1, z_new.get_approximate_volume()), 2)

        return z_new

    def template_verification_with_milp(self, relaxation, label, idx_layer):

        if isinstance(relaxation, Star):
            new_relaxation = relaxation
        else:
            new_relaxation = Star(relaxation)

        relaxation_net = Star_Net(self.net)
        relaxation_net.relaxation_at_layers = [new_relaxation]
        relaxation_net.use_overapproximation_only = False
        relaxation_net.use_general_zonotope = False
        # relaxation_net.early_stopping_objective = 1E-5
        # relaxation_net.early_stopping_bound = -1E-5
        relaxation_net.use_tighter_bounds = True
        relaxation_net.use_tighter_bounds_using_milp = True
        relaxation_net.timelimit = 900

        if idx_layer in self.milp_models.keys():
            milp_model = self.milp_models[idx_layer]
        else:
            milp_model = None
        # print(milp_model, self.milp_models)

        isVerified, violation = relaxation_net.process_from_layer(
            label, idx_layer + 1, return_violation=True, milp_model=milp_model)

        loss = relaxation_net.milp_model.objVal

        return isVerified, loss

    def ctg_convert_dataset_to_relaxation_list(self, dataset, selected_label,
                                               layer_indices=None, **params):

        self.net.eval()
        torch.manual_seed(12)
        if layer_indices is None:
            layer_indices = self.layer_indices_from_type(
                params['selected_layers'], params['num_skip_layers'], params['exclude_last_layer'])
        self.reset_counts(layer_indices)

        relaxation_lists = {}
        for idx_layer in layer_indices:
            relaxation_lists[idx_layer] = []

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                  shuffle=False, num_workers=0,
                                                  collate_fn=utils.custom_collate)

        for relaxations, label, isPredicted, isVerified in tqdm(data_loader):

            if label == selected_label and isVerified:

                for idx_layer in layer_indices:
                    z = relaxations[idx_layer]

                    relaxation_lists[idx_layer].append(z)

        self.print(
            'Used relaxations: {}'.format(len(relaxation_lists[layer_indices[0]])), 1)

        return relaxation_lists

    def complete_template_generation(self, dataset, selected_label, **params):

        self.print('Generate complete templates:', 1)

        relaxation_lists = self.ctg_convert_dataset_to_relaxation_list(
            dataset, selected_label, **params)

        for idx_layer in relaxation_lists.keys():
            self.apply_complete_template_generation(
                relaxation_lists[idx_layer], idx_layer, selected_label, **params)

    def ctg_cluster_with_cse(self, relaxation_list, idx_layer, label, **params):

        dissimilarity = self.get_dissimilarity_matrix(
            relaxation_list, idx_layer, label, **params)

        embeddings = self.generate_constant_shift_embeddings(
            dissimilarity, **params)

        cluster_assignments = self.ctg_cluster_with_coordinates(
            embeddings, **params)

        return cluster_assignments

    def ctg_cluster_with_coordinates(self, embeddings, num_clusters=None, **params):
        from sklearn import cluster, mixture
        import numpy as np

        embeddings = embeddings.numpy().astype(np.float64)
        np.random.seed(17)
        if num_clusters is None:
            num_clusters = embeddings.shape[0] // params['initial_cluster_size']

        if params['cluster_method'] == 'kmeans':
            alg = cluster.KMeans(num_clusters, init='k-means++')
            cluster_assignments = alg.fit_predict(embeddings)
            # centroid = torch.Tensor(alg.cluster_centers_)

        elif params['cluster_method'] == 'gaussianmixture':
            try:
                alg = mixture.GaussianMixture(num_clusters)
                cluster_assignments = alg.fit_predict(embeddings)
                # centroid = torch.Tensor(alg.means_)
            except Exception:
                self.print('Skip to kmeans')
                alg = cluster.KMeans(num_clusters, init='k-means++')
                cluster_assignments = alg.fit_predict(embeddings)
                # centroid = torch.Tensor(alg.cluster_centers_)

        return cluster_assignments

    def ctg_create_clusters(self, cluster_assignments, relaxation_list, **params):

        t = time()

        cluster_assignments = list(cluster_assignments)
        assert(len(relaxation_list) == len(cluster_assignments))

        true_num_clusters = len(set(cluster_assignments)) - \
            (1 if -1 in cluster_assignments else 0)

        cluster_set = {}
        indices_multiple = [i for i in range(true_num_clusters)
                            if cluster_assignments.count(i) > 1]
        num_singles = true_num_clusters - len(indices_multiple)

        for idx_cluster in indices_multiple:
            indices_assigned_relaxations = [i for i in range(len(relaxation_list))
                                            if cluster_assignments[i] == idx_cluster]

            assigned_relaxations = [relaxation_list[i]
                                    for i in indices_assigned_relaxations]

            num_relaxations = len(assigned_relaxations)

            cluster_information = {}
            cluster_information['relaxations'] = assigned_relaxations
            cluster_information['indices'] = indices_assigned_relaxations
            cluster_information['num_relaxations'] = num_relaxations

            cluster_set[idx_cluster] = cluster_information

        unused_indices = [i for i in range(len(cluster_assignments))
                          if cluster_assignments[i] not in indices_multiple]

        unused_relaxations = [relaxation_list[i] for i in unused_indices]

        unused_cluster_information = {}
        unused_cluster_information['relaxations'] = unused_relaxations
        unused_cluster_information['indices'] = unused_indices
        unused_cluster_information['num_relaxations'] = len(unused_indices)
        unused_cluster_information['union'] = None

        cluster_set[-1] = unused_cluster_information

        self.print('Clusters generated: Singles/Total: {}/{}, time used: {:.2f}'.format(
            num_singles, true_num_clusters, time() - t), 1)

        return cluster_set

    def ctg_verify_clusters(self, cluster_set, idx_layer, label, **params):

        t = time()

        cluster_indices = [i for i in cluster_set.keys() if i > -1]

        unused_cluster = cluster_set[-1]
        num_verified_clusters = 0

        for idx_iter, idx_cluster in enumerate(cluster_indices):

            c = cluster_set[idx_cluster]

            relaxations = c['relaxations']

            z_union = relaxations[0].union(
                relaxations[1:], params['union_method'], **params['params_union'])

            template_list, _ = self.iterative_truncation(
                z_union, relaxations, idx_layer, label, **params)

            # isVerified, _ = self.template_verification_with_milp(
            #     z_union, label, idx_layer)

            # if isVerified:
            #     template_list = [z_union]
            # else:
            #     template_list = []

            if len(template_list) > 0:
                c['union'] = template_list[0]
                num_verified_clusters += 1

                self.print('Verified cluster {} of {} with {} relaxations and {} constraints'.format(
                    idx_iter, len(cluster_indices), c['num_relaxations'],
                    0 if c['union'].d is None else c['union'].d.numel()), 2)

            else:
                c['union'] = None

                unused_cluster['relaxations'].extend(relaxations)
                unused_cluster['indices'].extend(c['indices'])
                unused_cluster['num_relaxations'] += c['num_relaxations']

        self.print('Cluster unions verified: {}/{}, time used: {:.2f}'.format(
            num_verified_clusters, len(cluster_indices), time() - t), 1)
        return cluster_set

    def ctg_split_larger_not_verified_clusters(self, cluster_set, dissimilarity, idx_layer,
                                               label, **params):

        num_relaxations = dissimilarity.shape[0]

        unused_cluster_indices = [i for i in cluster_set.keys()
                                  if cluster_set[i]['union'] is None]

        new_unused_relaxations = []

        highest_cluster_index = max(cluster_set.keys())

        num_verified_clusters_before = len(
            cluster_set) - 1 - len(unused_cluster_indices)
        num_verified_clusters_after = num_verified_clusters_before + 0
        num_not_verified_clusters_before = len(unused_cluster_indices)
        num_not_verified_clusters_after = num_not_verified_clusters_before + 0
        images_includes_before = sum(
            [c['num_relaxations'] for c in cluster_set.values() if c['union'] is not None])

        # try to split larger clusters
        for idx_cluster in unused_cluster_indices:
            cluster = cluster_set[idx_cluster]
            cluster_relaxations = cluster['relaxations']

            if cluster['num_relaxations'] <= params['initial_cluster_size'] // 2:
                new_unused_relaxations.extend(
                    zip(cluster['indices'], cluster['relaxations']))
                num_not_verified_clusters_after += 1
                continue

            cluster_indices = cluster['indices']
            dissimilarity_cluster = dissimilarity[cluster_indices, :]
            dissimilarity_cluster = dissimilarity_cluster[:, cluster_indices]

            embeddings = self.generate_constant_shift_embeddings(
                dissimilarity_cluster, **params)

            num_subclusters = 3

            cluster_assignments = self.ctg_cluster_with_coordinates(
                embeddings, num_clusters=num_subclusters, **params)

            for cluster_label in range(num_subclusters):

                indices_assigned_relaxations = [i for i in range(len(cluster_relaxations))
                                                if cluster_assignments[i] == idx_cluster]
                assigned_relaxations = [cluster_relaxations[i]
                                        for i in indices_assigned_relaxations]
                if len(assigned_relaxations) > 1:
                    z_union = assigned_relaxations[0].union(
                        assigned_relaxations[1:], params['union_method'], **params['params_union'])

                    template_list, _ = self.iterative_truncation(
                        z_union, assigned_relaxations, idx_layer, label, **params)

                    isVerified = len(template_list) > 0
                else:
                    isVerified = False
                indices_overall = [cluster_indices[i]
                                   for i in indices_assigned_relaxations]
                # print(idx_cluster, isVerified, len(indices_overall))

                if isVerified:
                    cluster_information = {}
                    cluster_information['relaxations'] = assigned_relaxations
                    cluster_information['indices'] = indices_overall
                    cluster_information['num_relaxations'] = len(
                        indices_overall)
                    cluster_information['union'] = template_list[0]

                    highest_cluster_index += 1
                    cluster_set[highest_cluster_index] = cluster_information

                    num_verified_clusters_after += 1

                else:
                    new_unused_relaxations.extend(
                        zip(indices_overall, assigned_relaxations))
                    num_not_verified_clusters_after += 1

        images_includes_after = sum(
            [c['num_relaxations'] for c in cluster_set.values() if c['union'] is not None])
        self.print('Num verified clusters before/after splitting: {}/{}'.format(
            num_verified_clusters_before, num_verified_clusters_after), 1)
        self.print('Num not-verified clusters before/after splitting: {}/{}'.format(
            num_not_verified_clusters_before, num_not_verified_clusters_after), 1)
        self.print('Images included in templates before/after splitting: {}/{} out of {}'.format(
            images_includes_before, images_includes_after, num_relaxations), 1)

        unused_cluster_information = cluster_set[-1]
        unused_indices, unused_relaxations = list(zip(*new_unused_relaxations))
        unused_cluster_information['relaxations'] = list(unused_relaxations)
        unused_cluster_information['indices'] = list(unused_indices)
        unused_cluster_information['num_relaxations'] = len(unused_indices)

        return cluster_set

    def ctg_add_single_relaxations_to_cluster(self, cluster_set, dissimilarity, idx_layer,
                                              label, **params):
        num_relaxations = dissimilarity.shape[0]
        images_includes_before = sum(
            [c['num_relaxations'] for c in cluster_set.values() if c['union'] is not None])

        unused_relaxations = cluster_set[-1]['relaxations']
        new_unused_relaxations = []

        self.print('Num relaxations sum: {}'.format(sum(
            [len(c['relaxations']) for c in cluster_set.values() if c['union'] is not None])), 1)
        self.print('Num unused relaxations sum: {}'.format(sum(
            [len(c['relaxations']) for (idx, c) in cluster_set.items()
             if (c['union'] is None and idx > -1)]), 1))

        self.print('Num unused relaxations: {}'.format(
            len(unused_relaxations)), 1)

        t = time()

        # cluster_assignment_lookup = [-1] * num_relaxations
        cluster_centers = []
        cluster_rotated_bounds = []
        cluster_lookup_indices = []

        idx_cluster = 0
        for idx_cluster, cluster in cluster_set.items():
            if cluster['union'] is not None:
                # for idx_relaxation in cluster['indices']:
                #     cluster_assignment_lookup[idx_relaxation] = idx_cluster

                cluster_lookup_indices.append(idx_cluster)
                idx_cluster += 1

                z = cluster['union']
                cluster_centers.append(z.a0)
                a0_rot = z.a0.matmul(z.U_rot)
                A_rot = z.A.matmul(z.U_rot).abs().sum(0, keepdims=True)

                lower_bound = a0_rot - A_rot
                upper_bound = a0_rot + A_rot

                cluster_rotated_bounds.append((lower_bound, upper_bound))

        # cluster_center_list = [(idx, c['union'].a0) for idx, c in cluster_set.items()
        #                        if c['union'] is not None]
        cluster_centers = torch.cat(cluster_centers, 0)

        unused_centers = [z.a0 for z in unused_relaxations]
        unused_centers = torch.cat(unused_centers, 0)

        def l2_distance(x, y):
            x_norm = (x**2).sum(1).view(-1, 1)
            y_norm = (y**2).sum(1).view(1, -1)
            dist = x_norm + y_norm - 2.0 * \
                torch.mm(x, torch.transpose(y, 0, 1))

            return dist

        indices_closest = l2_distance(
            unused_centers, cluster_centers).argsort(1)

        # try to append single relaxations to other clusters
        for idx_relaxation, relaxation in enumerate(unused_relaxations):
            idx_relaxation_overall = cluster_set[-1]['indices'][idx_relaxation]
            idx_closest_clusters = indices_closest[idx_relaxation, :8].tolist()

            self.print('Try adding relaxation {} ({}) to clusters {}'.format(
                idx_relaxation, idx_relaxation_overall, str(idx_closest_clusters)), 1)

            for idx_cluster in idx_closest_clusters:
                idx_cluster_overall = cluster_lookup_indices[idx_cluster]
                cluster = cluster_set[idx_cluster_overall]

                template = cluster['union']
                lower_bound, upper_bound = cluster_rotated_bounds[idx_cluster]

                if isinstance(template, Star):
                    if not template.check_constraints(relaxation):
                        continue

                a0_rot = relaxation.a0.matmul(template.U_rot)
                A_rot = relaxation.A.matmul(
                    template.U_rot).abs().sum(0, keepdims=True)

                lower_bound_new = torch.min(lower_bound, a0_rot - A_rot)
                upper_bound_new = torch.max(upper_bound, a0_rot + A_rot)

                if lower_bound_new.equal(lower_bound) and upper_bound_new.equal(upper_bound):
                    isVerified = True
                    template_new = template
                else:
                    if isinstance(template, Star):
                        constraints = (template.C, template.d)
                    else:
                        constraints = None
                    s = Star()
                    s.init_from_bounds(lower_bound_new, upper_bound_new,
                                       template.U_rot, template.U_rot_inv, constraints)

                    apply_iterative_truncation = True
                    if apply_iterative_truncation:
                        template_list, _ = self.iterative_truncation(
                            s, cluster['relaxations'] + [relaxation],
                            idx_layer, label, max_iterations=5, **params)
                        if len(template_list) > 0:
                            isVerified = True
                            template_new = template_list[0]
                        else:
                            isVerified = False
                    else:
                        isVerified, _ = self.template_verification_with_milp(
                            s, label, idx_layer)

                        if isVerified:
                            template_new = s

                if isVerified:
                    cluster['union'] = template_new
                    cluster['relaxations'] += [relaxation]
                    cluster['indices'] += [idx_relaxation_overall]
                    cluster['num_relaxations'] += 1
                    self.print('Relaxation {} added to cluster {} ({})'.format(
                        idx_relaxation_overall, idx_cluster, idx_cluster_overall), 2)
                    break
                else:
                    new_unused_relaxations.append(
                        (idx_relaxation_overall, relaxation))

        images_includes_after = sum(
            [c['num_relaxations'] for c in cluster_set.values() if c['union'] is not None])
        self.print('Images included in templates after adding: {}/{} out of {}'.format(
            images_includes_before, images_includes_after, num_relaxations), 1)

        self.print('Time used: {:.2f}'.format(time() - t), 1)

        unused_cluster_information = cluster_set[-1]
        unused_indices, unused_relaxations = list(zip(*new_unused_relaxations))
        unused_cluster_information['relaxations'] = list(unused_relaxations)
        unused_cluster_information['indices'] = list(unused_indices)
        unused_cluster_information['num_relaxations'] = len(unused_indices)

        return cluster_set

    def ctg_merge_clusters(self, cluster_set, idx_layer, label, **params):

        path = 'cluster_set_temp.pkl'

        cluster_centers = [c['union'].a0 for idx, c in cluster_set.items()
                           if c['union'] is not None]
        cluster_centers = torch.cat(cluster_centers, 0)

        cluster_lookup_indices = [idx for idx, c in cluster_set.items()
                                  if c['union'] is not None]

        def check_constraints_violations(template, relaxations,
                                         print_str=None):
            # num_constraints = 0 if template.C is None else template.d.numel()
            num_violations = 0
            # z_widened = z_union.widening(
            #     ['add_min_value'], **{'min_value_all': 1E-6})

            for z in relaxations:
                if not template.check_constraints(z):
                    num_violations += 1
            self.print('Num constraints before/inter/after: {}, num violations: {}/{}'.format(
                print_str, num_violations, len(relaxations)), 2)

        def l2_distance(x, y):
            x_norm = (x**2).sum(1).view(-1, 1)
            y_norm = (y**2).sum(1).view(1, -1)
            dist = x_norm + y_norm - 2.0 * \
                torch.mm(x, torch.transpose(y, 0, 1))

            return dist

        max_distance = 1E4

        distance_matrix = l2_distance(cluster_centers, cluster_centers)
        distance_matrix = distance_matrix + \
            torch.eye(distance_matrix.shape[0])*max_distance

        def indices_smallest(matrix):
            min_dim0, argmins_dim0 = matrix.min(0)
            argmin_dim1 = min_dim0.argmin().item()
            argmin_dim0 = argmins_dim0[argmin_dim1].item()
            return argmin_dim0, argmin_dim1

        num_clusters = distance_matrix[0, :].numel()

        while distance_matrix.min().min() < max_distance:
            num_combinations = (
                distance_matrix < max_distance).int().sum().item() // 2
            minimal_length = distance_matrix.min().min().item()
            self.print('Num clusters left: {}, num_combinations: {}, minimal distance: {:.4f}'.format(
                num_clusters, num_combinations, minimal_length), 1)

            idx_cluster, idy_cluster = indices_smallest(distance_matrix)
            idx_cluster_overall = cluster_lookup_indices[idx_cluster]
            idy_cluster_overall = cluster_lookup_indices[idy_cluster]

            cluster_x = cluster_set[idx_cluster_overall]
            cluster_y = cluster_set[idy_cluster_overall]

            constraints_list = []
            num_constraints_before = 0

            # Check linear constraints of one union with relaxations of other cluster
            for union, relaxations in [(cluster_x['union'], cluster_y['relaxations']),
                                       (cluster_y['union'], cluster_x['relaxations'])]:
                if not isinstance(union, Star):
                    continue

                if union.C is None:
                    continue

                fulfilled_constraints_summary = torch.ones_like(union.d).bool()
                num_constraints_before += union.d.numel()

                for z in relaxations:
                    isFulfilled, fulfilled_constraints = \
                        union.check_constraints(
                            z, return_fulfilled_constraints=True)

                    if not isFulfilled:
                        fulfilled_constraints_summary *= fulfilled_constraints

                constraints_list.append(
                    (union.C[fulfilled_constraints_summary, :], union.d[fulfilled_constraints_summary]))

            # Create huge union, add linear constraints and then try to verify
            all_relaxations = cluster_x['relaxations'] + \
                cluster_y['relaxations']

            z_union = all_relaxations[0].union(
                all_relaxations[1:], params['union_method'], **params['params_union'])

            if not isinstance(union, Box_Star):
                z_union = Star(z_union)
            for C, d in constraints_list:
                z_union.add_linear_constraints(C, d)

            num_constraints_intermediate = 0 if z_union.C is None else z_union.d.numel()

            # check_constraints_violations(z_union, all_relaxations)

            apply_iterative_truncation = True
            num_iterations = 50
            if apply_iterative_truncation:
                template_list, _ = self.iterative_truncation(
                    z_union, all_relaxations, idx_layer, label, num_iterations, **params)
                if len(template_list) > 0:
                    isVerified = True
                    z_union = template_list[0]
                else:
                    isVerified = False
            else:
                isVerified, _ = self.template_verification_with_milp(
                    z_union, label, idx_layer)

            if isVerified:
                num_constraints_after = 0 if z_union.C is None else z_union.d.numel()
                print_str = '{}/{}/{}'.format(num_constraints_before,
                                              num_constraints_intermediate, num_constraints_after)
                check_constraints_violations(
                    z_union, all_relaxations, print_str)

                cluster_x['union'] = z_union
                cluster_x['relaxations'] = all_relaxations
                cluster_x['indices'] += cluster_y['indices']
                cluster_x['num_relaxations'] += cluster_y['num_relaxations']

                cluster_y['union'] = None

                distance_x = distance_matrix[idx_cluster, :]
                distance_y = distance_matrix[idy_cluster, :]

                distance_new = torch.min(distance_x, distance_y)
                distance_new[distance_x == max_distance] = max_distance
                distance_new[distance_y == max_distance] = max_distance

                distance_matrix[idx_cluster, :] = distance_new
                distance_matrix[:, idx_cluster] = distance_new
                distance_matrix[idy_cluster, :] = max_distance
                distance_matrix[:, idy_cluster] = max_distance

                num_constraints = 0 if z_union.C is None else z_union.d.numel()
                self.print('Cluster merge: Merged clusters {} and {} with {} constraints'.format(
                    idx_cluster_overall, idy_cluster_overall, num_constraints), 1)

                num_clusters -= 1

            else:
                distance_matrix[idx_cluster, idy_cluster] = max_distance
                distance_matrix[idy_cluster, idx_cluster] = max_distance

            pickle.dump((cluster_set, distance_matrix,
                         cluster_lookup_indices), open(path, 'wb'))
        return cluster_set

    def ctg_widen_clusters(self, cluster_set, idx_layer, label, **params):
        clusters = [(idx, c)
                    for idx, c in cluster_set.items() if c['union'] is not None]

        for idx_cluster, cluster in clusters:
            template_list, _ = self.truncation_and_widening(
                cluster['union'], cluster['relaxations'], idx_layer, label, **params)

            if len(template_list) > 0:
                cluster['union'] = template_list[0]

        return cluster_set

    def ctg_truncate_clusters(self, cluster_set, idx_layer, label, **params):

        t = time()

        cluster_lookup_indices = [idx for idx, c in cluster_set.items()
                                  if c['union'] is not None]
        num_clusters = len(cluster_lookup_indices)
        num_verified_clusters = 0

        for idx_iter, idx_cluster in enumerate(cluster_lookup_indices):

            c = cluster_set[idx_cluster]
            relaxations = c['relaxations']
            z = c['union']

            num_constraints = 0 if z.d is None else z.d.numel()

            print_str = 'Verification of cluster {} starts'.format(idx_iter)
            print_str += '\nThis cluster features {} relaxations and {} constraints'.format(
                len(relaxations), num_constraints)
            self.print(print_str, 2)

            template_list, _ = self.iterative_truncation(
                z, relaxations, idx_layer, label, **params)

            if len(template_list) > 0:
                c['union'] = template_list[0]
                num_verified_clusters += 1

                self.print('Verified cluster {} of {} with {} relaxations and {} constraints'.format(
                    idx_iter, num_clusters, c['num_relaxations'],
                    0 if c['union'].d is None else c['union'].d.numel()), 2)

            else:
                c['union'] = None
                self.print('Verification of cluster {} failed'.format(
                    idx_iter), 2)

        self.print('Cluster unions verified: {}/{}, time used: {:.2f}'.format(
            num_verified_clusters, num_clusters, time() - t), 1)
        return cluster_set

    def ctg_backpropagate_clusters(self, cluster_set, idx_layer, template_layer, label, **params):

        clusters = [c for c in cluster_set.values()
                    if c['union'] is not None]

        self.print('Backpropagate {} cluster for label {} form layer {} to layer {}'.format(
            len(clusters), label, template_layer, idx_layer), 1)

        for c in clusters:

            pass

    def apply_complete_template_generation(self, relaxation_list, idx_layer, label, **params):

        # Option CSE
        cluster_assignments = self.ctg_cluster_with_cse(
            relaxation_list, idx_layer, label, **params)

        # # Option direct clustering
        # num_initial_clusters = len(
        #     relaxation_list) // params['initial_cluster_size']
        # params['num_templates_per_class'] = num_initial_clusters
        # _, cluster_assignments = self.cluster_relaxations(
        #     relaxation_list, idx_layer, **params)

        cluster_set = self.ctg_create_cluster_union_and_verify(
            relaxation_list, cluster_assignments, idx_layer, label, **params)

        return cluster_set

    def generate_constant_shift_embeddings(self, dissimilarity, p=20, **params):

        num_relaxations = dissimilarity.shape[0]
        Q = torch.eye(num_relaxations) - torch.ones((num_relaxations,
                                                     num_relaxations)) / num_relaxations
        Dc = Q.matmul(dissimilarity).matmul(Q)
        Sc = -Dc / 2
        eigenvalues, V = torch.eig(Sc, eigenvectors=True)
        eigenvalues = eigenvalues.norm(dim=1)
        eigenvalues = eigenvalues - eigenvalues.min()
        indices_highest_eigenvalues = eigenvalues.argsort(descending=True)[:p]

        Dp_half = torch.diag(eigenvalues[indices_highest_eigenvalues].sqrt())
        Vp = V[:, indices_highest_eigenvalues]

        X = Vp.matmul(Dp_half)

        return X

    def get_dissimilarity_matrix(self, relaxation_list, idx_layer, label, **params):

        # import itertools
        num_relaxations = len(relaxation_list)

        # get list of centers
        lower_bound, upper_bound = relaxation_list[0].get_bounds()
        for z in relaxation_list[1:]:
            bounds = z.get_bounds()

            lower_bound = torch.min(lower_bound, bounds[0])
            upper_bound = torch.max(upper_bound, bounds[1])

        active_neurons = upper_bound[0, :] > lower_bound[0, :]

        a0_list = [z.a0_flat[:, active_neurons]
                   for z in relaxation_list]
        a0_cat = torch.cat(a0_list, 0)

        # get l2-distance
        a0_norm = (a0_cat**2).sum(1).view(-1, 1)
        a0_norm_t = a0_norm.view(1, -1)
        l2_distances = a0_norm + a0_norm_t \
            - 2 * a0_cat.matmul(a0_cat.transpose(0, 1))

        # get knn-neighbours
        neighbours = torch.argsort(l2_distances)[
            :, 1:params['num_neighbours']+1].tolist()

        for idx in range(num_relaxations):
            for idy in neighbours[idx]:
                if idx in neighbours[idy]:
                    neighbours[idy].remove(idx)

        # Get distance matrix
        t = time()
        max_dissimilarity = -1E6
        dissimilarity = max_dissimilarity * \
            (torch.ones((num_relaxations, num_relaxations)) - torch.eye(num_relaxations))

        def get_distances(idx_x, z_x, kneighbours):
            for idx_y in kneighbours:
                z_y = relaxation_list[idx_y]

                z_union = z_x.union(
                    z_y, 'pca', **params['params_union'])

                _, z_net, _ = self.propagate_relaxation(
                    z_union, label, idx_layer, **params)

                distance = z_net.get_verification_loss(
                    label, allow_negative_values=True)

                dissimilarity[idx_x, idx_y] = distance
                dissimilarity[idx_y, idx_x] = distance

        num_cores = multiprocessing.cpu_count() // 4

        Parallel(n_jobs=num_cores, require='sharedmem')(delayed(get_distances)(idx, x, neighbours[idx])
                                                        for idx, x in enumerate(relaxation_list))

        max_dissimilarity_among_neighbors = dissimilarity.max()
        dissimilarity[dissimilarity == max_dissimilarity] = 2 * \
            max_dissimilarity_among_neighbors
        dissimilarity = dissimilarity.div_(
            max_dissimilarity_among_neighbors).exp_()

        self.print(
            'Time used for Dissimilarity matrix: {:.2f}'.format(time() - t), 1)

        return dissimilarity

    def verify_patches_simple(self, dataset, patch_size, **params):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                  shuffle=False, num_workers=2)
        self.net.eval()

        counts_true = 0
        counts_total = dataset.targets.shape[0]

        disable_tqdm = self.verbosity < 2
        for _, (inputs, labels) in enumerate(tqdm(data_loader, disable=disable_tqdm)):

            # inputs, labels = inputs.to(self.device), labels.to(self.device)

            if not torch.argmax(self.net(inputs), 1) == labels:
                self.print('False prediction', 2)
                continue

            # relaxation_net = Zonotope_Net(self.net, self.device)
            # counts_true += relaxation_net.process_patch_combination(
            #     inputs, patch_size, labels)

            # Debugging start
            num_image_dim = inputs.shape[-1]
            num_patches_per_dimension = num_image_dim - patch_size + 1

            isVerified_All = []
            t = time()

            for idx in range(num_patches_per_dimension):
                for idy in range(num_patches_per_dimension):
                    x_lb = inputs.clone()
                    x_ub = inputs.clone()

                    x_lb[:, :, idx:(idx+patch_size), idy:(idy+patch_size)] = 0
                    x_ub[:, :, idx:(idx+patch_size), idy:(idy+patch_size)] = 1

                    z_net = params['relaxation_net_type'](self.net)
                    # z_net = Zonotope_Net(self.net)

                    z_net.initialize_from_bounds(x_lb, x_ub)
                    z_net.forward_pass()

                    isVerified = z_net.calculate_worst_case(labels)
                    isVerified_All.append(isVerified)

            counts_true += all(isVerified_All)
            self.print('Patches Verified/Total: {}/{}'.format(
                sum(isVerified_All), len(isVerified_All)), 2)
            self.print('Time: {:.2f}\n'.format(time()-t), 2)

        self.print('Verified: {}/{}'.format(counts_true, counts_total), 1)
        return counts_true / counts_total

    def verify_patch_proof_transfer(self, dataset, patch_size, **params):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                  shuffle=False, num_workers=2)
        self.net.eval()

        counts_true = 0
        counts_total = dataset.targets.shape[0]

        layer_indices = self.layer_indices_from_type(
            params['selected_layers'], params['num_skip_layers'], params['exclude_last_layer'])

        self.timelogger = utils.TimeLogger()
        timer_names = ['template_generation', 'initialization', ]
        timer_names += ['layer_{}'.format(i)
                        for i in range(len(self.net.layers))]
        timer_names += ['label_check', 'submatching', 'full_forward_pass']
        self.timelogger.add_timers(timer_names)

        # Test
        self.timelogger.add_timers(['test_num_verified'])

        disable_tqdm = self.verbosity < 2
        for idx_input, (inputs, labels) in enumerate(tqdm(data_loader, disable=disable_tqdm)):

            t1 = time()
            label = labels.item()

            if not torch.argmax(self.net(inputs), 1) == labels:
                self.print('False prediction', 2)
                continue

            self.timelogger.start_timer('template_generation')

            templates = self.patch_generate_templates(
                inputs, label, layer_indices, **params)
            # templates = {}
            self.timelogger.stop_timer('template_generation')

            self.timelogger.start_timer('full_forward_pass')
            isVerified = self.apply_patch_proof_transfer(inputs, patch_size,
                                                         label, templates, **params)
            self.timelogger.stop_timer('full_forward_pass')

            if isVerified:
                counts_true += 1

            self.print(
                'Time used for proof verification of item {}: {:.2f}\n'.format(idx_input, time()-t1), 2)

        self.print('Verified: {}/{}'.format(counts_true, counts_total), 1)
        self.print(self.timelogger.print_summary(), 1)
        return counts_true / counts_total

    def patch_generate_templates(self, inputs, labels, layer_indices, return_noise=False, **params):

        t = time()

        method = params['patch_template_method']
        self.timelogger2 = utils.TimeLogger()
        self.timelogger2.add_timers(
            ['first_shrinking', 'order_reduction', 'second_shrinking'])
        self.timelogger2.add_timers(
            ['forward_pass_shrinking_one', 'forward_pass_shrinking_two'])

        patch_names = [method]
        dataset = 'mnist' if inputs.shape[-1] == 28 else 'cifar'

        if method is None:
            return [{}]

        elif method == 'l_infinity':
            if dataset == 'mnist':
                eps = 0.05
            else:
                eps = 0.01
            noise = torch.ones_like(inputs) * eps

        elif method == 'center_patch':
            eps = 0.6
            patch_size = 8

            idx_x = (inputs.shape[2] - patch_size) // 2
            idx_y = (inputs.shape[3] - patch_size) // 2

            noise = torch.zeros_like(inputs)
            noise[:, :, idx_x:(idx_x+patch_size),
                  idx_y:(idx_y+patch_size)] = eps

        elif method == 'gradient_optimized':

            return self.patch_gradient_optimized_template(
                inputs, labels, layer_indices, return_noise, **params)

        elif method == 'border':
            eps = 0.05
            width = 10
            patch_size = inputs.shape[2] - 2 * width

            idx_x = (inputs.shape[2] - patch_size) // 2
            idx_y = (inputs.shape[3] - patch_size) // 2

            noise = torch.ones_like(inputs) * eps
            noise[:, :, idx_x:(idx_x+patch_size),
                  idx_y:(idx_y+patch_size)] = 0

        elif method == 'center_and_border':
            if dataset == 'mnist':
                eps_center = 0.4
                eps_border = 0.1
            else:
                eps_center = 0.18
                eps_border = 0.015

            center_size = 8
            noise = []

            idx_low = (inputs.shape[2] - center_size) // 2
            idx_up = idx_low + center_size
            idy_low = (inputs.shape[3] - center_size) // 2
            idy_up = idy_low + center_size

            n_center = torch.zeros_like(inputs)
            n_center[:, :, idx_low:idx_up, idy_low:idy_up] = eps_center
            noise.append(n_center)

            idx_low -= 1
            idx_up += 1
            idy_low -= 1
            idy_up += 1

            n_border = torch.ones_like(inputs) * eps_border
            n_border[:, :, idx_low:idx_up, idy_low:idy_up] = 0
            noise.append(n_border)

            patch_names = ['center', 'border']

        elif method == 'pyramid':
            eps = 0.10

            center_x = inputs.shape[2] / 2
            center_y = inputs.shape[3] / 2

            noise = torch.ones_like(inputs)

            for idx in range(inputs.shape[2]):
                for idy in range(inputs.shape[3]):
                    distance = max(abs(idx - center_x)/center_x,
                                   abs(idy - center_y)/center_y)
                    noise[:, :, idx, idy] = (1.0 - distance) * eps

        elif method == 'random':
            eps = 0.20
            noise = torch.rand_like(inputs) * eps

        elif 'chessboard' in method:
            if dataset == 'mnist':
                eps = 0.3
            else:
                eps = 0.06

            num_subpatches = int(method[-1])
            num_patches_x = inputs.shape[2]
            num_patches_y = inputs.shape[3]

            patches_step_x = num_patches_x / num_subpatches
            patches_step_y = num_patches_y / num_subpatches

            patch_names = ['cb{}_{}'.format(num_subpatches, i)
                           for i in range(num_subpatches * num_subpatches)]

            noise = []
            for idx in range(num_subpatches):
                for idy in range(num_subpatches):
                    idx_low = round(patches_step_x * idx - 0.49)
                    idx_up = round(patches_step_x * (idx + 1) + 0.49)
                    idy_low = round(patches_step_y * idy - 0.49)
                    idy_up = round(patches_step_y * (idy + 1) + 0.49)

                    n = torch.zeros_like(inputs)
                    n[:, :, idx_low:idx_up, idy_low:idy_up] = eps

                    noise.append(n)

        elif 'rotation' in method:
            return self.patch_rotation_template(
                inputs, labels, layer_indices, method, return_noise, **params)

        else:
            self.print(
                'Unknown patch template creation method: {}'.format(method), 0)
            raise RuntimeError

        if not isinstance(noise, list):
            noise = [noise]

        def shrink_template(n, name):
            return self.patch_shrink_templates_until_verified(
                n, inputs, labels, layer_indices,
                skip_first_shrinking=False, patch_name=name, **params)

        # num_cores = multiprocessing.cpu_count() // 4
        # res = Parallel(n_jobs=num_cores)(delayed(shrink_template)(n)
        #                                  for n in noise)
        # template_list, factors = zip(*res)

        template_list = []
        factors = []
        for n, name in zip(noise, patch_names):
            templates, factor = shrink_template(n, name)
            factors.append(factor)
            template_list.append(templates)

        self.print(
            'Time used for template generation: {:.2f}'.format(time()-t), 3)
        logger.debug(self.timelogger2.print_summary())
        # print(factors)

        if return_noise:
            noise_factored = [n * f for n, f in zip(noise, factors)]
            return template_list, noise_factored
        else:
            return template_list

    def patch_generate_templates_from_mask(self, inputs, labels, layer_indices, masks,
                                           return_noise=False, **params):

        t = time()

        # method = params['patch_template_method']
        self.timelogger2 = utils.TimeLogger()
        self.timelogger2.add_timers(
            ['first_shrinking', 'order_reduction', 'second_shrinking'])
        self.timelogger2.add_timers(
            ['forward_pass_shrinking_one', 'forward_pass_shrinking_two'])

        for idx_layer, (mask, epsilon) in masks.items():

            num_clusters = mask.shape[0]
            # num_epsilon = len(epsilon)
            # selected_eps = epsilon[int(num_epsilon * 0.8), :]

            noise = []

            for idx_cluster in range(num_clusters):

                # n = mask[[idx_cluster], :, :] * selected_eps[idx_cluster]
                n = mask[[idx_cluster], :, :]
                n = n.unsqueeze(0)

                dilation = False
                if dilation:
                    noise_shifted = utils.shift_image(n, 1)
                    noise_shifted.append(n)

                    n = torch.cat(noise_shifted, 0).max(0, keepdims=True)[0]

                noise.append(n)

        def shrink_template(n):
            return self.patch_shrink_templates_until_verified(
                n, inputs, labels, layer_indices, skip_first_shrinking=False, **params)

        # num_cores = multiprocessing.cpu_count() // 4
        # res = Parallel(n_jobs=num_cores)(delayed(shrink_template)(n)
        #                                  for n in noise)
        # template_list, factors = zip(*res)

        template_list = []
        factors = []
        for n in noise:
            templates, factor = shrink_template(n)
            factors.append(factor)
            template_list.append(templates)

        self.print(
            'Time used for template generation: {:.2f}'.format(time()-t), 3)
        logger.debug(self.timelogger2.print_summary())

        if return_noise:
            noise_factored = [n * f for n, f in zip(noise, factors)]
            return template_list, noise_factored
        else:
            return template_list

    def patch_shrink_templates_until_verified(self, noise, inputs, labels, layer_indices,
                                              patch_name=None, skip_first_shrinking=False, **params):

        templates = {i: None for i in layer_indices}
        self.timelogger2.start_timer('first_shrinking')

        if skip_first_shrinking:

            factor = 0.05
            relaxation_net = self.patch_propagate_noise(
                inputs, noise * factor, max(layer_indices))

        else:
            isVerified, relaxation_net, factor = self.patch_shrinking_one(
                inputs, noise, labels, patch_name=patch_name, **params)

            if not isVerified:
                return {}, 1.0

        self.timelogger2.stop_timer('first_shrinking')

        # Convert Zonotope to Parallelotope and shrink until it verifies
        for idx_layer in layer_indices:
            z = relaxation_net.relaxation_at_layers[idx_layer+1]

            self.timelogger2.start_timer('order_reduction')
            if params['patch_template_domain'] == 'parallelotope':
                z = z.order_reduction()
            elif params['patch_template_domain'] == 'box':
                z = z.to_box_star()
            else:
                self.print('Unknown patch template domain: {}'.format(
                    params['patch_template_domain']), 0)
                raise RuntimeError
            self.timelogger2.stop_timer('order_reduction')

            self.timelogger2.start_timer('second_shrinking')
            isVerified, z = self.patch_shrinking_two(
                z, labels, idx_layer, 1.0, 0.01, 0.01, patch_name, **params)
            if isVerified:
                templates[idx_layer] = z
            self.timelogger2.stop_timer('second_shrinking')

        remove_layers = []
        # Widening
        for idx_layer, z in templates.items():
            if z is None:
                self.print(
                    'Template creation failed at layer {}'.format(idx_layer), 2)
                remove_layers.append(idx_layer)
                continue

            if params['widening_for_patch_templates']:

                num_iterations = 5
                z = self.apply_gradient_based_widening2(
                    z, labels, idx_layer, num_iterations, **params)
                templates[idx_layer] = z
                # self.verification_templates[labels.item()][idx_layer].append(z)

        for idx_layer in remove_layers:
            del templates[idx_layer]

        return templates, factor

    def patch_shrinking_one(self, inputs, noise, labels, up=1.0, low=1E-3,
                            tolerance=1E-2, patch_name=None, **params):

        isFinished = False

        factor_verified = None
        relaxation_net_verified = None

        while not isFinished:

            middle = (up + low) * 0.5

            lower_bound = torch.clamp_min(inputs - noise * middle, 0)
            upper_bound = torch.clamp_max(inputs + noise * middle, 1)

            try:
                self.timelogger2.start_timer('forward_pass_shrinking_one')
            except Exception:
                pass

            relaxation_net = Zonotope_Net(self.net, self.device)
            relaxation_net.initialize_from_bounds(lower_bound, upper_bound)

            if False:
                # if params['use_patch_oracle']:
                oracle = params['patch_oracle'][labels][patch_name]
                template_layer = params['patch_oracle']['template_layer']

                for i in range(template_layer+1):
                    relaxation_net.apply_layer(i)

                z = relaxation_net.relaxation_at_layers[-1]
                lower_bound, upper_bound = z.get_bounds()
                bounds = torch.cat([lower_bound, upper_bound], 0)
                bounds = oracle.preprocess_data([bounds])

                isVerified = oracle.predict(bounds)[0]

            else:
                relaxation_net.forward_pass()
                isVerified = relaxation_net.calculate_worst_case(labels)

            try:
                self.timelogger2.stop_timer('forward_pass_shrinking_one')
            except Exception:
                pass

            isFinished, low, up = self.patch_binary_search(
                low, up, isVerified, tolerance)

            if isVerified:
                relaxation_net_verified = relaxation_net
                factor_verified = middle

        if factor_verified is not None:
            isVerified = True
            # if params['use_patch_oracle']:
            #     relaxation_net_verified.forward_pass(template_layer+1)

        else:
            isVerified = False

        return isVerified, relaxation_net_verified, factor_verified

    def patch_propagate_noise(self, inputs, noise, stop_layer):

        lower_bound = torch.clamp_min(inputs - noise, 0)
        upper_bound = torch.clamp_max(inputs + noise, 1)

        try:
            self.timelogger2.start_timer('forward_pass_shrinking_one')
        except Exception:
            pass

        relaxation_net = Zonotope_Net(self.net, self.device)
        relaxation_net.initialize_from_bounds(lower_bound, upper_bound)

        for idx_layer in range(stop_layer+1):
            relaxation_net.apply_layer(idx_layer)

        try:
            self.timelogger2.stop_timer('forward_pass_shrinking_one')
        except Exception:
            pass

        return relaxation_net

    def patch_shrinking_two(self, z, labels, idx_layer, up=1.0, low=0.4, tolerance=0.1,
                            patch_name=None, **params):

        z_net = Zonotope_Net(self.net, self.device)
        z_net.relaxation_at_layers = [z]
        isFinished = False

        middle_prev = 1.0
        factor_verified = None

        while not isFinished:

            middle = (up + low) * 0.5
            multiplication_factor = (middle / middle_prev)

            z.A *= multiplication_factor
            if isinstance(z, Box_Star):
                z.ub = z.a0 + torch.diag(z.A).view(1, -1)
                z.lb = z.a0 - torch.diag(z.A).view(1, -1)
                # z.ub = z.ub * multiplication_factor + \
                #     z.a0 * (1-multiplication_factor)
                # z.lb = z.lb * multiplication_factor + \
                #     z.a0 * (1-multiplication_factor)
            else:
                z.A_rotated_bounds *= (middle / middle_prev)
            middle_prev = middle

            self.timelogger2.start_timer('forward_pass_shrinking_two')
            z_net.truncate()

            # if False:
            if params['use_patch_oracle']:
                oracle = params['patch_oracle'][labels][patch_name]
                template_layer = params['patch_oracle']['template_layer']

                for i in range(idx_layer+1, template_layer+1):
                    z_net.apply_layer(i)

                z = z_net.relaxation_at_layers[-1]
                lower_bound, upper_bound = z.get_bounds()
                bounds = torch.cat([lower_bound, upper_bound], 0)
                bounds = oracle.preprocess_data(
                    [bounds], information='extended')

                isVerified = oracle.predict(bounds)[0]

            else:

                isVerified = z_net.process_from_layer(
                    labels, idx_layer+1)

            self.timelogger2.stop_timer('forward_pass_shrinking_two')

            if isVerified:
                factor_verified = middle

            isFinished, low, up = self.patch_binary_search(
                low, up, isVerified, tolerance)

        if factor_verified is not None:
            isVerified = True
            z.A *= (factor_verified / middle_prev)

            if isinstance(z, Box_Star):
                z.ub = z.a0 + torch.diag(z.A).view(1, -1)
                z.lb = z.a0 - torch.diag(z.A).view(1, -1)
            else:
                z.A_rotated_bounds *= (factor_verified / middle_prev)

        else:
            isVerified = False

        if params['use_patch_oracle']:
            z_net.relaxation_at_layers = [z]

            isVerified = False
            max_iter = 5
            idx_iter = 0
            while not isVerified and idx_iter < max_iter:
                idx_iter += 1

                isVerified = z_net.process_from_layer(
                    labels, idx_layer+1)

                if not isVerified:

                    z.A *= 0.9
                    z.A_rotated_bounds *= 0.9
                    z_net.truncate()

        if not isVerified:
            self.print('Template dropped: {}'.format(patch_name), 3)

        return isVerified, z

    def patch_binary_search(self, low, up, isVerified, tolerance):

        isFinished = (up - low) <= tolerance

        middle = (up + low) * 0.5

        if isVerified:
            low = middle
        else:
            up = middle

        return isFinished, low, up

    def patch_preprocess_template_masks(self, dataset, idx_layer, **params):

        dissimilarity, coordinates, input_shape = self.patch_get_dissimilarity_matrix(
            dataset, idx_layer, **params)

        embeddings = self.generate_constant_shift_embeddings(
            dissimilarity, 50, **params)

        num_templates = params['num_templates_per_class']

        cluster_assignments = self.ctg_cluster_with_coordinates(
            embeddings, num_templates, **params)

        input_shape.insert(0, num_templates)
        pixel_mask = torch.zeros(input_shape).long()

        for idx_cluster, (idx, idy) in zip(cluster_assignments, coordinates):

            pixel_mask[idx_cluster, idx, idy] = 1

        eps_list = []
        for idx_cluster in range(num_templates):

            eps = self.patch_get_highest_eps_for_mask(
                dataset, pixel_mask[idx_cluster])

            eps_list.append(torch.Tensor(sorted(eps)).unsqueeze(-1))

        eps_values = torch.cat(eps_list, -1)

        return (pixel_mask, eps_values), dissimilarity

    def patch_rotation_template(self, inputs, labels, layer_indices, method,
                                return_noise, **params):

        num_templates, angle_range = (int(x) for x in method[8:].split('_'))

        angle_step = angle_range / num_templates
        angles = [angle_range - (2*i+1) * angle_step
                  for i in range(num_templates)]
        angles = sorted(angles, key=abs)

        # code from: https://stackoverflow.com/questions/64197754/how-do-i-rotate-a-pytorch-image-tensor-around-its-center-in-a-way-that-supports # noqa
        def get_rot_mat(theta):
            theta = torch.tensor(theta * 3.141592653589793 / 180)
            return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                                 [torch.sin(theta), torch.cos(theta), 0]])

        def rot_img(x, theta):
            if theta == 0:
                return x
            rot_mat = get_rot_mat(theta)[None, ...].repeat(x.shape[0], 1, 1)
            grid = torch.nn.functional.affine_grid(rot_mat, x.size())
            x = torch.nn.functional.grid_sample(x, grid)
            return x

        template_list = []
        factors = []
        noise = []

        for angle in angles:
            inputs_rot = rot_img(inputs, angle)

            dataset = 'mnist' if inputs.shape[-1] == 28 else 'cifar'
            if dataset == 'mnist':
                eps = 0.05
            else:
                eps = 0.01
            n = torch.ones_like(inputs) * eps

            templates, factor = self.patch_shrink_templates_until_verified(
                n, inputs_rot, labels, layer_indices,
                skip_first_shrinking=False, patch_name=None, **params)

            factors.append(factor)
            template_list.append(templates)
            noise.append(n)

        if return_noise:
            noise_factored = [n * f for n, f in zip(noise, factors)]
            return template_list, noise_factored
        else:
            return template_list

    def patch_get_dissimilarity_matrix(self, dataset, idx_layer, **params):
        input_shape = None
        coordinates = []
        # num_pixels = None
        dissimilarity = None
        isInitialized = False

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                  shuffle=False, num_workers=2)

        for inputs, labels in data_loader:
            if not isInitialized:
                input_shape = list(inputs.shape[-2:])
                # num_pixels = input_shape[0] * input_shape[1]

            relaxation_list = []
            z_net = Zonotope_Net(self.net)

            for idx in range(input_shape[0]):
                for idy in range(input_shape[1]):
                    if not isInitialized:
                        coordinates.append((idx, idy))

                    lower_bound = inputs.clone()
                    upper_bound = inputs.clone()

                    lower_bound[:, :, idx, idy] = 0
                    upper_bound[:, :, idx, idy] = 1

                    z_net.initialize_from_bounds(lower_bound, upper_bound)

                    for idy_layer in range(idx_layer + 1):
                        z_net.apply_layer(idy_layer)

                    relaxation_list.append(z_net.relaxation_at_layers[-1])

            dissimilarity_i = self.patch_get_zonotope_dissimilarity(
                relaxation_list, **params)

            if not isInitialized:
                dissimilarity = dissimilarity_i
            else:
                dissimilarity += dissimilarity_i

            isInitialized = True

        return dissimilarity, coordinates, input_shape

    def patch_get_zonotope_dissimilarity(self, relaxation_list, **params):

        num_pixels = len(relaxation_list)
        dissimilarity = torch.zeros((num_pixels, num_pixels))

        bounds = [z.get_bounds() for z in relaxation_list]

        lower_bound, upper_bound = zip(*bounds)

        for bound in [lower_bound, upper_bound]:
            bound = [x.view(-1, 1) for x in bound]
            bound = torch.cat(bound, 1)

            bound_diff = bound.unsqueeze(1) - bound.unsqueeze(2)
            bound_diff = bound_diff.abs().sum(0)

            dissimilarity += bound_diff

        return dissimilarity

    def patch_get_highest_eps_for_mask(self, dataset, pixels, **params):

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                  shuffle=False, num_workers=2)
        eps_list = []

        for inputs, labels in data_loader:

            if not torch.argmax(self.net(inputs), 1) == labels:
                continue

            _, _, eps = self.patch_shrinking_one(
                inputs, pixels, labels, up=1.0, low=1E-4, tolerance=5E-3)

            if eps is not None:
                eps_list.append(eps)
            else:
                eps_list.append(0.0)

        return eps_list

    def patch_gradient_optimized_template(self, inputs, labels, layer_indices,
                                          return_noise=False, **params):

        max_iter = 100
        eps_init = 0.05
        factor_init = 0.8
        learning_rate = 1E-3

        weight_verification = 1E3
        weight_factor = 1E1
        # weight_widening = 1E-1

        template_layer = layer_indices[0]

        noise = torch.ones_like(inputs) * eps_init
        factor = torch.Tensor([factor_init])
        noise.requires_grad_(True)
        factor.requires_grad_(True)

        optimizer = torch.optim.SGD([noise], lr=learning_rate)
        relaxation_net = Zonotope_Net(self.net, self.device)
        relaxation_net2 = Zonotope_Net(self.net, self.device)

        for idx_iter in range(max_iter):

            lower_bound = torch.clamp_min(inputs - noise, 0)
            upper_bound = torch.clamp_max(inputs + noise, 1)

            relaxation_net.initialize_from_bounds(lower_bound, upper_bound)

            for idx_layer in range(template_layer+1):
                relaxation_net.apply_layer(idx_layer)

            z1 = relaxation_net.relaxation_at_layers[-1]
            z2 = z1.order_reduction()
            # with torch.no_grad():
            #     z2.a0 *= -1
            #     z2.A *= -1
            # z2 = z1

            # z2.A = z2.A * factor
            # z2.A_rotated_bounds = z2.A_rotated_bounds * factor

            relaxation_net2.relaxation_at_layers = [z2]
            for idx_layer in range(template_layer+1, len(self.net.layers)):
                relaxation_net2.apply_layer(idx_layer)

            isVerified = relaxation_net2.calculate_worst_case(labels)

            verification_loss = relaxation_net2.get_verification_loss(labels)
            widening_loss = noise.sum()
            factor_loss = factor

            loss = verification_loss * weight_verification - \
                widening_loss - factor_loss * weight_factor

            # loss = verification_loss * weight_verification
            # loss = widening_loss

            print_str = 'Verification loss: {:.3f}, '.format(
                verification_loss.item())
            print_str += 'Widening loss: {:.3f}, '.format(widening_loss.item())
            print_str += 'Factor loss: {:.3f}, '.format(factor_loss.item())
            print_str += 'Total loss: {:.3f}'.format(loss.item())
            self.print(print_str, 0)
            # print(relaxation_net2.y)
            # print(noise.max(), noise.min())

            if idx_iter + 1 < max_iter:
                optimizer.zero_grad()

                # z1.a0.retain_grad()
                # z2.a0.retain_grad()
                # z1.A.retain_grad()
                # z2.A.retain_grad()
                # relaxation_net2.y.retain_grad()
                # verification_loss.retain_grad()
                # for i, z in enumerate(relaxation_net2.relaxation_at_layers):
                #     z.a0.retain_grad()
                #     z.A.retain_grad()

                loss.backward()
                with torch.no_grad():
                    torch.nn.utils.clip_grad_norm_(
                        [noise, factor], 1E4)
                # print(noise.grad.abs().max() / weight_verification)

                # noise.grad *= -1

                # print('noise:', noise.grad is not None)
                # print('z1:', z1.a0.grad is not None, z1.A.grad is not None)
                # print('z2:', z2.a0.grad is not None, z2.A.grad is not None)
                # for i, z in enumerate(relaxation_net2.relaxation_at_layers):
                #     print(i+template_layer, z.a0.shape,
                #           z.a0.grad is not None, z.A.grad is not None)
                # print('y:', relaxation_net2.y.grad is not None)
                # print('v_loss:', verification_loss.grad is not None)
                # print('loss:', loss.grad is not None)
                # print(verification_loss)
                # print(loss)

                optimizer.step()

                with torch.no_grad():

                    noise.clamp_min_(0)
                    noise.clamp_max_(1)
                    factor.clamp_min_(0)
                    factor.clamp_max_(1)

        if isVerified:

            if return_noise:
                return [{template_layer: z2}], noise.detach()
            else:
                return [{template_layer: z2}]

        else:
            if return_noise:
                return [{}], noise.detach()
            else:
                return [{}]

    def optimize_model_graph(self, dataset):
        # Run two zonotopes through the network and make a gradient backpropagation
        # Torch then optimizes the forward propagation for following zonotopes faster

        eps = 0.01
        max_iterations = 2

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                  shuffle=True, num_workers=2)

        for idx_input, (inputs, labels) in enumerate(data_loader):

            if idx_input >= max_iterations:
                break

            z_net = Zonotope_Net(self.net)
            z_net.initialize(inputs, eps)
            z = z_net.relaxation_at_layers[0]
            z.a0.requires_grad_(True)
            z.A.requires_grad_(True)

            optimizer = torch.optim.Adam([z.a0, z.A], lr=1E-3)

            z_net.forward_pass()
            z_net.calculate_worst_case(labels)
            loss = z_net.get_verification_loss(labels)

            optimizer.zero_grad()
            loss.backward()
            # optimizer.step()

    def apply_patch_proof_transfer(self, inputs, patch_size, labels, templates, **params):
        temp = [list(t.keys()) for t in templates]
        layer_indices = []
        for t in temp:
            layer_indices.extend(t)
        layer_indices = list(set(layer_indices))

        num_image_dim = inputs.shape[-1]
        num_patches_per_dimension = num_image_dim - patch_size + 1

        isVerified_All = []
        num_submatches = 0

        for idx in range(num_patches_per_dimension):
            for idy in range(num_patches_per_dimension):

                self.timelogger.start_timer('initialization')

                z_net = params['relaxation_net_type'](self.net)
                x_lb = inputs.clone()
                x_ub = inputs.clone()

                x_lb[:, :, idx:(idx+patch_size), idy:(idy+patch_size)] = 0
                x_ub[:, :, idx:(idx+patch_size), idy:(idy+patch_size)] = 1

                z_net.initialize_from_bounds(x_lb, x_ub)
                self.timelogger.stop_timer('initialization')

                isSubmatch = False
                for idx_layer in range(len(self.net.layers)):
                    timer_name = 'layer_{}'.format(idx_layer)
                    self.timelogger.start_timer(timer_name)

                    z_net.apply_layer(idx_layer)
                    self.timelogger.stop_timer(timer_name)

                    if idx_layer in layer_indices:
                        z = z_net.relaxation_at_layers[-1]
                        self.timelogger.start_timer('submatching')
                        isSubmatch = False
                        for template in templates:
                            if not isSubmatch and idx_layer in template.keys():
                                isSubmatch = template[idx_layer].submatching(z)
                        self.timelogger.stop_timer('submatching')

                        if isSubmatch:
                            num_submatches += 1
                            isVerified_All.append(1)

                            # Test
                            self.timelogger.start_timer('test_num_verified')
                            self.timelogger.stop_timer('test_num_verified')

                            break

                if not isSubmatch:
                    self.timelogger.start_timer('label_check')
                    isVerified = z_net.calculate_worst_case(labels)
                    self.timelogger.stop_timer('label_check')
                    isVerified_All.append(isVerified)

                    if isVerified:
                        # Test
                        self.timelogger.start_timer('test_num_verified')
                        self.timelogger.stop_timer('test_num_verified')

        self.print('Patches Submatches/Verified/Total: {}/{}/{}'.format(
            num_submatches, sum(isVerified_All), len(isVerified_All)), 2)

        return all(isVerified_All)

    def verify_deepg_proof_transfer(self, folder, num_elements, **params):
        self.net.eval()

        counts_true = 0

        layer_indices = self.layer_indices_from_type(
            params['selected_layers'], params['num_skip_layers'], params['exclude_last_layer'])

        self.timelogger = utils.TimeLogger()
        timer_names = ['template_generation', 'initialization', ]
        timer_names += ['layer_{}'.format(i)
                        for i in range(len(self.net.layers))]
        timer_names += ['label_check', 'submatching', 'full_forward_pass']
        self.timelogger.add_timers(timer_names)

        # Test
        self.timelogger.add_timers(['test_num_verified'])

        disable_tqdm = self.verbosity < 2
        for idx_sample in tqdm(range(num_elements), disable=disable_tqdm):

            t1 = time()
            inputs, lower_bounds, upper_bounds, specs = utils.load_deepg_specs(
                idx_sample, folder)
            label = self.net(inputs)[0].argmax().item()

            self.timelogger.start_timer('template_generation')

            templates = self.patch_generate_templates(
                inputs, label, layer_indices, **params)
            # templates = {}
            self.timelogger.stop_timer('template_generation')

            self.timelogger.start_timer('full_forward_pass')
            isVerified = self.apply_deepg_proof_transfer(lower_bounds, upper_bounds,
                                                         label, templates, **params)
            self.timelogger.stop_timer('full_forward_pass')

            if isVerified:
                counts_true += 1

            self.print(
                'Time used for proof verification of item {}: {:.2f}\n'.format(idx_sample, time()-t1), 2)

        self.print('Verified: {}/{}'.format(counts_true, num_elements), 1)
        self.print(self.timelogger.print_summary(), 1)
        return counts_true / num_elements

    def apply_deepg_proof_transfer(self, lower_bounds, upper_bounds, label, templates, **params):
        temp = [list(t.keys()) for t in templates]
        layer_indices = []
        for t in temp:
            layer_indices.extend(t)
        layer_indices = list(set(layer_indices))

        num_splits = lower_bounds.shape[0]

        isVerified_All = []
        num_submatches = 0

        for idx_split in range(num_splits):

            self.timelogger.start_timer('initialization')

            z_net = params['relaxation_net_type'](self.net)
            z_net.initialize_from_bounds(
                lower_bounds[[idx_split]], upper_bounds[[idx_split]])
            self.timelogger.stop_timer('initialization')

            isSubmatch = False
            for idx_layer in range(len(self.net.layers)):
                timer_name = 'layer_{}'.format(idx_layer)
                self.timelogger.start_timer(timer_name)

                z_net.apply_layer(idx_layer)
                self.timelogger.stop_timer(timer_name)

                if idx_layer in layer_indices:
                    z = z_net.relaxation_at_layers[-1]
                    self.timelogger.start_timer('submatching')
                    isSubmatch = False
                    for template in templates:
                        if not isSubmatch and idx_layer in template.keys():
                            isSubmatch = template[idx_layer].submatching(z)
                    self.timelogger.stop_timer('submatching')

                    if isSubmatch:
                        num_submatches += 1
                        isVerified_All.append(1)

                        # Test
                        self.timelogger.start_timer('test_num_verified')
                        self.timelogger.stop_timer('test_num_verified')

                        break

            if not isSubmatch:
                self.timelogger.start_timer('label_check')
                isVerified = z_net.calculate_worst_case(label)
                self.timelogger.stop_timer('label_check')
                isVerified_All.append(isVerified)

                if isVerified:
                    # Test
                    self.timelogger.start_timer('test_num_verified')
                    self.timelogger.stop_timer('test_num_verified')

        self.print('Patches Submatches/Verified/Total: {}/{}/{}'.format(
            num_submatches, sum(isVerified_All), len(isVerified_All)), 2)

        return all(isVerified_All)

    def derive_nullspace_directions(self, layer):
        import scipy
        from scipy import linalg  # noqa: F401

        for idx_layer in reversed(range(layer, len(self.net.layers))):

            if not isinstance(self.net.layers[idx_layer], torch.nn.Linear):
                continue
            if idx_layer in self.nullspace_directions.keys():
                continue

            weight = self.net.layers[idx_layer].weight.data.clone()
            weight = weight.transpose(0, 1)
            num_input, num_output = weight.shape

            # concatenate with layer layers
            if idx_layer < (len(self.net.layers) - 1):
                next_layer = idx_layer + 1
                while not isinstance(self.net.layers[next_layer],
                                     (torch.nn.Linear, torch.nn.Conv2d)):
                    next_layer += 1
                prev_weight = self.nullspace_directions[next_layer][0]
                weight = weight.matmul(prev_weight)
                # weight = prev_weight.transpose(1, 0).matmul(weight)

                # print('weight before', weight.shape, prev_weight.shape)
            # get nullspace
            if num_output < num_input:

                w_nullspace = scipy.linalg.null_space(weight.numpy().T, 1E-10)
                weight = torch.cat([weight, torch.Tensor(w_nullspace)], 1)

            weight_inverse = torch.inverse(weight)

            self.nullspace_directions[idx_layer] = [weight, weight_inverse]

    def layer_indices_from_type(self, selected_layers, num_skip_layers=0,
                                exclude_last_layer=False):

        indices_selected_layers = []

        if num_skip_layers < 0:
            num_skip_layers += len(self.net.layers)
        selected_layers_torch = []
        for layer in selected_layers:
            if layer == 'linear':
                selected_layers_torch.append(torch.nn.Linear)
                selected_layers_torch.append(torch.nn.Conv2d)
            elif layer == 'relu':
                selected_layers_torch.append(torch.nn.ReLU)
            elif layer == 'pooling':
                selected_layers_torch.append(torch.nn.MaxPool2d)
                selected_layers_torch.append(torch.nn.AvgPool2d)
            elif layer == 'flatten':
                selected_layers_torch.append(torch.nn.Flatten)
            elif isinstance(layer, int):
                indices_selected_layers.append(layer)
            else:
                logger.warn('Unknown layer: {}'.format(layer))

        selected_layers_torch = tuple(selected_layers_torch)
        if exclude_last_layer:
            last_layer = len(self.net.layers)-3
        else:
            last_layer = len(self.net.layers)-1
        for idx_layer in range(num_skip_layers, last_layer):
            if isinstance(self.net.layers[idx_layer], selected_layers_torch):
                indices_selected_layers.append(idx_layer)

        indices_selected_layers.sort()

        return indices_selected_layers

    def remove_gradient(self):
        self.net.remove_gradient()

    def add_gradient(self, skip_layer=1):
        self.net.add_gradient()

    def save_net(self, path):
        torch.save(self.net, path)
        self.print('Model save to {}'.format(path), 0)

    def load_net(self, path):
        self.net = torch.load(path, map_location=torch.device(self.device))
        self.net.eval()
        self.print('Model load from {}'.format(path), 2)

    def create_and_save_milp_model(self, path, input_shape, **params):

        selected_layers = self.layer_indices_from_type(
            params['selected_layers'], params['num_skip_layers'], params['exclude_last_layer'])

        input = torch.zeros(input_shape)

        for idx_layer, layer in enumerate(self.net.layers):
            input = layer(input)
            if idx_layer in selected_layers:
                s_net = Star_Net(self.net)
                s_net.use_overapproximation_only = False
                s_net.use_general_milp_modeling = True

                s_net.initialize_from_bounds(input - 0.1, input + 0.1)
                s_net.forward_pass(idx_layer + 1)

                milp_model = s_net.milp_model
                # milp_model.remove(s_net.milp_input_constraints)

                milp_model.write('{}_{}.mps'.format(path, idx_layer))

                # import gurobipy as gp
                # new_model = gp.read('{}_{}.mps'.format(path, idx_layer))

    def load_milp_model(self, path, **params):
        import gurobipy as gp

        selected_layers = self.layer_indices_from_type(
            params['selected_layers'], params['num_skip_layers'])

        for idx_layer in selected_layers:

            try:
                milp_model = gp.read('{}_{}.mps'.format(path, idx_layer))
                milp_model.Params.OutputFlag = 0
                self.milp_models[idx_layer] = milp_model

                self.print(
                    'MILP model added for layer {}'.format(idx_layer), 2)

            except Exception:
                self.print(
                    'MILP model not found for layer {}'.format(idx_layer), 2)

    def save_templates(self, path):
        pickle.dump(self.verification_templates,
                    open('{}.pkl'.format(path), 'wb'))

    def load_templates(self, path):
        self.verification_templates = pickle.load(
            open('{}.pkl'.format(path), 'rb'))

        layers = []
        for label in self.verification_templates.keys():
            for idx_layer in self.verification_templates[label].keys():
                layers.append(idx_layer)

        layer_indices_template = list(set(layers))
        layer_indices_template.sort()

        self.reset_counts(layer_indices_template)

    def fgsm_untargeted_attack(self, input, true_label, eps_step, bounds=None):
        modified_input = input.clone().detach().requires_grad_(True).to(self.device)
        target = torch.LongTensor(true_label).to(self.device)

        if False:
            prediction = self.net(modified_input, aux_outputs=True)
            loss = self.criterion(prediction[0], target)

            weights = [0.0, 0.0, 1.0]

            for idx, value in enumerate(prediction[1:]):
                loss += self.criterion(value, target) * weights[idx]

        else:
            prediction = self.net(modified_input)
            loss = self.criterion(prediction, target)

        self.net.zero_grad()

        loss.backward()

        modified_input = modified_input + eps_step * modified_input.grad.sign()

        if bounds is not None:
            modified_input.clamp_(min=bounds[0], max=bounds[1])

        return modified_input

    def pgd_untargeted_attack(self, input, true_label, eps, num_steps, bounds=None):
        self.net.eval()

        input_min = input - eps
        input_max = input + eps
        eps_steps = 4 * eps / num_steps

        modified_input = input.detach().clone().to(self.device)
        noise = torch.empty_like(modified_input).uniform_(-eps, eps)
        modified_input.add(noise)

        for _ in range(num_steps):
            modified_input = self.fgsm_untargeted_attack(
                modified_input, true_label, eps_steps, bounds)
            modified_input = torch.max(input_min, modified_input)
            modified_input = torch.min(input_max, modified_input)

        return modified_input

    def reset_counts(self, layer_indices_template):
        submatching_dict = {}
        submatching_total_dict = {}
        union_dict = {}

        for idx_layer in layer_indices_template:
            union_dict[idx_layer] = 0
            submatching_dict[idx_layer] = {}
            submatching_total_dict[idx_layer] = 0
            for idx_label in range(10):
                submatching_dict[idx_layer][idx_label] = 0

        self.verification_counts = {
            'false_prediction': 0, 'submatching': submatching_dict,
            'submatching_total': submatching_total_dict, 'union': union_dict,
            'new_template': 0, 'manual_positive': 0, 'manual_negative': 0}

    def print(self, str, importance=3):
        if importance <= self.verbosity:
            logger.info(str)
        else:
            logger.debug(str)

    def print_number_of_templates(self):
        for label in self.verification_templates.keys():
            print_str = 'Number of templates for label {}:     '.format(label)
            for idx_layer in self.verification_templates[label].keys():
                num_templates = len(
                    self.verification_templates[label][idx_layer])
                print_str += 'Layer {}: {}, '.format(idx_layer, num_templates)

            self.print(print_str, 0)
