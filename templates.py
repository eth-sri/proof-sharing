import torch
import logging
import os
import pickle
from tqdm import tqdm
from time import time
from joblib import Parallel, delayed
import multiprocessing
from sklearn import cluster as sklearn_cluster

from relaxations import Zonotope_Net, Star_Net, Box, Parallelotope
import utils

logger = logging.getLogger()


def binary_search(low, up, tol, isVerified):

    isFinished = (up - low) <= tol
    middle = (up + low) * 0.5

    if isVerified:
        low = middle
    else:
        up = middle

    return isFinished, low, up


def shrinking_one(input, noise, net, label, relu_transformer, up=1.0, low=1E-3, tol=1E-2):

    isFinished = False
    relaxations_net_verified = None

    while not isFinished:

        middle = (up + low) * 0.5

        lower_bound = torch.clamp_min(input - noise * middle, 0)
        upper_bound = torch.clamp_max(input + noise * middle, 1)

        relaxation_net = Zonotope_Net(
            net, relu_transformer=relu_transformer)
        relaxation_net.initialize_from_bounds(lower_bound, upper_bound)
        relaxation_net.forward_pass()
        isVerified = relaxation_net.calculate_worst_case(label)

        isFinished, low, up = binary_search(
            low, up, tol, isVerified)

        if isVerified:
            relaxations_net_verified = relaxation_net

    isVerified = relaxations_net_verified is not None    
    
    return isVerified, relaxations_net_verified



class OnlineTemplates:

    def __init__(self, net, layers, label, domain='box',
                 relu_transformer='zonotope'):
        self.net = net
        self.layers = layers
        self.label = label
        self.domain = domain
        self.templates = {x: [] for x in layers}
        self.relu_transformer = relu_transformer

    def create_templates(self, inputs, method):

        input_list, noise_list = self._get_input_and_noise(inputs, method)

        for input, noise in zip(input_list, noise_list):            
            isVerified, relaxation_net = shrinking_one(input, noise, self.net, self.label, self.relu_transformer)
            if not isVerified:
                continue
            relaxations = relaxation_net.relaxation_at_layers[1:]

            for idx_layer in self.layers:
                z = relaxations[idx_layer]

                if self.domain == 'box':
                    z = z.to_box()
                elif self.domain == 'parallelotope':
                    z = z.to_parallelotope()
                else:
                    logger.error(
                        'Unknown template domain: {}'.format(self.domain))
                    raise RuntimeError

                isVerified, z = self._shrinking_two(z, idx_layer)
                if isVerified:
                    self.templates[idx_layer].append(z)

    def _get_input_and_noise(self, inputs, method):
        dataset = 'mnist' if inputs.shape[-1] == 28 else 'cifar'

        if method is None:
            return [], []

        elif method == 'l_infinity':
            if dataset == 'mnist':
                eps = 0.05
            else:
                eps = 0.01
            noise = torch.ones_like(inputs) * eps

            return [inputs], [noise]

        elif method == 'center':
            if dataset == 'mnist':
                eps = 0.4
            else:
                eps = 0.18

            patch_size = 6

            idx_x = (inputs.shape[2] - patch_size) // 2
            idx_y = (inputs.shape[3] - patch_size) // 2

            noise = torch.zeros_like(inputs)
            noise[:, :, idx_x:(idx_x+patch_size),
                  idx_y:(idx_y+patch_size)] = eps

            return [inputs], [noise]

        elif method == 'border':
            if dataset == 'mnist':
                eps = 0.1
            else:
                eps = 0.015
            width = 10
            patch_size = inputs.shape[2] - 2 * width

            idx_x = (inputs.shape[2] - patch_size) // 2
            idx_y = (inputs.shape[3] - patch_size) // 2

            noise = torch.ones_like(inputs) * eps
            noise[:, :, idx_x:(idx_x+patch_size),
                  idx_y:(idx_y+patch_size)] = 0

            return [inputs], [noise]

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

            return [inputs] * 2, noise

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

            return [inputs], [noise]

        elif method == 'random':
            if dataset == 'mnist':
                eps = 0.20
            else:
                eps = 0.05
            noise = torch.rand_like(inputs) * eps

            return [inputs], [noise]

        elif 'grid' in method:
            # Grid size given after 'grid'
            # So 'grid3' creates a 3x3 grid with 9 templates in total
            if dataset == 'mnist':
                eps = 0.3
            else:
                eps = 0.06

            num_subpatches = int(method[-1])
            num_patches_x = inputs.shape[2]
            num_patches_y = inputs.shape[3]

            patches_step_x = num_patches_x / num_subpatches
            patches_step_y = num_patches_y / num_subpatches

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

            return [inputs] * (num_subpatches**2), noise

        elif 'rotation' in method:
            # Number of templates and rotation range given after 'rotation'
            # So 'rotation3_40' creates 3 templates, equally covering the range
            # [-40, 40] degrees

            num_templates, angle_range = (int(x)
                                          for x in method[8:].split('_'))

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
                rot_mat = get_rot_mat(
                    theta)[None, ...].repeat(x.shape[0], 1, 1)
                grid = torch.nn.functional.affine_grid(
                    rot_mat, x.size(), align_corners=False)
                x = torch.nn.functional.grid_sample(
                    x, grid, align_corners=False)
                return x

            input_list = []
            noise_list = []
            for angle in angles:
                inputs_rot = rot_img(inputs, angle)
                input_list.append(inputs_rot)

                if dataset == 'mnist':
                    eps = 0.05
                else:
                    eps = 0.01
                n = torch.ones_like(inputs) * eps
                noise_list.append(n)

            return input_list, noise_list

        else:
            logging.error(
                'Unknown patch template creation method: {}'.format(method))
            raise RuntimeError

    def _shrinking_two(self, z, layer, up=1.0, low=1E-2, tol=1E-2):

        z_net = Zonotope_Net(
            self.net, relu_transformer=self.relu_transformer)
        z_net.relaxation_at_layers = [z]
        isFinished = False

        middle_prev = 1.0
        factor_verified = None

        while not isFinished:

            middle = (up + low) * 0.5
            multiplication_factor = (middle / middle_prev)
            middle_prev = middle

            z.scale(multiplication_factor)

            z_net.truncate()
            isVerified = z_net.process_from_layer(
                self.label, layer+1)

            if isVerified:
                factor_verified = middle

            isFinished, low, up = binary_search(
                low, up, tol, isVerified)

        if factor_verified is not None:
            isVerified = True

            z.scale(factor_verified / middle_prev)

        return isVerified, z

    def submatching(self, z, layer):
        assert layer in self.layers

        for t in self.templates[layer]:
            isSubmatch = t.submatching(z)
            if isSubmatch:
                return True
        return False


class OfflineTemplates:

    def __init__(self, net, layers, label, domain='box',
                 relu_transformer='zonotope'):
        self.net = net
        self.layers = layers
        self.label = label
        self.domain = domain
        self.templates = {x: [] for x in layers}
        self.relu_transformer = relu_transformer

    def create_templates(self, dataset, epsilon, path_to_net, use_hyperplanes=False,
                         num_templates=100, max_epsilon=False):

        for layer in self.layers:

            relaxations = self._get_intermediate_relaxations(
                dataset, epsilon, layer, path_to_net, max_epsilon)
            dissimilarity = self._get_dissimilarity_matrix(
                relaxations, epsilon, layer, path_to_net)
            cluster_set = self._create_cluster_set(relaxations, dissimilarity)
            cluster_set = self._verify_templates(
                cluster_set, use_hyperplanes, layer)
            cluster_set = self._merge_templates(
                cluster_set, use_hyperplanes, layer)

            self._store_templates(cluster_set, path_to_net, epsilon, use_hyperplanes,
                                  layer, num_templates, use_widening=False)
            cluster_set = self._widen_templates(
                cluster_set, use_hyperplanes, layer)
            self._store_templates(cluster_set, path_to_net, epsilon, use_hyperplanes,
                                  layer, num_templates, use_widening=True)

    def _get_intermediate_relaxations(self, dataset, epsilon, layer, path_to_net, max_epsilon):

        path = os.path.dirname(path_to_net)
        net_name = os.path.basename(path_to_net).rsplit('.')[0]
        naming = '_'.join([net_name, 'intermediate', str(layer),
                           '{:.3f}'.format(epsilon),
                           self.relu_transformer
                           ])
        if max_epsilon:
            naming += '_max'

        
        prefix = path + '/intermediate_zonotopes/' + naming

        relaxation_list = []
        num_verified = 0
        num_predicted = 0

        print(naming, prefix)

        if os.path.exists(prefix + '_' + str(self.label) + '_00000.pkl'):
            # Load already precomputed intermediate zonotopes

            intermediate_dataset = utils.IntermediateDataset(
                prefix, None, [self.label])

            data_loader = torch.utils.data.DataLoader(intermediate_dataset, batch_size=1,
                                                      shuffle=False, num_workers=0,
                                                      collate_fn=utils.custom_collate)
            num_samples = len(data_loader)

            for relaxations, label, isPredicted, isVerified in tqdm(data_loader):

                if label == self.label and isVerified:

                    relaxation_list.append(relaxations[layer])

                num_predicted += isPredicted
                num_verified += isVerified

            logger.info(
                'Intermediate zonotopes loaded for layer ' + str(layer))
        else:
            # Create and store intermediate zonotopes
            logger.info(
                'Create intermediate zonotopes for layers ' + str(self.layers))

            data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                      shuffle=False, num_workers=1)
            num_samples = len(data_loader)
            data_drop = {}

            for idx_sample, (inputs, labels) in enumerate(tqdm(data_loader)):

                isPredicted = (torch.argmax(
                    self.net(inputs), 1) == labels).item()
                label = labels.item()


                if max_epsilon:
                    isVerified, relaxation = shrinking_one(inputs,
                                                           torch.ones_like(inputs),
                                                           self.net,
                                                           label,
                                                           self.relu_transformer)
                    if relaxation is None:
                        relaxation = Zonotope_Net(
                            self.net, relu_transformer=self.relu_transformer)
                        isVerified = bool(
                            relaxation.process_input_once(inputs, epsilon, labels))
                else:
                    relaxation = Zonotope_Net(
                        self.net, relu_transformer=self.relu_transformer)
                    isVerified = bool(
                        relaxation.process_input_once(inputs, epsilon, labels))

                data_drop['isPredicted'] = isPredicted
                data_drop['isVerified'] = isVerified
                data_drop['label'] = label

                num_predicted += isPredicted
                num_verified += isVerified

                for idx_layer in self.layers:
                    z = relaxation.relaxation_at_layers[idx_layer + 1]
                    intermediate_relaxations = {idx_layer: torch.cat(
                        [z.a0, z.A], 0).detach()}
                    data_drop['intermediate_relaxations'] = intermediate_relaxations

                    number_str = str(idx_sample).zfill(5)
                    naming = '_'.join([net_name, 'intermediate', str(idx_layer),
                                       '{:.3f}'.format(epsilon),
                                       self.relu_transformer, str(self.label), number_str])
                    drop_name = path + '/intermediate_zonotopes/' + naming + '.pkl'

                    pickle.dump(data_drop, open(drop_name, 'wb'))

                    if idx_layer == layer:
                        relaxation_list.append(z)

            logger.info(
                'Intermediate zonotopes created for layers ' + str(self.layers))

        logger.info('Num Predicted: {}/{} -> {:.3f}'.format(
            num_predicted, num_samples, num_predicted/num_samples))
        logger.info('Num Verified: {}/{} -> {:.3f}'.format(
            num_verified, num_samples, num_verified/num_samples))

        return relaxation_list

    def _get_dissimilarity_matrix(self, relaxations, epsilon, layer, path_to_net):

        path = os.path.dirname(path_to_net)
        net_name = os.path.basename(path_to_net).rsplit('.')[0]
        naming = '_'.join([net_name, 'dissimilarity', str(layer), str(self.label),
                           '{:.3f}'.format(epsilon), self.relu_transformer])
        full_path = path + '/templates/' + naming + '.pkl'

        logger.info('Creating dissimilarity matrix')

        if os.path.exists(full_path):
            # Load dissimilarity matrix
            dissimilarity = pickle.load(open(full_path, 'rb'))
            assert len(relaxations) == dissimilarity.shape[0]

            logger.info('Dissimilarity matrix loaded')

        else:
            # Create dissimilarity matrix
            num_relaxations = len(relaxations)
            num_neighbours = 20

            # get list of centers
            lower_bound, upper_bound = relaxations[0].get_bounds()
            for z in relaxations[1:]:
                bounds = z.get_bounds()

                lower_bound = torch.min(lower_bound, bounds[0])
                upper_bound = torch.max(upper_bound, bounds[1])

            active_neurons = upper_bound[0, :] > lower_bound[0, :]

            a0_list = [z.a0_flat[:, active_neurons]
                       for z in relaxations]
            a0_cat = torch.cat(a0_list, 0)

            # get l2-distance
            a0_norm = (a0_cat**2).sum(1).view(-1, 1)
            a0_norm_t = a0_norm.view(1, -1)
            l2_distances = a0_norm + a0_norm_t \
                - 2 * a0_cat.matmul(a0_cat.transpose(0, 1))

            # get knn-neighbours
            neighbours = torch.argsort(l2_distances)[
                :, 1:num_neighbours+1].tolist()

            for idx in range(num_relaxations):
                for idy in neighbours[idx]:
                    if idx in neighbours[idy]:
                        neighbours[idy].remove(idx)

            # Get distance matrix
            max_dissimilarity = -1E6
            dissimilarity = max_dissimilarity * \
                (torch.ones((num_relaxations, num_relaxations)) -
                 torch.eye(num_relaxations))

            def get_distances(idx_x, z_x, kneighbours):
                for idx_y in kneighbours:
                    z_y = relaxations[idx_y]

                    if self.domain == 'box':
                        z_union = z_x.union(z_y, 'box')
                    elif self.domain == 'parallelotope':
                        z_union = z_x.union(z_y, 'pca')
                    else:
                        logger.error('Unknown domain: {}'.format(self.domain))
                        raise RuntimeError

                    z_net = Zonotope_Net(
                        self.net, relu_transformer=self.relu_transformer)
                    z_net.relaxation_at_layers = [z_union]
                    z_net.process_from_layer(self.label, layer+1)

                    distance = z_net.get_verification_loss(
                        self.label, allow_negative_values=True)

                    dissimilarity[idx_x, idx_y] = distance
                    dissimilarity[idx_y, idx_x] = distance

            num_cores = multiprocessing.cpu_count() // 4

            Parallel(n_jobs=num_cores, require='sharedmem')(delayed(get_distances)(idx, x, neighbours[idx])
                                                            for idx, x in enumerate(relaxations))

            max_dissimilarity_among_neighbors = dissimilarity.max()
            dissimilarity[dissimilarity == max_dissimilarity] = 2 * \
                max_dissimilarity_among_neighbors
            dissimilarity = dissimilarity.div_(
                max_dissimilarity_among_neighbors).exp_()

            pickle.dump(dissimilarity, open(full_path, 'wb'))

            logger.info('Dissimilarity matrix created')

        return dissimilarity

    def _create_cluster_set(self, relaxations, dissimilarity):

        logger.info('Cluster intermediate zonotopes')

        average_cluster_size = 50
        num_clusters = len(relaxations) // average_cluster_size

        cluster_assignments = self._cluster_relaxations(
            dissimilarity, num_clusters)

        cluster_assignments = list(cluster_assignments)
        assert(len(relaxations) == len(cluster_assignments))

        true_num_clusters = len(set(cluster_assignments)) - \
            (1 if -1 in cluster_assignments else 0)

        cluster_set = {}
        indices_multiple = [i for i in range(true_num_clusters)
                            if cluster_assignments.count(i) > 1]

        for idx_cluster in indices_multiple:
            indices_assigned_relaxations = [i for i in range(len(relaxations))
                                            if cluster_assignments[i] == idx_cluster]

            assigned_relaxations = [relaxations[i]
                                    for i in indices_assigned_relaxations]

            num_relaxations = len(assigned_relaxations)

            cluster_information = {}
            cluster_information['relaxations'] = assigned_relaxations
            cluster_information['indices'] = indices_assigned_relaxations
            cluster_information['num_relaxations'] = num_relaxations

            cluster_set[idx_cluster] = cluster_information

        unused_indices = [i for i in range(len(cluster_assignments))
                          if cluster_assignments[i] not in indices_multiple]

        unused_relaxations = [relaxations[i] for i in unused_indices]

        unused_cluster_information = {}
        unused_cluster_information['relaxations'] = unused_relaxations
        unused_cluster_information['indices'] = unused_indices
        unused_cluster_information['num_relaxations'] = len(unused_indices)
        unused_cluster_information['union'] = None

        cluster_set[-1] = unused_cluster_information

        logger.info('Intermediate zonotopes clustered')

        return cluster_set

    def _cluster_relaxations(self, dissimilarity, num_clusters):

        # Use Constant Shift Embeddings to transform dissimilarity matrix
        # into Euclidean space
        num_relaxations = dissimilarity.shape[0]
        p = 20

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

        # Cluster relaxations
        alg = sklearn_cluster.KMeans(num_clusters, init='k-means++')
        cluster_assignments = alg.fit_predict(X)

        return cluster_assignments

    def _verify_templates(self, cluster_set, use_hyperplanes, layer):

        t = time()

        cluster_indices = [i for i in cluster_set.keys() if i > -1]

        unused_cluster = cluster_set[-1]
        num_verified_clusters = 0

        for idx_iter, idx_cluster in enumerate(cluster_indices):

            c = cluster_set[idx_cluster]

            relaxations = c['relaxations']

            if self.domain == 'box':
                z_union = relaxations[0].union(relaxations[1:], 'box')
            elif self.domain == 'parallelotope':
                z_union = relaxations[0].union(relaxations[1:], 'pca')

            isVerified = self._verify_with_milp(
                z_union, layer, relaxations, use_hyperplanes, max_iterations=30,
                interpolation_weight=0.05)

            if isVerified:
                c['union'] = z_union
                num_verified_clusters += 1

                logger.info('Verified cluster {} of {} with {} relaxations and {} constraints'.format(
                    idx_iter+1, len(cluster_indices), c['num_relaxations'],
                    c['union'].num_constraints))

            else:
                c['union'] = None

                unused_cluster['relaxations'].extend(relaxations)
                unused_cluster['indices'].extend(c['indices'])
                unused_cluster['num_relaxations'] += c['num_relaxations']

                logger.info('Verification of cluster {} of {} with {} relaxations failed'.format(
                    idx_iter+1, len(cluster_indices), c['num_relaxations']))

        logger.info('Cluster unions verified: {}/{}, time used: {:.2f}'.format(
            num_verified_clusters, len(cluster_indices), time() - t))

        return cluster_set

    def _verify_with_milp(self, z, layer, relaxations=None, use_hyperplanes=True, max_iterations=30,
                          interpolation_weight=0.05):

        t = time()

        s_net = Star_Net(self.net, relu_transformer=self.relu_transformer)
        s_net.relaxation_at_layers = [z]

        s_net.use_overapproximation_only = False
        s_net.use_general_zonotope = False
        s_net.num_solutions = 10
        s_net.early_stopping_objective = 1.0
        s_net.early_stopping_bound = -1E-5
        s_net.timelimit = 60*30
        s_net.use_tighter_bounds = True
        s_net.use_tighter_bounds_using_milp = True
        s_net.use_lp = False
        s_net.use_retightening = False
        s_net.milp_neuron_ratio = 1.0

        isVerified, violations = s_net.process_from_layer(self.label, layer+1,
                                                          return_violation=True)

        try:
            objective = s_net.milp_model.objVal
        except Exception:
            objective = s_net.get_verification_loss(
                self.label, allow_negative_values=True)
        objective_str = '{:.4f}'.format(objective)

        logger.info('-1, loss: {}, d: None, time: {:.2f}'.format(
            objective_str, time() - t))

        if violations is None:
            return isVerified
        else:
            if s_net.milp_model.Status == 2:
                prev_objective = s_net.milp_model.objVal
            else:
                prev_objective = 1E5

        if not use_hyperplanes:
            return isVerified

        idx_iter = 0

        while not isVerified and idx_iter < max_iterations:

            C, d = self._derive_hyperplane_intersection(z, violations, relaxations,
                                                        interpolation_weight)

            isVerified, violations = s_net.rerun_with_additional_constraint(
                C, d)

            try:
                objective = s_net.milp_model.objVal
            except Exception:
                objective = s_net.get_verification_loss(
                    self.label, allow_negative_values=True)
            objective_str = '{:.4f}'.format(objective)

            d_values = ', '.join(['{:.3f}'.format(x.item()) for x in d])
            logger.info('{}, loss: {}, d: {}, time: {:.2f}'.format(
                idx_iter, objective_str, d_values, time() - t))

            if violations is None:
                break

            if (idx_iter > 10 and objective > 1.0) or (idx_iter > 20 and objective > 0.2):
                logger.info('Loss too large, stop truncation')
                break

            if s_net.milp_model.Status == 2:
                if objective >= prev_objective:
                    break
                else:
                    prev_objective = objective

            idx_iter += 1

        return isVerified

    def _derive_hyperplane_intersection(self, z, violations, relaxations,
                                        interpolation_weight):

        center = z.a0
        center = center.view(1, -1)
        C_list = []
        d_list = []

        for violation in violations:

            violation = violation.view_as(center)

            isViolationAlreadyTruncated = False
            for C_test, d_test in zip(C_list, d_list):
                if (C_test * violation).sum() > d_test:
                    isViolationAlreadyTruncated = True
                    break

            if isViolationAlreadyTruncated:
                continue

            violation_direction = violation - center

            # Static weighted interpolation
            distances = [z.largest_value_in_direction(violation_direction, local=False).item()
                         for z in relaxations]

            d_violation = (violation_direction *
                           violation).sum()

            C = violation_direction
            d_largest = torch.Tensor([max(distances)]).view(-1)

            d = d_largest + interpolation_weight * \
                (d_violation - d_largest)

            C_list.append(C)
            d_list.append(d)

        C_all = torch.cat(C_list, 0)
        d_all = torch.Tensor(d_list)
        return C_all, d_all

    def _merge_templates(self, cluster_set, use_hyperplanes, layer):

        cluster_centers = [c['union'].a0 for idx, c in cluster_set.items()
                           if c['union'] is not None]
        cluster_centers = torch.cat(cluster_centers, 0)

        cluster_lookup_indices = [idx for idx, c in cluster_set.items()
                                  if c['union'] is not None]

        def l2_distance(x, y):
            x_norm = (x**2).sum(1).view(-1, 1)
            y_norm = (y**2).sum(1).view(1, -1)
            dist = x_norm + y_norm - 2.0 * \
                torch.mm(x, torch.transpose(y, 0, 1))

            return dist

        def indices_smallest(matrix):
            min_dim0, argmins_dim0 = matrix.min(0)
            argmin_dim1 = min_dim0.argmin().item()
            argmin_dim0 = argmins_dim0[argmin_dim1].item()
            return argmin_dim0, argmin_dim1

        max_distance = 1E4

        distance_matrix = l2_distance(cluster_centers, cluster_centers)
        distance_matrix = distance_matrix + \
            torch.eye(distance_matrix.shape[0])*max_distance

        num_clusters = distance_matrix[0, :].numel()

        while distance_matrix.min().min() < max_distance:
            num_combinations = (
                distance_matrix < max_distance).int().sum().item() // 2
            minimal_length = distance_matrix.min().min().item()
            logger.info('Num clusters left: {}, num_combinations: {}, minimal distance: {:.4f}'.format(
                num_clusters, num_combinations, minimal_length))

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

                if union.num_constraints == 0:
                    continue

                fulfilled_constraints_summary = torch.ones_like(union.d).bool()
                num_constraints_before += union.num_constraints

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

            if self.domain == 'box':
                z_union = all_relaxations[0].union(
                    all_relaxations[1:], 'box')
            elif self.domain == 'paralellotope':
                z_union = all_relaxations[0].union(
                    all_relaxations[1:], 'pca')

            for C, d in constraints_list:
                z_union.add_linear_constraints(C, d)
            num_constraints_intermediate = z_union.num_constraints

            isVerified = self._verify_with_milp(z_union, layer, all_relaxations, use_hyperplanes,
                                                max_iterations=50, interpolation_weight=0.05)

            if isVerified:
                num_constraints_after = z_union.num_constraints

                logger.info('Num constraints before/inter/after: {}/{}/{}'.format(
                    num_constraints_before, num_constraints_intermediate, num_constraints_after))

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

                num_constraints = z_union.num_constraints
                logger.info('Cluster merge: Merged clusters {} and {} with {} constraints'.format(
                    idx_cluster_overall, idy_cluster_overall, num_constraints))

                num_clusters -= 1

            else:
                distance_matrix[idx_cluster, idy_cluster] = max_distance
                distance_matrix[idy_cluster, idx_cluster] = max_distance

        return cluster_set

    def _widen_templates(self, cluster_set, use_hyperplanes, layer):
        clusters = [c for c in cluster_set.values()
                    if c['union'] is not None]
        for idx_cluster, cluster in enumerate(clusters):

            num_iterations = 20
            scaling_factor = 1.05
            isVerifiedOnce = False

            z_prev = cluster['union'].clone()
            relaxations = cluster['relaxations']

            for idx_iter in range(num_iterations):
                logger.info('{} {}'.format(idx_cluster, idx_iter))

                if idx_iter == 0:
                    interpolation_weight = 0.1
                    max_iterations = 30
                else:
                    interpolation_weight = 0.4 + 0.02*idx_iter
                    max_iterations = 10

                z_new = z_prev.clone()
                z_new.scale(scaling_factor)

                isVerified = self._verify_with_milp(z_new, layer, relaxations,
                                                    use_hyperplanes, max_iterations, interpolation_weight)

                if isVerified:
                    z_prev = z_new
                    isVerifiedOnce = True
                else:
                    break

            if isVerifiedOnce:
                cluster['union'] = z_prev

        return cluster_set

    def _store_templates(self, cluster_set, path_to_net, epsilon, use_hyperplanes,
                         layer, num_templates, use_widening):

        path = os.path.dirname(path_to_net)
        net_name = os.path.basename(path_to_net).rsplit('.')[0]
        naming = '_'.join([net_name, 'templates', str(layer), str(self.label),
                           '{:.3f}'.format(epsilon), self.domain])
        prefix = path + '/templates/' + naming
        if use_hyperplanes:
            prefix += '_star'
        if use_widening:
            prefix += '_widened'

        full_path = prefix + '.pkl'

        templates = [(c['union'], c['num_relaxations']) for c in cluster_set.values()
                     if c['union'] is not None]
        templates.sort(key=lambda x: x[1], reverse=True)
        templates = [x[0] for x in templates]

        template_dump_list = []
        for z in templates:
            template_info = (z.type, layer, self.label)
            if z.type == 'box':
                template_values = (z.a0, z.A, z.lb, z.ub, z.C, z.d)
            elif z.type == 'parallelotope':
                template_values = (z.a0, z.A, z.A_rotated_bounds,
                                   z.U_rot, z.U_rot_inv, z.lb, z.ub, z.C, z.d)
            elif z.type == 'zonotope':
                logger.error('Template domain zonotope should not happen')
                raise RuntimeError
            else:
                logger.error('Unknown template domain: '.format(z.type))
                raise RuntimeError

            template_dump_list.append((template_info, template_values))

        pickle.dump(template_dump_list, open(full_path, 'wb'))

        self.templates[layer] = templates[:num_templates]

    def load_templates(self, path_to_net, filenames, num_templates):

        path = os.path.dirname(path_to_net)

        for filename in filenames:
            full_path = path + '/templates/' + filename

            template_load_list = pickle.load(open(full_path, 'rb'))

            for template_info, template_values in template_load_list:

                template_type, layer, label = template_info

                if layer not in self.layers:
                    logger.warn('Template at layer {} not in {}'.format(
                        layer, self.layers))
                    continue
                if label is not self.label:
                    logger.warn('Template of label {} not in {}'.format(
                        label, self.label))
                    continue

                if template_type == 'box':
                    a0, A, lb, ub, C, d = template_values
                    z = Box(a0, A, (C, d), lb, ub)
                    self.templates[layer].append(z)
                elif template_type == 'parallelotope':
                    a0, A, A_rot, U, U_inv, lb, ub, C, d = template_values
                    z = Parallelotope(a0, A, (C, d), U, U_inv, A_rot)
                    z._lb = lb
                    z._ub = ub
                    self.templates[layer].append(z)
                elif template_type == 'zonotope':
                    logger.error('Template domain zonotope should not happen')
                    raise RuntimeError
                else:
                    logger.error(
                        'Unknown template domain: '.format(template_type))
                    raise RuntimeError

        for layer in self.templates.keys():
            self.templates[layer] = self.templates[layer][:num_templates]
        logger.info('Loaded templates per layer: {}'.format(
            {i: len(x) for i, x in self.templates.items()}))
        logger.info('Number of halfspace constraints per layer: {}'.format(
            {i: sum([y.num_constraints for y in x]) for i, x in self.templates.items()}))

    def submatching(self, z, layer):
        assert layer in self.layers

        for t in self.templates[layer]:
            isSubmatch = t.submatching(z)
            if isSubmatch:
                return True
        return False
