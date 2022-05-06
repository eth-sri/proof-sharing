import torch
import networks
import scipy
from scipy import spatial, linalg  # noqa: F401
import numpy as np
# import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools
import logging

import gurobipy as gp
from gurobipy import GRB

logger = logging.getLogger()


class Zonotope:
    def __init__(self, a0=None, A=None, constraints=None):
        self._a0 = a0
        self._A = A
        self._lb = None
        self._ub = None

        self._C = None
        self._d = None
        if constraints is not None:
            self.add_linear_constraints(constraints[0], constraints[1])

    @property
    def type(self):
        return 'zonotope'

    @property
    def a0(self):
        return self._a0

    @property
    def A(self):
        return self._A

    @property
    def A_rotated_bounds(self):
        raise NotImplementedError
        return self._A_rotated_bounds

    @property
    def num_dimensions(self):
        if self._a0 is not None:
            return self.a0.numel()
        else:
            return self.lb.numel()

    @property
    def num_epsilons(self):
        return self.A.shape[0]

    @property
    def U_rot(self):
        raise NotImplementedError
        return self._U_rot

    @property
    def U_rot_inv(self):
        raise NotImplementedError
        return self._U_rot_inv

    @property
    def a0_flat(self):
        if len(self.a0.shape) == 2:
            return self.a0
        else:
            return self.a0.view(1, -1)

    @property
    def A_flat(self):
        if len(self.A.shape) == 2:
            return self.A
        else:
            return self.A.view(self.A.shape[0], -1)

    @property
    def lb(self):
        if self._lb is None and self._a0 is not None:
            self.update_bounds()
        return self._lb

    @property
    def ub(self):
        if self._ub is None and self._a0 is not None:
            self.update_bounds()
        return self._ub

    def update_bounds(self):
        A_abs = self.A.abs().sum(0, keepdims=True)
        self._lb = self.a0 - A_abs
        self._ub = self.a0 + A_abs

    @property
    def C(self):
        return self._C

    @property
    def d(self):
        return self._d

    @property
    def num_constraints(self):
        if self.d is not None:
            return self._d.numel()
        else:
            return 0

    def init_from_bounds(self, lower_bound, upper_bound, U_rot=None, U_rot_inv=None,
                         constraints=None):

        a0_new_rot = (upper_bound + lower_bound) / 2.0
        A_rot = (upper_bound - lower_bound) / 2.0
        a0_new_rot = a0_new_rot.view((1, -1))

        if U_rot_inv is not None:
            a0_new = a0_new_rot.matmul(U_rot_inv)
            A_new = torch.diag(A_rot.squeeze(0)).matmul(U_rot_inv)
        else:
            a0_new = a0_new_rot
            A_new = torch.diag(A_rot.squeeze(0))

        self._a0 = a0_new
        self._A = A_new
        self._U_rot = U_rot
        self._U_rot_inv = U_rot_inv
        self._A_rotated_bounds = A_rot

        if constraints is not None:
            self._C = constraints[0]
            self._d = constraints[1]

    def submatching(self, other):

        if (self.lb > other.lb).any() or (self.ub < other.ub).any():
            return False

        raise NotImplementedError

    def get_bounds(self, detach=True):

        if detach:
            return self.lb.detach(), self.ub.detach()
        else:
            return self.lb, self.ub

    def to_box(self):
        return Box(self.a0, constraints=(self.C, self.d), lb=self.lb, ub=self.ub)

    def to_parallelotope(self):
        return self.order_reduction()

    def to_box_star(self):
        raise NotImplementedError
        lower_bound, upper_bound = self.get_bounds()
        b = Box_Star(lower_bound, upper_bound)
        return b

    def plot(self, color='b'):
        assert self.num_dimensions == 2
        C, d = self.get_halfspace_representation()

        Cd = np.concatenate((C, -d), 1)
        interior_point = self.a0.detach().numpy()[0]

        hs = scipy.spatial.HalfspaceIntersection(
            Cd.astype('double'), interior_point.astype('double'))
        vertices = hs.intersections.astype('float')

        # Sort vertices
        vertices_centered = vertices - interior_point
        angles = np.arctan2(vertices_centered[:, 0], vertices_centered[:, 1])
        idx_vertices = np.argsort(angles)
        # idx_vertices = np.append(idx_vertices, idx_vertices[0])
        vertices = vertices[idx_vertices, :]

        patch = patches.Polygon(vertices,
                                fill=False, edgecolor=color, linewidth=2.5)

        return patch

    def get_halfspace_representation(self):
        # Source: Matthias Althoff:
        # On Computing the Minkowski Difference of Zonotopes
        assert self.num_dimensions == 2
        A_np = self.A.detach().numpy().T
        a0_np = self.a0.detach().numpy().T

        C_plus = np.concatenate((A_np[[1], :], - A_np[[0], :]), 0)
        C_plus_norm = np.linalg.norm(C_plus, 2, 0, keepdims=True)
        C_plus_keep = (C_plus_norm > 1E-6)[0]
        C_plus = C_plus[:, C_plus_keep] / C_plus_norm[:, C_plus_keep]
        C = np.concatenate((C_plus, -C_plus), 1).T

        d_delta = np.sum(np.abs(np.matmul(C_plus.T, A_np)), 1, keepdims=True)
        d_plus = np.matmul(C_plus.T, a0_np) + d_delta
        d_minus = -d_plus + 2*d_delta
        d = np.concatenate((d_plus, d_minus), 0)

        return C, d

    def union(self, other, method='pca', **params):
        methods = {
            'box': self.union_box,
            'pca': self.union_pca,
            'nullspace': self.union_nullspace,
            'component_wise': self.union_component_wise,
            'global': self.union_global,
        }

        if method not in methods:
            logger.error('Choose valid union method:', list(methods.keys()))
        else:
            func = methods[method]

        return func(other, **params)

    def union_global(self, other, order_reduction=False, **_):
        # Source: Eric Goubault, Tristan Le Gall and Sylvie Putot:
        # 'An Accurate Join for Zonotopes, Preserving Affine Input/Output Relations'

        self.balance_num_epsilons([other])

        num_eps = self.num_epsilons
        num_dim = self.num_dimensions
        shape_zonotope = list(self.a0.shape[:])

        self_a0_flat, self_A_flat = self.a0_flat, self.A_flat
        other_a0_flat, other_A_flat = other.a0_flat, other.A_flat

        self_zonotope_ensemble = torch.cat((self_a0_flat, self_A_flat), 0)
        other_zonotope_ensemble = torch.cat((other_a0_flat, other_A_flat), 0)

        difference_zonotope_np = (
            self_zonotope_ensemble - other_zonotope_ensemble).detach().numpy()

        affine_relations = torch.tensor(
            scipy.linalg.null_space(difference_zonotope_np).T)

        num_affine_relations = affine_relations.shape[0]
        assert(num_affine_relations < num_dim)

        if num_affine_relations == 0:
            z_new = self.union_component_wise(other)
        else:
            z_new = Zonotope()

            M = torch.cat(
                (affine_relations,
                 -affine_relations.matmul(self_zonotope_ensemble.transpose(1, 0))), 1)

            PL_np, U_np = scipy.linalg.lu(
                M[:, :num_affine_relations].detach().numpy(), permute_l=True)

            M_d = torch.Tensor(U_np)
            M_dd = torch.Tensor(PL_np).inverse().matmul(
                M[:, num_affine_relations:])
            M = torch.cat((M_d, M_dd), 1)

            z_new._a0 = torch.zeros((1, num_dim))
            z_new._A = torch.zeros((num_dim + num_eps - num_affine_relations,
                                    num_dim))

            # Apply component_wise union for dimensions > num_affine_relations
            z_self_non_affine = Zonotope(self_a0_flat[:, num_affine_relations:],
                                         self_A_flat[:, num_affine_relations:])
            z_other_non_affine = Zonotope(other_a0_flat[:, num_affine_relations:],
                                          other_A_flat[:, num_affine_relations:])
            z_new_non_affine = z_self_non_affine.union_component_wise(
                z_other_non_affine)

            z_new._a0[:, num_affine_relations:] = z_new_non_affine.a0
            z_new._A[:, num_affine_relations:] = z_new_non_affine.A

            # Plug in affine relations for dimensions <= num_affine_relations
            for i in range(num_affine_relations-1, -1, -1):
                weights = torch.cat(
                    (z_new.a0[:, i+1:num_dim], z_new.A[:, i+1:num_dim]), 0)
                M_i = M[i, i]
                M_x = M[[i], i+1:num_dim].transpose(1, 0)
                M_eps = torch.cat((M[[i], num_dim:],
                                   torch.zeros((1, num_dim - num_affine_relations))),
                                  1).transpose(1, 0)

                R_i = (weights.matmul(M_x) + M_eps) / (-M_i)

                z_new._a0[0, i] = R_i[0, 0]
                z_new._A[:, i] = R_i[1:, 0]

            z_new._a0.resize_(shape_zonotope)
            shape_zonotope[0] = z_new.A.shape[0]
            z_new._A.resize_(shape_zonotope)

        self.remove_zero_epsilons()
        other.remove_zero_epsilons()

        if order_reduction:
            z_new = z_new.order_reduction()

        return z_new

    def union_component_wise(self, other, order_reduction=False, **_):
        # Source: Eric Goubault, Tristan Le Gall and Sylvie Putot:
        # 'An Accurate Join for Zonotopes, Preserving Affine Input/Output Relations'

        z_new = Zonotope()

        self.balance_num_epsilons([other])

        union_lb = torch.min(self.lb, other.lb)
        union_up = torch.max(self.ub, other.ub)

        z_new._a0 = 0.5 * (union_up + union_lb)
        previous_epsilons = torch.min(self.A.abs(), other.A.abs()) \
            * self.A.sign() \
            * (self.A * other.A > 0.0).float()
        new_epsilons = 0.5 * (union_up - union_lb) \
            - previous_epsilons.abs().sum(0, keepdims=True)

        shape_new_epsilons = [new_epsilons.numel()]
        shape_new_epsilons.extend(new_epsilons.shape[1:])

        new_epsilons_diag = torch.diag(
            new_epsilons.view(-1)).view(shape_new_epsilons)

        z_new._A = torch.cat((previous_epsilons, new_epsilons_diag), 0)

        self.remove_zero_epsilons()
        other.remove_zero_epsilons()

        if order_reduction:
            z_new = z_new.order_reduction()

        return z_new

    def union_box(self, other, **_):
        lower_bound, upper_bound = self.lb.clone(), self.ub.clone()

        if not isinstance(other, list):
            other = [other]

        for z in other:
            lower_bound = torch.min(lower_bound, z.lb)
            upper_bound = torch.max(upper_bound, z.ub)

        return Box(lb=lower_bound, ub=upper_bound)

    def union_pca(self, other, order_reduction=True, min_width_index_factor=0,
                  include_center=False, avoid_center_rotation=False,
                  use_center_outer_box=False, **_):

        z_new_list = []
        include_center_selection = [False, True] if include_center is None \
            else [include_center]

        if not isinstance(other, list):
            other = [other]

        for include_center in include_center_selection:
            U = self.get_principle_axis(
                other, include_center, use_center_outer_box)
            z_new = self.union_in_rotated_axis(
                U, U.transpose(1, 0), other, order_reduction,
                min_width_index_factor, avoid_center_rotation)

            z_new_list.append(z_new)

        volumes = [x.get_approximate_volume() for x in z_new_list]
        argmin = volumes.index(min(volumes))

        z_new = z_new_list[argmin]
        z_new.reshape_as(self)
        return z_new

    def union_nullspace(self, other, U, U_inv, order_reduction=True, **_):
        if not isinstance(other, list):
            other = [other]

        z_new = self.union_in_rotated_axis(
            U, U_inv, other, order_reduction)

        z_new.reshape_as(self)
        return z_new

    def order_reduction(self, U=None, U_inv=None):
        # Source: Anna-Kathrin Kopetzki, Bastian Schurmann, and Matthias Althoff:
        # 'Methods for Order Reduction of Zonotopes'

        include_center = False
        order_reduction = True
        min_width_index_factor = 0
        avoid_center_rotation = True
        use_center_outer_box = False

        if U is None:
            U = self.get_principle_axis(
                [], include_center, use_center_outer_box)

            U_inv = U.transpose(1, 0)

        z_new = self.union_in_rotated_axis(
            U, U_inv, [], order_reduction,
            min_width_index_factor, avoid_center_rotation)

        z_new.reshape_as(self)
        z_new.add_linear_constraints(self.C, self.d)

        return z_new

    def get_principle_axis(self, others=[], include_center=False,
                           use_center_outer_box=False):

        A_list = [self.A_flat]
        a0_list = [self.a0_flat]

        lower_bound, upper_bound = self.get_bounds(detach=False)

        for other in others:
            A_list.append(other.A_flat)
            a0_list.append(other.a0_flat)

            bounds = other.get_bounds(detach=False)
            lower_bound = torch.min(lower_bound, bounds[0])
            upper_bound = torch.max(upper_bound, bounds[1])

        lower_bound = lower_bound.flatten(1)
        upper_bound = upper_bound.flatten(1)

        active_neurons = upper_bound[0, :] > lower_bound[0, :]
        upper_bound = upper_bound[:, active_neurons]
        lower_bound = lower_bound[:, active_neurons]
        a0_list = [x[:, active_neurons] for x in a0_list]
        A_list = [x[:, active_neurons] for x in A_list]

        if include_center:

            if use_center_outer_box:
                a0_new = (upper_bound + lower_bound) / 2
                a0_offset = [x - a0_new for x in a0_list]
            else:
                a0_center = torch.zeros_like(a0_list[0])
                for a0 in a0_list:
                    a0_center.add_(a0)
                a0_center.div_(len(a0_list))

                a0_offset = [x - a0_center for x in a0_list]

            A_list += a0_offset

        A_cat = torch.cat(A_list, 0)

        covariance = A_cat.transpose(0, 1).matmul(A_cat)

        try:
            U_active, _, _ = torch.svd(covariance)
            U = torch.eye(self.num_dimensions)

            for idx_dim_active, idx_dim_total in enumerate(active_neurons.nonzero(as_tuple=False)):
                U[idx_dim_total, active_neurons] = U_active[idx_dim_active, :]

        except RuntimeError as err:
            U = torch.eye(self.num_dimensions)
            logger.warn('Unsuccesful SVD', err)
            logger.warn('Nan values:', torch.isnan(A_cat).any())

        return U

    def union_in_rotated_axis(self, U, U_inv, others=[], order_reduction=True,
                              min_width_index_factor=0, avoid_center_rotation=False):

        A_rotated_max = self.A_flat.matmul(U).abs_().sum(0)
        a0_rotated = self.a0_flat.matmul(U).squeeze_(0)

        lower_bound_rot = a0_rotated - A_rotated_max
        upper_bound_rot = a0_rotated + A_rotated_max

        for other in others:
            A_rotated_max = other.A_flat.matmul(U).abs_().sum(0)
            a0_rotated = other.a0_flat.matmul(U).squeeze_(0)

            lower_bound_rot = torch.min(
                lower_bound_rot, a0_rotated - A_rotated_max)
            upper_bound_rot = torch.max(
                upper_bound_rot, a0_rotated + A_rotated_max)

        a0_rotated = (lower_bound_rot + upper_bound_rot) / 2.0
        A_rotated_max = (upper_bound_rot - lower_bound_rot) / 2.0

        if avoid_center_rotation:
            lower_bound, upper_bound = self.get_bounds(detach=False)

            for other in others:
                bounds = other.get_bounds(detach=False)
                lower_bound = torch.min(lower_bound, bounds[0])
                upper_bound = torch.max(upper_bound, bounds[1])

            z_new_a0 = (upper_bound + lower_bound) / 2.0
        else:
            z_new_a0 = a0_rotated.unsqueeze(0).matmul(U_inv)

        if order_reduction:
            z_new_A = torch.diag(A_rotated_max).matmul(U_inv)

            z_new = Parallelotope(z_new_a0, z_new_A, U_rot=U, U_rot_inv=U_inv,
                                  A_rotated_bounds=A_rotated_max)
        else:
            # Apply component-wise-union
            self.balance_num_epsilons(others)

            prev_A = self.A_flat.matmul(U)
            self.remove_zero_epsilons()

            for other in others:
                other_A = other.A_flat.matmul(U)
                prev_A = torch.min(prev_A.abs(), other_A.abs()) \
                    * prev_A.sign() \
                    * (prev_A * other_A > 0.0).float()

                other.remove_zero_epsilons()

            new_A = 0.5 * (upper_bound_rot - lower_bound_rot) \
                - prev_A.abs().sum(0, keepdims=False)
            new_A = torch.diag(new_A)

            z_new_A = torch.cat([prev_A, new_A], 0).matmul(U_inv)

            z_new = Zonotope(z_new_a0, z_new_A)

        return z_new

    def widening(self, methods, **params):

        assert self.num_dimensions == self.num_epsilons
        assert self.A_rotated_bounds is not None
        assert self.U_rot is not None

        a0_new = self.a0.clone().detach_()
        A_rotated_max = self.A_rotated_bounds.clone().detach_()
        U = self.U_rot.clone().detach_()
        U_inv = self.U_rot_inv.clone().detach_()

        for method in methods:
            if method == 'set_min_width':

                min_width_index = round(
                    A_rotated_max.numel() * params['min_width_index_factor'])
                min_width_indices = A_rotated_max.sort()[0]

                if min_width_indices[min_width_index] > 0:
                    new_min_value = min_width_indices[min_width_index]
                else:
                    new_min_value = A_rotated_max.unique()[1]

                A_rotated_max[A_rotated_max < new_min_value] = new_min_value

            elif method == 'add_min_value':
                A_rotated_max.add_(params['min_value_all'])

            elif method == 'set_min_width_non_zero':
                A_rotated_max[A_rotated_max.lt(0.0)].clamp_min_(
                    params['min_value_all'])

            elif method == 'avoid_zero_width':
                A_rotated_max[A_rotated_max.eq(0.0)] = params['min_value_zero']

            elif method == 'remapping_test':
                for _ in range(params['num_remappings']):
                    z_new_A = torch.diag(A_rotated_max).matmul(
                        U_inv)
                    A_rotated_max2 = z_new_A.matmul(U).abs_().sum(0)
                    A_rotated_max = torch.max(A_rotated_max, A_rotated_max2)

            elif method == 'stretch_all':
                A_rotated_max.mul_(params['stretch_factor'])

            elif method in ['gradient_based', 'gradient_based2', 'milp_based']:
                continue

            else:
                logger.error('Unknown widening method: {}'.format(method))

        z_new_A = torch.diag(A_rotated_max).matmul(U_inv)
        z_new = self.__class__(self)
        z_new._A = z_new_A
        z_new._a0 = a0_new
        z_new.reshape_as(self)
        # z_new._U_rot = U
        # z_new._U_rot_inv = U_inv
        z_new._A_rotated_bounds = A_rotated_max

        return z_new

    def balance_num_epsilons(self, others):

        num_eps_max = self.num_epsilons
        for other in others:
            num_eps_max = max(num_eps_max, other.num_epsilons)

        self.add_zero_epsilons(num_eps_max - self.num_epsilons)

        for other in others:
            other.add_zero_epsilons(num_eps_max - other.num_epsilons)

    def add_zero_epsilons(self, num_eps):
        if num_eps > 0:

            shape = list(self.A.shape)
            shape[0] = num_eps

            self.A = torch.cat([self.A, torch.zeros(shape)], 0)
            self.A_flat = None

    def remove_zero_epsilons(self):
        self._A = self.A[self.A_flat.abs().sum(1) > 0, :]

    def scale(self, factor):

        if self._A is not None:
            self._A *= factor

        if hasattr(self, '_A_rotated_bounds'):
            if self._A_rotated_bounds is not None:
                self._A_rotated_bounds *= factor

        if self._lb is not None:
            self._lb += (self.a0 - self.lb) * (1 - factor)

        if self._ub is not None:
            self._ub += (self.a0 - self.ub) * (1 - factor)

    def reshape_as(self, other):
        shape_other = list(other.a0.shape[:])

        if self._a0 is not None:
            self._a0 = self.a0.view(shape_other)

        if self._lb is not None:
            self._lb = self.lb.view(shape_other)

        if self._ub is not None:
            self._ub = self.ub.view(shape_other)

        if self._A is not None:
            shape_other[0] = self.A.shape[0]
            self._A = self.A.view(shape_other)

    def clone(self):

        attributes = ['_a0', '_A', '_A_rotated_bounds', '_lb', '_ub',
                      '_U_rot', '_U_rot_inv', '_C', '_d']

        z = self.__class__()

        for att in attributes:
            if hasattr(self, att):
                self_att = getattr(self, att)
                if self_att is not None:
                    setattr(z, att, self_att.clone())

        return z

    def halfspace_contraction(self, halfspace_condition):
        # Halfspace conditions are defined as C*eps <= d
        # Source: Stanley Bak, Hoang-Dung Tran, Kerianne Hobbs and Taylor T. Johnson:
        # Improved Geometric Path Enumeration for Verifying ReLU Neural Networks

        new_zonotopes = []
        C, d = halfspace_condition

        for sign in [1, -1]:
            a0_new = self.a0.clone()
            A_new = self.A.clone()

            C_new = sign * C
            d_new = sign * d

            eps_far_away = -C_new.sign()

            max_diff = d_new - (C_new * eps_far_away).sum()

            contraction_counter = 0

            for idx_eps in range(self.num_epsilons):
                if C_new[idx_eps] == 0:
                    continue

                eps_i_intersect = max_diff / \
                    C_new[idx_eps] + eps_far_away[idx_eps]

                if eps_i_intersect.abs() < 1.0:
                    shift = 0.5*(eps_far_away[idx_eps] + eps_i_intersect)

                    # max_diff += C_new[i] * (eps_far_away[i] - eps_i_intersect)
                    a0_new += shift * A_new[idx_eps]
                    A_new[idx_eps] = (1-shift.abs()) * A_new[idx_eps]

                    contraction_counter += 1

            z_new = Zonotope(a0_new, A_new)

            if contraction_counter > 0:
                # logger.info('Contraction counter:', contraction_counter)
                new_zonotopes.append(z_new)

        return new_zonotopes

    def get_exact_volume(self):

        if self.num_epsilons > 10:
            logger.warn(
                'Warning, many error terms for exact volume: {}'.format(self.num_epsilons))

        A_reduced_flat = 2 * self.A_flat

        volume = 0.0
        for error_terms in itertools.combinations(range(self.num_epsilons), self.num_dimensions):
            volume += A_reduced_flat[error_terms, :].det().abs_().item()

        return volume

    def get_approximate_volume(self):
        if self.num_dimensions > self.num_epsilons:
            return 0.0
        elif self.num_dimensions == self.num_epsilons:
            A_log_sum = torch.log(2*self.A_rotated_bounds).sum()

            if A_log_sum.exp() > 0:
                return A_log_sum.exp().item()
            else:
                return A_log_sum.item()
        else:
            return self.to_parallelotope().get_exact_volume()

    def add_linear_constraints(self, C, d):

        if C is not None and d is not None:
            num_constraints = d.numel()

            if num_constraints > 0:

                C = C.view([num_constraints, -1])
                d = d.view(-1)

                assert C.shape[1] == self.num_dimensions

                if self.num_constraints > 0:
                    self._C = torch.cat([self.C, C], 0)
                    self._d = torch.cat([self.d, d], -1)
                else:
                    self._C = C
                    self._d = d

    def check_constraints(self, other, return_fulfilled_constraints=False,
                          check_intersection=False):
        # Constraints are given in the way C*x <= d
        # C has shape (num_constraints, num_neurons)
        # d has shape (num_constraints)
        if self.num_constraints == 0:
            return True

        CA_values = self.C.matmul(other.A_flat.transpose(1, 0)).abs().sum(1)
        Ca0_values = (other.a0_flat * self.C).sum(1)
        values = CA_values + Ca0_values

        isAllConstraintsFulfilled = values <= self.d

        isFulfilled = isAllConstraintsFulfilled.all().item()

        if check_intersection:
            isIntersection = Ca0_values - CA_values <= self.d
            isIntersection *= torch.bitwise_not(isAllConstraintsFulfilled)

        if return_fulfilled_constraints:

            if check_intersection:
                return isFulfilled, isAllConstraintsFulfilled, isIntersection
            else:
                return isFulfilled, isAllConstraintsFulfilled
        else:
            return isFulfilled

    def largest_value_in_direction(self, n, local=False, return_point=False):

        # Get the epsilon values, such that the generators point in
        # the direction of the normal vector n
        n = n.view(1, -1)
        nA_values = n.matmul(self.A_flat.transpose(1, 0))

        # Calculate largest value in direction of normal vector of constraint
        if local:
            largest_value = nA_values.abs().sum(1)
        else:
            na0_values = (self.a0_flat * n).sum(1)
            largest_value = nA_values.abs().sum(1) - na0_values

        if return_point:
            point = self.a0_flat + nA_values.sign().matmul(self.A_flat)
            return largest_value, point
        else:
            return largest_value


class Parallelotope(Zonotope):

    def __init__(self, a0=None, A=None, constraints=None,
                 U_rot=None, U_rot_inv=None, A_rotated_bounds=None):
        self._a0 = a0
        self._A = A
        self._U_rot = U_rot
        self._U_rot_inv = U_rot_inv
        self._A_rotated_bounds = A_rotated_bounds
        self._lb = None
        self._ub = None

        self._C = None
        self._d = None
        if constraints is not None:
            self.add_linear_constraints(constraints[0], constraints[1])

    @property
    def type(self):
        return 'parallelotope'

    @property
    def A_rotated_bounds(self):
        return self._A_rotated_bounds

    @property
    def U_rot(self):
        return self._U_rot

    @property
    def U_rot_inv(self):
        return self._U_rot_inv

    def submatching(self, other):

        if isinstance(other, torch.Tensor):
            # Submatching Batch of Single Points
            assert other.numel() // other.shape[0] == self.num_dimensions

            points_offset = other - self.a0
            points_rot = points_offset.matmul(self.U_rot)
            points_inside_box = points_rot.abs() <= self.A_rotated_bounds
            points_inside_box = points_inside_box.all(1)

            if self.num_constraints:
                return points_inside_box
            else:
                points_fulfill_constraints = other.matmul(
                    self.C.transpose(0, 1)) < self.d

                points_inside = points_inside_box * \
                    points_fulfill_constraints.all(1)
            return points_inside
        else:
            # Submatching Abstract Representations
            assert self.num_dimensions == other.num_dimensions
            assert self.U_rot is not None

            if (self.lb > other.lb).any() or (self.ub < other.ub).any():
                return False

            B = torch.cat([other.A_flat, other.a0_flat - self.a0_flat], 0)
            B_rotated_bounds = B.matmul(self.U_rot).abs().sum(0)
            outside_rectangle = (
                B_rotated_bounds > self.A_rotated_bounds).any()

            if outside_rectangle:
                return False
            else:
                return self.check_constraints(other)

    def get_exact_volume(self):
        A_log_sum = torch.log(2*self.A_rotated_bounds).sum()

        if A_log_sum.exp() > 0:
            return A_log_sum.exp().item()
        else:
            return A_log_sum.item()

    def get_approximate_volume(self):
        return self.get_exact_volume()


class Box(Zonotope):
    def __init__(self, a0=None, A=None, constraints=None,
                 lb=None, ub=None):

        self._a0 = a0
        self._A = A
        self._lb = lb
        self._ub = ub

        self._C = None
        self._d = None
        if constraints is not None:
            self.add_linear_constraints(constraints[0], constraints[1])

    @property
    def type(self):
        return 'box'

    @property
    def a0(self):
        if self._a0 is None:
            self._a0 = 0.5*(self.lb + self.ub)
        return self._a0

    @property
    def A(self):
        if self._A is None:
            shapes = [-1] + list(self.lb.shape)[1:]

            bounds_difference = ((self.ub - self.lb)/2).view(-1)
            A = torch.diag(bounds_difference)

            isDimensionWide = bounds_difference > 0
            if not isDimensionWide.all():
                A = A[isDimensionWide, :]
            self._A = A.view(shapes)

        return self._A

    @property
    def lb(self):
        return self._lb

    @property
    def ub(self):
        return self._ub

    @property
    def num_epsilons(self):
        raise NotImplementedError
        return self.A.shape[0]

    def submatching(self, other):
        if (self.lb > other.a0).any() or (self.ub < other.a0).any():
            return False

        isInsideBox = (self.lb <= other.lb).all() * (self.ub >= other.ub).all()

        if not isInsideBox:
            return False
        elif self.C is None:
            return True
        else:
            return self.check_constraints(other)

    def plot(self, color='b'):
        assert self.num_dimensions == 2

        lower_corner_np = self.lb.detach().numpy()[0]
        upper_corner_np = self.ub.detach().numpy()[0]

        width = upper_corner_np[0] - lower_corner_np[0]
        height = upper_corner_np[1] - lower_corner_np[1]

        patch = patches.Rectangle(lower_corner_np, width, height,
                                  fill=False, edgecolor=color, linewidth=2.5)

        return patch

    def get_exact_volume(self):
        A_log_sum = torch.log(self.ub - self.lb).sum()

        if A_log_sum.exp() > 0:
            return A_log_sum.exp().item()
        else:
            return A_log_sum.item()

    def get_approximate_volume(self):
        return self.get_exact_volume()


class Star(Zonotope):
    # This is a class for zonotopes with additional linear constraints
    def __init__(self, a0=None, A=None, constraints=None):

        if constraints is None:
            self.C = None
            self.d = None
        else:
            self.C = constraints[0]
            self.d = constraints[1]

        if isinstance(a0, Star):
            s = a0
            self.a0 = s.a0
            self.A = s.A
            self.U_rot = s.U_rot
            self.U_rot_inv = s.U_rot_inv
            self.A_rotated_bounds = s.A_rotated_bounds
            self.C = s.C
            self.d = s.d

        elif isinstance(a0, Zonotope):
            z = a0
            super(Star, self).__init__(z.a0, z.A)

            self.A_rotated_bounds = z.A_rotated_bounds
            self.U_rot = z.U_rot
            self.U_rot_inv = z.U_rot_inv

        else:
            super(Star, self).__init__(a0, A)

    def init_from_bounds(self, lower_bound, upper_bound, U_rot=None,
                         U_rot_inv=None, constraints=None):
        super(Star, self).init_from_bounds(
            lower_bound, upper_bound, U_rot, U_rot_inv)

        if constraints is not None:
            self.C = constraints[0]
            self.d = constraints[1]

    def submatching(self, other, method='parallelotope'):
        if ((self.a0 - other.a0).matmul(self.U_rot).abs() > self.A_rotated_bounds).any():
            return False
        isInsideBox = super(Star, self).submatching(other, method)

        if not isInsideBox:
            return False
        elif self.C is None:
            return True
        else:
            return self.check_constraints(other)

    def submatching_single(self, points):
        points_offset = points - self.a0
        points_rot = points_offset.matmul(self.U_rot)
        points_inside_box = points_rot.abs() <= self.A_rotated_bounds
        points_inside_box = points_inside_box.all(1)

        if self.C is None:
            return points_inside_box
        else:

            points_fulfill_constraints = points.matmul(
                self.C.transpose(0, 1)) < self.d

            points_inside = points_inside_box * \
                points_fulfill_constraints.all(1)
            return points_inside

    def union(self, other, method='pca', **params):
        z_new = super(Star, self).union(other, method, **params)

        s_new = Star(z_new)
        C_new = self.C
        d_new = self.d

        if C_new is None:
            if isinstance(other, Star):
                C_new = other.C
                d_new = other.d
        else:
            if isinstance(other, Star):
                if other.C is not None:
                    C_new = torch.cat([C_new, other.C], 0)
                    d_new = torch.cat([d_new, other.d])

        s_new.C = C_new
        s_new.d = d_new

        return s_new

    def check_constraints(self, other, return_fulfilled_constraints=False,
                          check_intersection=False):
        # Constraints are given in the way C*x <= d
        # C has shape (num_constraints, num_neurons)
        # d has shape (num_constraints)
        if self.num_constraints == 0:
            return True

        CA_values = self.C.matmul(other.A_flat.transpose(1, 0)).abs().sum(1)
        Ca0_values = (other.a0_flat * self.C).sum(1)
        values = CA_values + Ca0_values

        isAllConstraintsFulfilled = values <= self.d

        isFulfilled = isAllConstraintsFulfilled.all().item()

        if check_intersection:
            isIntersection = Ca0_values - CA_values <= self.d
            isIntersection *= torch.bitwise_not(isAllConstraintsFulfilled)

        if return_fulfilled_constraints:

            if check_intersection:
                return isFulfilled, isAllConstraintsFulfilled, isIntersection
            else:
                return isFulfilled, isAllConstraintsFulfilled
        else:
            return isFulfilled

    def add_linear_constraints(self, C, d):

        if C is not None:
            if d.numel() > 0:
                num_constraints = d.numel()
                C = C.view([num_constraints, -1])
                d = d.view(-1)

                if self.C is None:
                    self.C = C
                    self.d = d
                else:
                    self.C = torch.cat([self.C, C], 0)
                    self.d = torch.cat([self.d, d], -1)

        return self


class Box_Star():
    # This is a class for boxes with additional linear constraints
    def __init__(self, lb=None, ub=None, constraints=None):

        if isinstance(lb, self.__class__):
            self.lb = lb.lb
            self.ub = lb.ub
            self.C = lb.C
            self.d = lb.d
            return

        self.lb = lb
        self.ub = ub
        if lb is not None and ub is not None:
            self.a0 = (lb + ub) / 2.0
            self.A = torch.diag(((ub-lb)/2).view(-1))
        else:
            self.a0 = None
            self.A = None

        if constraints is None:
            self.C = None
            self.d = None
        else:
            self.C = constraints[0]
            self.d = constraints[1]

        self.U_rot = None
        self.U_rot_inv = None

    def init_from_bounds(self, lower_bound, upper_bound, U_rot=None,
                         U_rot_inv=None, constraints=None):

        self.lb = lower_bound
        self.ub = upper_bound

        self.a0 = (lower_bound + upper_bound) / 2.0
        self.A = torch.diag(((upper_bound-lower_bound)/2).view(-1))

        if constraints is not None:
            self.C = constraints[0]
            self.d = constraints[1]

    def submatching(self, other, *_):
        if (self.lb > other.a0).any() or (self.ub < other.a0).any():
            return False
        other_lb, other_ub = other.get_bounds()

        isInsideBox = (self.lb <= other_lb).all() * (self.ub >= other_ub).all()

        if not isInsideBox:
            return False
        elif self.C is None:
            return True
        else:
            return self.check_constraints(other)

    def submatching_single(self, points):
        points_inside_box = (self.lb <= points) * (self.ub >= points)
        points_inside_box = points_inside_box.all(1)

        if self.C is None:
            return points_inside_box
        else:

            points_fulfill_constraints = points.matmul(
                self.C.transpose(0, 1)) < self.d

            points_inside = points_inside_box * \
                points_fulfill_constraints.all(1)

            return points_inside

    def get_bounds(self, detach=True):

        if detach:
            return self.lb.detach(), self.ub.detach()
        else:
            return self.lb, self.ub

    def get_dimensions_flat(self):
        return self.lb.numel(), self.lb.numel()

    def get_relaxation_flat(self, detach=True):

        raise NotImplementedError

    def reshape_as(self, other):
        shape_other = list(other.a0.shape[:])
        self.a0 = self.a0.view(shape_other)
        self.lb = self.lb.view(shape_other)
        self.ub = self.ub.view(shape_other)

        shape_other[0] = self.A.shape[0]
        self.A = self.A.view(shape_other)


class Zonotope_Net:

    def __init__(self, net, device='cpu', relu_transformer='zonotope'):
        self.device = device

        self.relaxation_type = Zonotope
        self.relu_transformer = relu_transformer
        self.relaxation_at_layers = []
        self.lambdas_list = {}
        self.is_lambda_optimizable = False
        self.net = net
        self.remove_gradient_of_net()

    def process_input_iteratively(self, inputs, eps, true_label, label_maximization=True,
                                  num_steps=1, step_duration=20):
        self.is_lambda_optimizable = True
        isVerified = self.process_input_once(
            inputs, eps, true_label, label_maximization)

        if isVerified:
            return True
        else:
            start_layer = min(self.lambdas_list.keys())
            self.truncate(start_layer)

            return self.optimize_lambdas_iteratively(true_label, start_layer, label_maximization,
                                                     num_steps=num_steps, step_duration=step_duration)

    def process_input_once(self, inputs, eps, true_label, label_maximization=True):

        self.initialize(inputs, eps)

        self.forward_pass()

        return self.calculate_worst_case(true_label, label_maximization)

    def process_input_recursively(self, inputs, eps, true_label, label_maximization=True,
                                  split_max_depth=0, start_layer=0, U_rot_inv=None, constraints=None):

        lower_bound, upper_bound = self.initialize(
            inputs, eps, gradients_for_bounds=True, U_rot_inv=U_rot_inv)

        self.forward_pass()

        self.forward_pass(start_layer)
        isVerified = self.calculate_worst_case(true_label, label_maximization)

        if isVerified:
            return True

        else:
            loss = self.get_verification_loss(true_label, label_maximization)
            loss.backward()

            splits = self.generate_splits(lower_bound, upper_bound, 200)

            num_splits = len(splits)
            num_verified = 0

            for lower_bound, upper_bound in splits:
                isVerified = self.verify_recursively(
                    lower_bound, upper_bound, true_label, label_maximization, split_max_depth,
                    start_layer, U_rot_inv, constraints)

                if isVerified:
                    num_verified += 1

            logger.info(
                'Verification counter: {} / {}'.format(num_verified, num_splits))

            return num_splits == num_verified

    def process_patch_combination(self, inputs, patch_size, true_label, label_maximization=True,
                                  early_stopping=True):
        num_image_dim = inputs.shape[-1]
        num_patches_per_dimension = num_image_dim - patch_size + 1

        isVerified_All = []

        for idx in range(num_patches_per_dimension):
            for idy in range(num_patches_per_dimension):
                x_lb = inputs.clone()
                x_ub = inputs.clone()

                x_lb[:, :, idx:(idx+patch_size), idy:(idy+patch_size)] = 0
                x_ub[:, :, idx:(idx+patch_size), idy:(idy+patch_size)] = 1

                # z_net = self.__class__(self.net, self.device)
                # z_net.initialize_from_bounds(x_lb, x_ub)
                # z_net.forward_pass()
                # isVerified = z_net.calculate_worst_case(
                #     true_label, label_maximization)

                self.initialize_from_bounds(x_lb, x_ub)
                self.forward_pass()
                isVerified = self.calculate_worst_case(
                    true_label, label_maximization)

                isVerified_All.append(isVerified)

                if early_stopping and not isVerified:
                    return False

        return all(isVerified_All)

    def process_patch_sparse(self, inputs, patch_size, true_label, label_maximization=True):

        self.initialize_patch_sparse(inputs, patch_size)

        for idx_layer, layer in enumerate(self.net.layers):
            if isinstance(layer, torch.nn.ReLU):
                break

        self.forward_pass(start_layer=idx_layer+1)

        return self.calculate_worst_case(true_label, label_maximization)

    def verify_recursively(self, lower_bound, upper_bound, true_label,
                           label_maximization=True, split_max_depth=0,
                           start_layer=0, U_rot_inv=None, constraints=None):
        last_iteration = split_max_depth == 0

        lower_bound.requires_grad_(not last_iteration)
        upper_bound.requires_grad_(not last_iteration)

        self.initialize_from_bounds(
            lower_bound, upper_bound, U_rot_inv=U_rot_inv)

        self.forward_pass(start_layer)
        isVerified = self.calculate_worst_case(
            true_label, label_maximization)

        if isVerified:
            return True
        elif last_iteration:
            return False
        else:
            loss = self.get_verification_loss(true_label, label_maximization)
            loss.backward()

            split_bounds = self.generate_splits(
                lower_bound, upper_bound, 2)

            isVerified = []
            for lower_bound, upper_bound in split_bounds:

                z_net = self.__class__(self.net)

                result = z_net.verify_recursively(
                    lower_bound, upper_bound, true_label, label_maximization, split_max_depth - 1,
                    start_layer, U_rot_inv, constraints)

                isVerified.append(result)

            return all(isVerified)

    def process_from_layer(self, true_label, start_layer, lambda_optimization=False,
                           label_maximization=True, num_steps=1, step_duration=20,
                           recursion=False):

        self.is_lambda_optimizable = lambda_optimization
        length_relaxation_at_layers = len(self.relaxation_at_layers)

        self.forward_pass(start_layer)

        isVerified = self.calculate_worst_case(true_label, label_maximization)

        if isVerified:
            return True
        elif lambda_optimization:
            if len(self.lambdas_list.keys()) > 0:
                start_layer_new = min(self.lambdas_list.keys())
            else:
                return False
            truncate_position = start_layer_new - \
                start_layer + length_relaxation_at_layers - 1
            self.truncate(truncate_position)
            return self.optimize_lambdas_iteratively(true_label, start_layer_new,
                                                     label_maximization,
                                                     step_duration=step_duration,
                                                     num_steps=num_steps)
        else:
            return False

    def optimize_lambdas_iteratively(self, true_label, start_layer=0,
                                     label_maximization=True,
                                     step_duration=20, num_steps=10):
        initial_update_step = 0.02
        update_step_decay = 0.7

        for i in range(num_steps):
            optimizer = torch.optim.Adam(
                self.lambdas_list.values(), lr=initial_update_step)

            for j in range(step_duration):

                self.optimize_lambdas(
                    optimizer, true_label, label_maximization)

                self.truncate()

                self.forward_pass(start_layer)
                self.calculate_worst_case(true_label, label_maximization)

                if torch.argmax(self.y) == true_label:
                    # logger.info('num_iterations: {}'.format(
                    #     (j+1) + i * step_duration))
                    return True
                # logger.info()

            initial_update_step *= update_step_decay

        return False

    def optimize_lambdas(self, optimizer, true_label, label_maximization=True):

        loss = self.get_verification_loss(true_label, label_maximization)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for lambdas in self.lambdas_list.values():
                lambdas.clamp_(0.0, 1.0)

    def get_verification_loss(self, true_label, label_maximization=True, allow_negative_values=False):
        assert hasattr(self, 'y')

        if label_maximization:
            y_diff = self.y - self.y[0, true_label]
        else:
            y_diff = -self.y + self.y[0, true_label]

        if allow_negative_values:
            false_labels = [i for i in range(
                y_diff.numel()) if i != true_label]
            loss = y_diff[0, false_labels].max()
        else:
            loss = y_diff.clamp_min(0).sum()

        return loss

    def generate_splits(self, lower_bound, upper_bound, minimal_split_number=200):

        num_dimensions = lower_bound.numel()
        shape = list(lower_bound.shape)

        gradients = torch.max(lower_bound.grad.abs(),
                              upper_bound.grad.abs()).add(1E-10).view(-1)

        lower_bound = lower_bound.requires_grad_(False).detach().view(-1)
        upper_bound = upper_bound.requires_grad_(False).detach().view(-1)

        gap = upper_bound - lower_bound
        smears = gradients * gap

        num_splits = torch.ones_like(smears).int()
        num_splits_total = 1

        while num_splits_total < minimal_split_number:
            smears_divided = smears / num_splits
            idx_most_influence = smears_divided.argmax()
            num_splits[idx_most_influence] += 1

            num_splits_total = num_splits.prod()

        num_splits = num_splits.tolist()

        cuts = [torch.linspace(lower_bound[i], upper_bound[i], num_splits[i] + 1).tolist()
                for i in range(num_dimensions)]

        split_enumeration = [range(x) for x in num_splits]
        splits = []

        for split_numbers in itertools.product(*split_enumeration):

            lower_bound_list = [cuts[i][split_numbers[i]]
                                for i in range(num_dimensions)]
            upper_bound_list = [cuts[i][split_numbers[i] + 1]
                                for i in range(num_dimensions)]

            splits.append((torch.Tensor(lower_bound_list).reshape(shape),
                           torch.Tensor(upper_bound_list).reshape(shape)))

        return splits

    def forward_pass(self, start_layer=0):

        if start_layer < 0:
            start_layer += len(self.net.layers)

        for idx_layer in range(start_layer, len(self.net.layers)):
            self.apply_layer(idx_layer)

    def apply_layer(self, idx_layer):
        layer = self.net.layers[idx_layer]

        if isinstance(layer, networks.Normalization):
            self.apply_normalization_layer(idx_layer)
        elif isinstance(layer, torch.nn.Linear):
            self.apply_linear_layer(idx_layer)
        elif isinstance(layer, (torch.nn.Conv2d, torch.nn.AvgPool2d)):
            self.apply_convolutional_layer(idx_layer)
        elif isinstance(layer, torch.nn.Flatten):
            self.apply_flatten_layer(idx_layer)
        elif isinstance(layer, torch.nn.ReLU):
            self.apply_relu_layer(idx_layer)
        elif isinstance(layer, torch.nn.MaxPool2d):
            self.apply_maxpool_layer(idx_layer)
        else:
            logger.warn('Unknown layer: {}'.format(layer))

    def apply_normalization_layer(self, idx_layer):
        layer = self.net.layers[idx_layer]
        z = self.relaxation_at_layers[-1]
        z_new = self.relaxation_type()

        sigma = layer.sigma
        mean = layer.mean

        a0 = z.a0.sub(mean).div(sigma)
        A = z.A.div(sigma)
        z_new = self.relaxation_type(a0, A)

        self.relaxation_at_layers.append(z_new)

    def apply_linear_layer(self, idx_layer):
        layer = self.net.layers[idx_layer]
        z = self.relaxation_at_layers[-1]

        a0 = layer(z.a0)
        A = self.net.bias_free_layers[idx_layer](z.A)
        z_new = self.relaxation_type(a0, A)

        self.relaxation_at_layers.append(z_new)

    def apply_convolutional_layer(self, idx_layer):
        self.apply_linear_layer(idx_layer)

    def apply_flatten_layer(self, idx_layer):
        layer = self.net.layers[idx_layer]
        z = self.relaxation_at_layers[-1]

        a0 = layer(z.a0)
        A = layer(z.A)
        z_new = self.relaxation_type(a0, A)

        self.relaxation_at_layers.append(z_new)

    def apply_relu_layer(self, idx_layer):
        z = self.relaxation_at_layers[-1]

        lx_all, ux_all = z.get_bounds(detach=False)
        lx = lx_all[0]
        ux = ux_all[0]

        positive_neurons = lx >= 0.0
        negative_neurons = ux <= 0.0
        mixed_neurons = torch.bitwise_not(positive_neurons | negative_neurons)

        if self.relu_transformer == 'zonotope':

            lambdas_template = ux / (ux - lx + 1E-9)
            lambdas_template = lambdas_template.clamp(0, 1)
            # lambdas_template = torch.zeros_like(ux)

            # If no optimization is used, take lambdas_template
            if not self.is_lambda_optimizable:
                lambdas = lambdas_template
                lower_lambdas = torch.zeros_like(
                    lambdas_template).type(torch.BoolTensor)
            else:

                # If at first run, create new lambdas from lambdas_template
                if idx_layer not in self.lambdas_list:
                    lambdas = lambdas_template.detach().requires_grad_(True)
                    lower_lambdas = torch.zeros_like(
                        lambdas_template).type(torch.BoolTensor)

                    self.lambdas_list[idx_layer] = lambdas

                # If at consecutive run, load lambdas
                else:
                    lambdas = self.lambdas_list[idx_layer]
                    lower_lambdas = lambdas < lambdas_template

            # Separate lambdas for values, for the two cases >/< ux/(ux+lx)
            upper_lambdas = torch.bitwise_not(lower_lambdas)
            lower_lambdas = lower_lambdas & mixed_neurons
            upper_lambdas = upper_lambdas & mixed_neurons

            a0 = z.a0 * (positive_neurons + lambdas * mixed_neurons) \
                - lambdas * upper_lambdas * lx / 2.0 \
                + (1 - lambdas) * lower_lambdas * ux / 2.0
            A = z.A * (positive_neurons + lambdas * mixed_neurons)

            # Create new error terms for the mixed neurons
            n_mixed_lower = [torch.sum(lower_lambdas).item()]
            n_mixed_upper = [torch.sum(upper_lambdas).item()]
            n_mixed_lower.extend(z.A.shape[1:])
            n_mixed_upper.extend(z.A.shape[1:])

            A_ext1 = torch.zeros(n_mixed_upper, dtype=torch.float32)
            A_ext1[:, upper_lambdas] = torch.diag(
                - lambdas[upper_lambdas] * lx[upper_lambdas] / 2.0)
            A_ext2 = torch.zeros(n_mixed_lower, dtype=torch.float32)
            A_ext2[:, lower_lambdas] = torch.diag(
                - (1 - lambdas[lower_lambdas]) * ux[lower_lambdas] / 2.0)

            A = torch.cat((A, A_ext1, A_ext2), 0)

        elif self.relu_transformer == 'box':

            a0 = z.a0 * positive_neurons + ux_all * 0.5 * mixed_neurons
            A = z.A * positive_neurons

            n_mixed = [mixed_neurons.int().sum()]
            n_mixed.extend(z.A.shape[1:])

            A_ext = torch.zeros(n_mixed)
            A_ext[:, mixed_neurons] = torch.diag(ux[mixed_neurons] * 0.5)

            A = torch.cat((A, A_ext), 0)

        else:
            logger.info('Unknown ReLU transformer: {}'.format(
                self.relu_transformer))
            raise RuntimeError

        z_new = self.relaxation_type(a0, A)

        self.relaxation_at_layers.append(z_new)

    def apply_maxpool_layer(self, idx_layer):
        z = self.relaxation_at_layers[-1]

        num_eps, num_dim = self.num_epsilons, self.num_dimensions
        num_channels = z.A.shape[1]
        height = z.A.shape[2]
        width = z.A.shape[3]
        spatial_new = height * width // 4
        num_dim_new = num_dim // 4

        # * Bring into format (dim_eps, dim_output_neuron, dim_input_neuron)
        unfold = torch.nn.Unfold((2, 2), stride=2)

        A = unfold(z.A).view(
            (num_eps, num_channels, 4, spatial_new))
        A = A.transpose_(3, 2).reshape(
            (num_eps, num_channels * spatial_new, 4))

        a0 = unfold(z.a0).view(
            (1, num_channels, 4, spatial_new))
        a0 = a0.transpose_(3, 2).reshape(
            (1, num_channels * spatial_new, 4))

        # * Check if some input neurons are strictly lower (sound, but not exact)
        strictly_lower_summary = torch.zeros(
            [num_channels * spatial_new, 4], dtype=torch.bool)

        for idx_input in range(4):
            idx_other_inputs = np.arange(4) != idx_input
            a0_diff = a0[:, :, [idx_input]] - a0[:, :, idx_other_inputs]
            A_diff = A[:, :, [idx_input]] - A[:, :, idx_other_inputs]

            # 1 vs 1 check (pairwise)
            best_case_single = a0_diff + A_diff.abs().sum(0, keepdims=True) >= 0.0
            is_not_strictly_lower = best_case_single.prod(-1).bool()

            # 1 vs 2 check
            a0_diff_pair = a0_diff[:, :, [0, 1, 2]] + a0_diff[:, :, [1, 2, 0]]
            A_diff_pair = A_diff[:, :, [0, 1, 2]] + A_diff[:, :, [1, 2, 0]]

            best_case_pair = a0_diff_pair + \
                A_diff_pair.abs().sum(0, keepdims=True) >= 0.0

            is_not_strictly_lower = is_not_strictly_lower & \
                best_case_pair.prod(-1).bool()

            # 1 vs 3 check
            a0_diff_triple = a0_diff.sum(-1, keepdims=False)
            A_diff_triple = A_diff.sum(-1, keepdims=False)

            best_case_triple = a0_diff_triple + \
                A_diff_triple.abs().sum(0, keepdims=True) >= 0.0

            is_not_strictly_lower = is_not_strictly_lower & \
                best_case_triple

            strictly_lower_summary[:, idx_input] = \
                is_not_strictly_lower.logical_not()

        # * Get the common error terms for the neurons, which are not strictly lower
        keep_neurons = torch.bitwise_not(strictly_lower_summary)

        A1 = A[:, :, 0]
        A2 = A[:, :, 1]
        A3 = A[:, :, 2]
        A4 = A[:, :, 3]
        common_terms01 = torch.min(A1.abs(), A2.abs()) * \
            torch.sign(A1) * (A1 * A2 > 0.0)
        common_terms23 = torch.min(A3.abs(), A4.abs()) * \
            torch.sign(A3) * (A3 * A4 > 0.0)

        A_common01 = common_terms01 * keep_neurons[:, 0] * keep_neurons[:, 1] \
            + A[:, :, 0] * strictly_lower_summary[:, 1] \
            + A[:, :, 1] * strictly_lower_summary[:, 0]
        valid_commons01 = keep_neurons[:, 0] | keep_neurons[:, 1]

        A_common23 = common_terms23 * keep_neurons[:, 2] * keep_neurons[:, 3] \
            + A[:, :, 2] * strictly_lower_summary[:, 3] \
            + A[:, :, 3] * strictly_lower_summary[:, 2]
        valid_commons23 = keep_neurons[:, 2] | keep_neurons[:, 3]

        A_common_all = torch.min(A_common01.abs(), A_common23.abs()) * \
            torch.sign(A_common01) * (A_common01 * A_common23 > 0.0)

        A_common_all = A_common_all * valid_commons01 * valid_commons23 \
            + A_common01 * torch.bitwise_not(valid_commons23) \
            + A_common23 * torch.bitwise_not(valid_commons01)

        # * Baseline: For the non common terms, fit a box
        A_abs_sum = A.abs().sum(0, keepdims=True)
        upper_bound, _ = (a0 + A_abs_sum).max(2)
        lower_bound, _ = (a0 - A_abs_sum).max(2)

        a0_new = (upper_bound + lower_bound) * 0.5

        A_common_all_abs_sum = A_common_all.abs().sum(0, keepdims=False)

        total_error = (upper_bound - lower_bound)[0] / 2.0
        new_error_terms = total_error - A_common_all_abs_sum

        new_error_terms_needed = new_error_terms > 0.0
        num_new_error_terms = new_error_terms_needed.int().sum().item()

        A_ext = torch.zeros((num_new_error_terms, num_dim_new))
        A_ext[:, new_error_terms_needed] = torch.diag(
            new_error_terms[new_error_terms_needed])

        A_new = torch.cat((A_common_all, A_ext), 0)

        a0 = a0_new.view((1, num_channels, height // 2, width // 2))
        A = A_new.view((num_eps + num_new_error_terms,
                        num_channels, height // 2, width // 2))
        z_new = self.relaxation_type(a0, A)

        self.relaxation_at_layers.append(z_new)

    def calculate_worst_case(self, true_label, label_maximization=True):
        z = self.relaxation_at_layers[-1]

        A_diff = z.A - z.A[:, [true_label]]
        A_diff_abs = torch.sum(A_diff.abs_(), 0, keepdims=True)

        if label_maximization:
            self.y = z.a0 + A_diff_abs
            most_likely_label = torch.argmax(self.y)
        else:
            self.y = z.a0 - A_diff_abs
            most_likely_label = torch.argmin(self.y)

        return (most_likely_label == true_label).item()

    def truncate(self, layer=0):
        if len(self.relaxation_at_layers) > 1:
            self.relaxation_at_layers = [self.relaxation_at_layers[layer]]

    def initialize(self, inputs, eps, gradients_for_bounds=False, U_rot=None, U_rot_inv=None):
        # This is an overloaded function which takes and input and the error
        # (inputs, eps) as well as bounds (lower_bound, upper_bound) as input

        if isinstance(eps, torch.Tensor):
            lower_bound = inputs.clone()
            upper_bound = eps.clone()
        else:
            lower_bound, upper_bound = self.generate_input_bounds(inputs, eps)

        lower_bound.requires_grad_(gradients_for_bounds)
        upper_bound.requires_grad_(gradients_for_bounds)

        self.initialize_from_bounds(lower_bound, upper_bound, U_rot, U_rot_inv)

        return lower_bound, upper_bound

    def generate_input_bounds(self, inputs, eps, limits=(0, 1)):
        lower_bound = torch.clamp_min(inputs - eps, limits[0])
        upper_bound = torch.clamp_max(inputs + eps, limits[1])

        return lower_bound, upper_bound

    def initialize_from_bounds(self, lower_bound, upper_bound, U_rot=None, U_rot_inv=None):
        if U_rot is None:
            z_new = Box(lb=lower_bound, ub=upper_bound)
        else:
            a0_rot = (lower_bound + upper_bound) / 2

            A_rot_bounds = ((upper_bound - lower_bound)/2).view(-1)
            isDimensionWide = A_rot_bounds > 0
            A_rot = torch.diag(A_rot_bounds)
            if not isDimensionWide.all():
                A_rot = A_rot[isDimensionWide, :]

            input_size = [-1] + lower_bound.shape[1:]
            A_rot = A_rot.view(input_size)

            a0 = a0_rot.matmul(U_rot_inv)
            A = A_rot.matmul(U_rot_inv)

            z_new = Parallelotope(a0, A, U_rot=U_rot, U_rot_inv=U_rot_inv,
                                  A_rotated_bounds=A_rot_bounds)

        self.relaxation_at_layers = [z_new]

    def initialize_patch_sparse(self, inputs, patch_size):
        num_affected_pixels = patch_size * patch_size

        input_size = list(inputs.shape)
        input_size[0] = inputs.numel()

        x = inputs
        x_lb = torch.diag(-inputs.view(-1)).view(input_size)
        x_ub = 1 + x_lb

        for idx_layer, layer in enumerate(self.net.layers):
            x = layer(x)

            if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d)):
                x_lb = self.net.bias_free_layers[idx_layer](x_lb)
                x_ub = self.net.bias_free_layers[idx_layer](x_ub)

                break
            else:
                x_lb = layer(x_lb)
                x_ub = layer(x_ub)

        y_lb_add = torch.min(x_lb, x_ub)
        y_ub_add = torch.max(x_lb, x_ub)

        y_lb_add, _ = torch.sort(y_lb_add, 0, descending=False)
        y_ub_add, _ = torch.sort(y_ub_add, 0, descending=True)

        y_lb = x + y_lb_add[:num_affected_pixels].sum(0, keepdims=True)
        y_ub = x + y_ub_add[:num_affected_pixels].sum(0, keepdims=True)

        if isinstance(self.net.layers[idx_layer+1], torch.nn.ReLU):
            y_lb = self.net.layers[idx_layer+1](y_lb)
            y_ub = self.net.layers[idx_layer+1](y_ub)

        self.initialize_from_bounds(y_lb, y_ub)

    def remove_gradient_of_net(self):
        if self.net is not None:
            self.net.remove_gradient()


class Star_Net(Zonotope_Net):

    def __init__(self, net, device='cpu', relu_transformer='zonotope'):

        super(Star_Net, self).__init__(net, device, relu_transformer)

        self.relaxation_type = Zonotope
        self.milp_model = None
        self.milp_variables_at_layer = {}
        self.milp_relu_constraints = {}
        self.milp_truncation_constraints = []

        self.use_overapproximation_only = True
        self.use_relu_multiplication = True
        self.use_redundant_constraints = False
        self.use_general_zonotope = True
        self.use_tighter_bounds = False
        self.use_tighter_bounds_using_milp = False
        self.use_retightening = False
        self.use_lp = False
        self.milp_neuron_ratio = 1.0
        self.use_warm_start = False
        self.timelimit = -1
        self.early_stopping_objective = None
        self.early_stopping_bound = None
        self.num_solutions = 1
        self.intermediate_lp_model = None
        self.intermediate_milp_model = None
        self.relu_bounds = {}

    def initialize_milp_model(self, idx_layer):
        self.milp_model = gp.Model('MILP_Model')
        self.milp_model.Params.OutputFlag = 0
        self.milp_model.Params.FeasibilityTol = 1e-6
        if self.timelimit > -1:
            self.milp_model.Params.TimeLimit = self.timelimit
        if self.early_stopping_objective is not None:
            self.milp_model.Params.BestObjStop = self.early_stopping_objective
        if self.early_stopping_bound is not None:
            self.milp_model.Params.BestBdStop = self.early_stopping_bound
        self.milp_model.Params.PoolSolutions = self.num_solutions

        idx_layer_name = 'init'

        s = self.relaxation_at_layers[0]
        num_variables_new = s.num_dimensions

        bounds = s.get_bounds()
        lower_bound_np = bounds[0].flatten().numpy()
        upper_bound_np = bounds[1].flatten().numpy()

        milp_variables_new = []

        if s.type == 'zonotope':
            logger.debug('Use zonotope representation for MILP verification')
            num_epsilons = s.num_epsilons
            eps_init = []

            for k in range(num_epsilons):
                var_name = 'eps_{}[0,{}]'.format(idx_layer_name, k)
                var = self.milp_model.addVar(vtype=GRB.CONTINUOUS, lb=-1.0,
                                             ub=1.0, name=var_name)
                eps_init.append(var)

            for j in range(num_variables_new):
                var_name = 'x_{}[0,{}]'.format(idx_layer_name, j)
                if self.use_redundant_constraints:
                    var = self.milp_model.addVar(vtype=GRB.CONTINUOUS, lb=lower_bound_np[j],
                                                 ub=upper_bound_np[j], name=var_name)
                else:
                    var = self.milp_model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY,
                                                 ub=GRB.INFINITY, name=var_name)
                milp_variables_new.append(var)

            a0, A = s.a0_flat, s.A_flat
            a0_np = a0.flatten().numpy()
            A_np = A.numpy()

            for j in range(num_variables_new):
                expr = gp.LinExpr()
                expr += -1*milp_variables_new[j]
                # matmult constraints
                for k in range(num_epsilons):
                    expr.addTerms(A_np[k, j], eps_init[k])
                expr.addConstant(a0_np[j])
                self.milp_model.addConstr(expr, GRB.EQUAL, 0)

            self.milp_variables_at_layer[-2] = (eps_init,)

        elif s.type == 'parallellotope':
            logger.debug('Use rotated bounds for MILP verification')
            # Use rotated bounds
            num_epsilons = s.num_epsilons

            milp_variables_rot = []

            a0, A = s.a0_flat, s.A_flat
            a0_rot = a0.matmul(s.U_rot).flatten()
            A_rot = A.matmul(s.U_rot).abs().sum(0).flatten()
            lower_bound_rot = (a0_rot - A_rot).detach().numpy()
            upper_bound_rot = (a0_rot + A_rot).detach().numpy()

            for j in range(num_variables_new):
                var_name = 'x_rot_{}[0,{}]'.format(idx_layer_name, j)
                var = self.milp_model.addVar(vtype=GRB.CONTINUOUS, lb=lower_bound_rot[j],
                                             ub=upper_bound_rot[j], name=var_name)
                milp_variables_rot.append(var)

            for j in range(num_variables_new):
                var_name = 'x_{}[0,{}]'.format(idx_layer_name, j)
                if self.use_redundant_constraints:
                    var = self.milp_model.addVar(vtype=GRB.CONTINUOUS, lb=lower_bound_np[j],
                                                 ub=upper_bound_np[j], name=var_name)
                else:
                    var = self.milp_model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY,
                                                 ub=GRB.INFINITY, name=var_name)
                milp_variables_new.append(var)

            for j in range(num_variables_new):
                expr = gp.LinExpr()
                expr += -1*milp_variables_new[j]
                # matmult constraints
                for k in range(num_variables_new):
                    expr.addTerms(
                        s.U_rot_inv[k, j].numpy(), milp_variables_rot[k])
                # expr.addConstant(a0_np[j])
                self.milp_model.addConstr(expr, GRB.EQUAL, 0)

            self.milp_variables_at_layer[-2] = (milp_variables_rot,)

        else:
            logger.debug('Use non-rotated bounds for MILP verification')
            # assert(num_epsilons == num_variables_new)

            for j in range(num_variables_new):
                var_name = 'x_{}[0,{}]'.format(idx_layer_name, j)
                var = self.milp_model.addVar(vtype=GRB.CONTINUOUS, lb=lower_bound_np[j],
                                             ub=upper_bound_np[j], name=var_name)
                milp_variables_new.append(var)

        self.milp_variables_at_layer[idx_layer] = (milp_variables_new,)
        # logger.debug(type(self.milp_variables_at_layer),
        #       type(self.milp_variables_at_layer[idx_layer]),
        #       type(self.milp_variables_at_layer[idx_layer][0]),
        #       self.milp_variables_at_layer[idx_layer][0][0])

    def add_input_constraints(self):

        s = self.relaxation_at_layers[0]
        # a0_np = s.a0.detach().flatten(1).numpy()
        # A_np = s.A.detach().flatten(1).numpy()

        self.milp_model.update()
        milp_variables_init = [
            v for v in self.milp_model.getVars() if 'x_init' in v.VarName]
        num_dimensions = len(milp_variables_init)

        # add additional linear constraint on the neurons
        if s.num_constraints > 0:
            C = s.C.numpy()
            d = s.d.numpy()
            num_constraints = s.d.numel()

            for i in range(num_constraints):
                expr = gp.LinExpr()
                for j in range(num_dimensions):
                    expr.addTerms(C[i, j], milp_variables_init[j])

                self.milp_model.addConstr(expr, GRB.LESS_EQUAL, d[i])

        self.milp_model.update()

    def initialize_from_bounds(self, lower_bound, upper_bound, U_rot=None, U_rot_inv=None, constraints=None):
        super(Star_Net, self).initialize_from_bounds(
            lower_bound, upper_bound, U_rot, U_rot_inv)

        if constraints is not None:
            s = self.relaxation_at_layers[0]
            s.add_linear_constraints(constraints[0], constraints[1])

        if not self.use_overapproximation_only:
            self.use_general_zonotope = False
            self.initialize_milp_model(-1)

    def verify_recursively(self, lower_bound, upper_bound, true_label,
                           label_maximization=True, split_max_depth=0,
                           start_layer=0, U_rot_inv=None, constraints=None):
        use_overapproximation_only_prev = self.use_overapproximation_only
        last_iteration = split_max_depth == 0

        if not last_iteration:
            self.use_overapproximation_only = True

        lower_bound.requires_grad_(not last_iteration)
        upper_bound.requires_grad_(not last_iteration)

        self.initialize_from_bounds(
            lower_bound, upper_bound, U_rot_inv=U_rot_inv, constraints=constraints)

        s = self.relaxation_at_layers[0]
        if s.check_constraints(s) == -1:
            return True

        self.forward_pass(start_layer)

        isVerified = self.calculate_worst_case(
            true_label, label_maximization)

        if isVerified:
            return True
        elif not last_iteration:
            loss = self.get_verification_loss(true_label, label_maximization)
            loss.backward()

            split_bounds = self.generate_splits(
                lower_bound, upper_bound, 2)

            isVerified = []
            for lower_bound, upper_bound in split_bounds:

                z_net = self.__class__(self.net)
                z_net.use_overapproximation_only = use_overapproximation_only_prev

                result = z_net.verify_recursively(
                    lower_bound, upper_bound, true_label, label_maximization, split_max_depth - 1,
                    start_layer, U_rot_inv, constraints)

                isVerified.append(result)

            return all(isVerified)

        elif self.use_overapproximation_only:
            return False
        else:
            self.initialize_from_bounds(lower_bound, upper_bound)

            self.forward_pass()
            self.add_input_constraints()

            return self.calculate_worst_case(true_label, label_maximization)

    def calculate_worst_case(self, true_label, label_maximization=True, return_violation=False):
        isVerified = super(Star_Net, self).calculate_worst_case(
            true_label, label_maximization)

        violation = None

        if (not isVerified) and (not self.use_overapproximation_only):
            isVerified, violation = self.run_optimization(
                true_label, label_maximization)

        if return_violation:
            return isVerified, violation
        else:
            return isVerified

    def process_from_layer(self, true_label, start_layer, milp_model=None,
                           label_maximization=True, return_violation=False):

        if self.use_overapproximation_only:
            self.forward_pass(start_layer)
        else:

            if milp_model is None:
                self.initialize_milp_model(start_layer-1)
                self.add_input_constraints()
                self.forward_pass(start_layer)
                self.milp_model.update()

            else:
                self.use_overapproximation_only = True
                self.forward_pass(start_layer)
                self.use_overapproximation_only = False

                self.milp_model = milp_model.copy()
                self.milp_model.update()
                self.add_input_constraints()

        return self.calculate_worst_case(true_label, label_maximization, return_violation)

    def apply_normalization_layer(self, idx_layer):
        super(Star_Net, self).apply_normalization_layer(idx_layer)

        if self.use_overapproximation_only:
            return

        s = self.relaxation_at_layers[-1]

        bounds = s.get_bounds()
        lower_bound_np = bounds[0].flatten().numpy()
        upper_bound_np = bounds[1].flatten().numpy()

        layer = self.net.layers[idx_layer]
        sigma = layer.sigma.data[0].item()
        mean = layer.mean.data[0].item()
        weight = 1.0 / sigma
        bias = - mean / sigma

        milp_variables_prev = self.milp_variables_at_layer[idx_layer-1][0]
        num_variables_new = len(milp_variables_prev)
        milp_variables_new = []

        for j in range(num_variables_new):
            var_name = 'x_{}[0,{}]'.format(idx_layer, j)
            if self.use_redundant_constraints:
                var = self.milp_model.addVar(vtype=GRB.CONTINUOUS, lb=lower_bound_np[j],
                                             ub=upper_bound_np[j], name=var_name)
            else:
                var = self.milp_model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY,
                                             ub=GRB.INFINITY, name=var_name)
            milp_variables_new.append(var)

        for j in range(num_variables_new):
            expr = gp.LinExpr()
            expr += -1*milp_variables_new[j]
            expr.addTerms(weight, milp_variables_prev[j])
            expr.addConstant(bias)

            self.milp_model.addConstr(expr, GRB.EQUAL, 0)

        self.milp_variables_at_layer[idx_layer] = (milp_variables_new, )

    def apply_linear_layer(self, idx_layer):
        super(Star_Net, self).apply_linear_layer(idx_layer)

        if self.use_overapproximation_only:
            return

        layer = self.net.layers[idx_layer]
        weight = layer.weight.data.numpy()
        bias = layer.bias.data.squeeze().numpy()

        milp_variables_prev = self.milp_variables_at_layer[idx_layer-1][0]
        num_variables_new, num_variables_prev = weight.shape

        bounds = self.relaxation_at_layers[-1].get_bounds()
        lower_bound_np = bounds[0].flatten().numpy()
        upper_bound_np = bounds[1].flatten().numpy()

        milp_variables_new = []

        # hasFollowingReLU = any([isinstance(layer, torch.nn.ReLU)
        #                         for layer in self.net.layers[idx_layer+1:]])
        # update_LP_model = self.use_tighter_bounds and (not self.use_lp) \
        #     and hasFollowingReLU and self.intermediate_lp_model is not None
        update_LP_model = self.use_tighter_bounds and (not self.use_lp) \
            and self.intermediate_lp_model is not None

        if update_LP_model:
            lp_variables_new = []
            self.intermediate_lp_model.update()
            lp_variables_prev = [self.intermediate_lp_model.getVarByName(v.VarName)
                                 for v in milp_variables_prev]

        # output of matmult
        for j in range(num_variables_new):
            var_name = 'x_{}[0,{}]'.format(idx_layer, j)
            if self.use_redundant_constraints:
                var = self.milp_model.addVar(vtype=GRB.CONTINUOUS, lb=lower_bound_np[j],
                                             ub=upper_bound_np[j], name=var_name)
            else:
                var = self.milp_model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY,
                                             ub=GRB.INFINITY, name=var_name)
            milp_variables_new.append(var)

            if update_LP_model:
                var = self.intermediate_lp_model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY,
                                                        ub=GRB.INFINITY, name=var_name)
                lp_variables_new.append(var)

        for j in range(num_variables_new):

            expr = gp.LinExpr()
            expr += -1*milp_variables_new[j]
            # matmult constraints
            for k in range(num_variables_prev):
                expr.addTerms(weight[j, k], milp_variables_prev[k])
            expr.addConstant(bias[j])
            self.milp_model.addConstr(expr, GRB.EQUAL, 0)

            if update_LP_model:
                expr = gp.LinExpr()
                expr += -1*lp_variables_new[j]
                # matmult constraints
                for k in range(num_variables_prev):
                    expr.addTerms(weight[j, k], lp_variables_prev[k])
                expr.addConstant(bias[j])
                self.intermediate_lp_model.addConstr(expr, GRB.EQUAL, 0)

        self.milp_variables_at_layer[idx_layer] = (milp_variables_new, )

    def apply_convolutional_layer(self, idx_layer):
        super(Star_Net, self).apply_linear_layer(idx_layer)

        if self.use_overapproximation_only:
            return

        layer = self.net.layers[idx_layer]
        weight = layer.weight.data.numpy()
        bias = layer.bias.data.squeeze().numpy()

        milp_variables_prev = self.milp_variables_at_layer[idx_layer-1][0]
        shape_prev = list(self.relaxation_at_layers[-2].a0.shape[1:])
        shape_new = list(self.relaxation_at_layers[-1].a0.shape[1:])

        weight_shape = list(weight.shape)
        strides = layer.stride
        padding = layer.padding

        bounds = self.relaxation_at_layers[-1].get_bounds()
        lower_bound_np = bounds[0][0, :, :, :].numpy()
        upper_bound_np = bounds[1][0, :, :, :].numpy()

        index_i_factor_new = shape_new[1]*shape_new[2]
        index_j_factor_new = shape_new[2]
        index_i_factor_prev = shape_prev[1]*shape_prev[2]
        index_j_factor_prev = shape_prev[2]

        milp_variables_new = []
        for i in range(shape_new[0]):
            for j in range(shape_new[1]):
                for k in range(shape_new[2]):
                    idx_new = index_i_factor_new * i + index_j_factor_new * j + k
                    var_name = 'x_{}[0,{}]'.format(idx_layer, idx_new)
                    if self.use_redundant_constraints:
                        var = self.milp_model.addVar(vtype=GRB.CONTINUOUS, lb=lower_bound_np[i, j, k],
                                                     ub=upper_bound_np[i, j, k], name=var_name)
                    else:
                        var = self.milp_model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY,
                                                     ub=GRB.INFINITY, name=var_name)
                    milp_variables_new.append(var)

        for c_new in range(shape_new[0]):
            for x_new in range(shape_new[1]):
                for y_new in range(shape_new[2]):
                    idx_new = index_i_factor_new * c_new + index_j_factor_new * x_new + y_new
                    expr = gp.LinExpr()
                    expr += -1*milp_variables_new[idx_new]

                    for c_prev in range(shape_prev[0]):
                        for x_shift in range(weight_shape[2]):
                            for y_shift in range(weight_shape[3]):

                                x_prev = x_new * strides[0] + \
                                    x_shift - padding[0]
                                y_prev = y_new * strides[1] + \
                                    y_shift - padding[1]

                                if(x_prev < 0 or x_prev >= shape_prev[1]):
                                    continue

                                if(y_prev < 0 or y_prev >= shape_prev[2]):
                                    continue

                                idx_prev = index_i_factor_prev * c_prev + \
                                    index_j_factor_prev * x_prev + y_prev

                                expr.addTerms(
                                    weight[c_new, c_prev, x_shift, y_shift], milp_variables_prev[idx_prev])

                    expr.addConstant(bias[c_new])
                    self.milp_model.addConstr(expr, GRB.EQUAL, 0)

        self.milp_variables_at_layer[idx_layer] = (milp_variables_new, )

    def apply_relu_layer(self, idx_layer):
        super(Star_Net, self).apply_relu_layer(idx_layer)

        bounds = self.relaxation_at_layers[-2].get_bounds()
        lower_bound = bounds[0].flatten()
        upper_bound = bounds[1].flatten()
        lower_bound_np = lower_bound.numpy()
        upper_bound_np = upper_bound.numpy()
        self.relu_bounds[idx_layer] = (lower_bound_np, upper_bound_np)

        if self.use_overapproximation_only:
            return

        hasLinearConstraints = self.relaxation_at_layers[0].d is not None
        hasPreviousReLU = len(self.milp_relu_constraints) > 0
        needTightening = hasLinearConstraints or hasPreviousReLU and self.use_tighter_bounds
        hasFollowingReLU = any([isinstance(layer, torch.nn.ReLU)
                                for layer in self.net.layers[idx_layer+1:]])

        use_lp_tightening = self.use_tighter_bounds and needTightening
        use_milp_tightening = self.use_tighter_bounds and hasPreviousReLU \
            and self.use_tighter_bounds_using_milp and (not self.use_lp)

        if self.use_tighter_bounds:
            self.milp_model.update()

            if use_milp_tightening:
                self.intermediate_milp_model = self.milp_model.copy()
                self.intermediate_milp_model.update()
                self.intermediate_milp_model.ModelName = 'MILP_Model_Copy'
                self.intermediate_milp_model.Params.TimeLimit = GRB.INFINITY
                self.intermediate_milp_model.Params.BestObjStop = -GRB.INFINITY
                self.intermediate_milp_model.Params.BestBdStop = -1E-5

            if self.use_lp or not hasPreviousReLU:
                self.intermediate_lp_model = self.milp_model.copy()
                self.intermediate_lp_model.ModelName = 'LP_Model_Copy'
                self.intermediate_lp_model.Params.BestObjStop = -GRB.INFINITY
                self.intermediate_lp_model.Params.BestBdStop = -1E-5

        milp_variables_prev = self.milp_variables_at_layer[idx_layer-1][0]
        num_variables_new = len(milp_variables_prev)

        milp_variables_new = []
        milp_indicator_new = []

        for j in range(num_variables_new):
            var_name = 'x_{}[0,{}]'.format(idx_layer, j)

            if self.use_redundant_constraints:
                upper_bound_j = max(0, upper_bound_np[j])
                var = self.milp_model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0.0, ub=upper_bound_j,  name=var_name)
            else:
                var = self.milp_model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0.0, ub=GRB.INFINITY,  name=var_name)

            milp_variables_new.append(var)

            if self.use_tighter_bounds and not self.use_lp:
                var = self.intermediate_lp_model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0.0, ub=GRB.INFINITY,  name=var_name)

            if not self.use_lp:
                var_name = 'x_ind_{}[0,{}]'.format(idx_layer, j)
                var = self.milp_model.addVar(vtype=GRB.BINARY,  name=var_name)
                milp_indicator_new.append(var)

        self.milp_model.update()
        if self.use_tighter_bounds:
            self.intermediate_lp_model.update()

        splits_total = 0
        splits_removed_lp = 0
        splits_removed_milp = 0
        relu_constraints = []

        for j in range(num_variables_new):
            varname_prev = milp_variables_prev[j].VarName
            varname_new = milp_variables_new[j].VarName

            if needTightening:
                if upper_bound_np[j] > 0 and lower_bound_np[j] < 0:

                    splits_total += 1

                    if use_lp_tightening:
                        num_splits_removed = self.refine_relu_bounds(
                            self.intermediate_lp_model, varname_prev, j, idx_layer)
                        splits_removed_lp += num_splits_removed

                    if use_milp_tightening and \
                            upper_bound_np[j] > 0 and lower_bound_np[j] < 0:
                        num_splits_removed = self.refine_relu_bounds(
                            self.intermediate_milp_model, varname_prev, j, idx_layer)
                        splits_removed_milp += num_splits_removed

        lp_neurons = self.determine_lp_neurons(
            lower_bound_np, upper_bound_np, idx_layer, hasFollowingReLU)

        for j in range(num_variables_new):
            varname_prev = milp_variables_prev[j].VarName
            varname_new = milp_variables_new[j].VarName

            constraints = self.encode_relu_transformer(
                self.milp_model, varname_prev, varname_new, j, idx_layer, lp_neurons[j])
            relu_constraints.append(constraints)

            # if hasFollowingReLU and not self.use_lp and self.use_tighter_bounds:
            self.encode_relu_transformer(
                self.intermediate_lp_model, varname_prev, varname_new, j, idx_layer, True)

        self.milp_relu_constraints[idx_layer] = relu_constraints

        if needTightening:
            logger.info('Layer {}, splits removed LP/MILP/Total: {}/{}/{}'.format(
                idx_layer, splits_removed_lp, splits_removed_milp, splits_total))

        self.milp_variables_at_layer[idx_layer] = (
            milp_variables_new, milp_indicator_new)

    def refine_relu_bounds(self, model, varname, idx_neuron, idx_layer):

        lower_bound, upper_bound = self.relu_bounds[idx_layer]

        def refine_upper_bound(model, v):
            try:
                model.reset(1)
                model.setObjective(v, GRB.MAXIMIZE)
                model.optimize()
                # val = v.x
                val = min(model.ObjBound, upper_bound[idx_neuron])
            except Exception:
                logger.info('Upper Bound of {} of model {} failed: {}'.format(
                    v.VarName, model.ModelName, model.Status))
                if model.Status == GRB.INTERRUPTED:
                    logger.error('MILP Status: Interrupted')
                    raise RuntimeError
            else:
                assert(upper_bound[idx_neuron] - val >= -1E-5)
                assert(lower_bound[idx_neuron] - val <= 1E-5)
                upper_bound[idx_neuron] = val
                if model.Status == GRB.TIME_LIMIT:
                    logger.warn(
                        'Timelimit hit while tightening bounds', varname)
                if model.Runtime > 20.0:
                    logger.warn(
                        'Large runtime while tightening bounds', model.Runtime, varname)

        def refine_lower_bound(model, v):
            try:
                model.reset(1)
                model.setObjective(-v, GRB.MAXIMIZE)
                model.optimize()
                # val = v.x
                val = max(-model.ObjBound, lower_bound[idx_neuron])
            except Exception:
                logger.warn('Upper Bound of {} of model {} failed: {}'.format(
                    v.VarName, model.ModelName, model.Status))

                if model.Status == GRB.INTERRUPTED:
                    logger.error('MILP Status: Interrupted')
                    raise RuntimeError
            else:
                assert(upper_bound[idx_neuron] - val >= -1E-5)
                assert(lower_bound[idx_neuron] - val <= 1E-5)
                lower_bound[idx_neuron] = val
                if model.Status == GRB.TIME_LIMIT:
                    logger.warn('Timelimit hit while tightening bounds',
                                varname)
                if model.Runtime > 20.0:
                    logger.warn('Large runtime while tightening bounds',
                                model.Runtime, varname)

        v = model.getVarByName(varname)
        splits_removed = 0

        if upper_bound[idx_neuron] > -lower_bound[idx_neuron]:
            refine_lower_bound(model, v)

            if lower_bound[idx_neuron] < 0:
                refine_upper_bound(model, v)
            else:
                splits_removed += 1
        else:
            refine_upper_bound(model, v)

            if upper_bound[idx_neuron] > 0:
                refine_lower_bound(model, v)
            else:
                splits_removed += 1

        return splits_removed

    def encode_relu_transformer(self, model, varname_prev, varname_new, idx_neuron,
                                idx_layer, use_lp):

        lower_bound, upper_bound = self.relu_bounds[idx_layer]
        model.update()
        v_prev = model.getVarByName(varname_prev)
        v_new = model.getVarByName(varname_new)

        constraints = []
        constr_idx = 0
        constraint_name = 'c_relu_{}[0,{}]'.format(idx_layer, idx_neuron)

        if upper_bound[idx_neuron] <= 0:
            expr = v_new
            c = model.addConstr(expr, GRB.EQUAL, 0,
                                name='{}_{}'.format(constraint_name, constr_idx))
            constraints.append(c)

        elif lower_bound[idx_neuron] >= 0:
            expr = v_new - v_prev
            c = model.addConstr(expr, GRB.EQUAL, 0,
                                name='{}_{}'.format(constraint_name, constr_idx))
            constraints.append(c)

        else:

            if use_lp:

                # y >= x
                expr = v_new - v_prev
                c = model.addConstr(expr, GRB.GREATER_EQUAL, 0,
                                    name='{}_{}'.format(constraint_name, constr_idx))
                constraints.append(c)
                constr_idx += 1

                # # y >= 0
                # expr = v_new
                # self.milp_model.addConstr(expr, GRB.GREATER_EQUAL, 0)

                # y <= u/(u-l)*x - u*l/(u-l)
                a = upper_bound[idx_neuron] / \
                    (upper_bound[idx_neuron] - lower_bound[idx_neuron])
                b = - lower_bound[idx_neuron] * a
                expr = v_new - v_prev * a
                c = model.addConstr(expr, GRB.LESS_EQUAL, b,
                                    name='{}_{}'.format(constraint_name, constr_idx))
                constraints.append(c)

            else:
                indicator_name = varname_new.replace('x_', 'x_ind_')
                indicator_new = model.getVarByName(indicator_name)

                if self.use_relu_multiplication:

                    # y <= x - l(1-a)
                    expr = v_new - v_prev - \
                        lower_bound[idx_neuron]*indicator_new
                    c = model.addConstr(
                        expr, GRB.LESS_EQUAL, -lower_bound[idx_neuron],
                        name='{}_{}'.format(constraint_name, constr_idx))
                    constraints.append(c)
                    constr_idx += 1

                    # y >= x
                    expr = v_new - v_prev
                    c = model.addConstr(expr, GRB.GREATER_EQUAL, 0,
                                        name='{}_{}'.format(constraint_name, constr_idx))
                    constraints.append(c)
                    constr_idx += 1

                    # y <= u.a
                    expr = v_new - \
                        upper_bound[idx_neuron]*indicator_new
                    c = model.addConstr(expr, GRB.LESS_EQUAL, 0,
                                        name='{}_{}'.format(constraint_name, constr_idx))
                    constraints.append(c)
                    constr_idx += 1

                    # # y >= 0
                    # expr = v_new
                    # self.milp_model.addConstr(expr, GRB.GREATER_EQUAL, 0)

                    # indicator constraint
                    c = model.addGenConstrIndicator(
                        indicator_new, True, v_prev, GRB.GREATER_EQUAL, 0.0,
                        name='{}_{}'.format(constraint_name, constr_idx))
                    constraints.append(c)
                else:
                    # x>=0 => y=x
                    c = model.addGenConstrIndicator(
                        indicator_new, True, v_prev, GRB.GREATER_EQUAL, 0.0,
                        name='{}_{}'.format(constraint_name, constr_idx))
                    constraints.append(c)
                    constr_idx += 1

                    c = model.addGenConstrIndicator(
                        indicator_new, True, v_prev - v_new,
                        GRB.EQUAL, 0.0, name='{}_{}'.format(constraint_name, constr_idx))
                    constraints.append(c)
                    constr_idx += 1

                    # x<0 => y=0
                    c = model.addGenConstrIndicator(
                        indicator_new, False, v_prev, GRB.LESS_EQUAL, 0.0,
                        name='{}_{}'.format(constraint_name, constr_idx))
                    constraints.append(c)
                    constr_idx += 1

                    c = model.addGenConstrIndicator(
                        indicator_new, False, v_new, GRB.EQUAL, 0.0,
                        name='{}_{}'.format(constraint_name, constr_idx))
                    constraints.append(c)

        return constraints

    def determine_lp_neurons(self, lower_bound_np, upper_bound_np, idx_layer, hasFollowingReLU):
        num_neurons = lower_bound_np.size
        num_milp_neurons = int(round(num_neurons * self.milp_neuron_ratio))

        if self.use_lp or self.milp_neuron_ratio == 0.0:
            return [True] * num_neurons
        # elif self.milp_neuron_ratio == 1.0:
        elif self.milp_neuron_ratio == 1.0 or hasFollowingReLU:
            return [False] * num_neurons

        lower_bound = torch.Tensor(lower_bound_np)
        upper_bound = torch.Tensor(upper_bound_np)
        mixed_neurons = (lower_bound < 0) * (upper_bound > 0)
        num_mixed_neurons = mixed_neurons.int().sum()

        if num_mixed_neurons <= num_milp_neurons:
            return torch.bitwise_not(mixed_neurons).tolist()

        # Gap
        gap = (upper_bound - lower_bound) * mixed_neurons
        gap_order = gap.argsort(descending=True)
        gap_score = gap_order.argsort()

        # Weight
        weight = self.net.layers[idx_layer+1].weight.data
        weight = weight.abs().sum(0) * mixed_neurons
        weight_order = weight.argsort(descending=True)
        weight_score = weight_order.argsort()

        # Combine
        total_score = gap_score + weight_score
        threshold = total_score.sort(0)[0][num_milp_neurons-1]
        lp_neurons = total_score > threshold

        num_milp_neurons = (torch.bitwise_not(lp_neurons)
                            * mixed_neurons).int().sum().item()
        logger.info('Selected mixed neurons with MILP: {}/{}'.format(
            num_milp_neurons, num_mixed_neurons))

        return lp_neurons.tolist()

    def apply_flatten_layer(self, idx_layer):
        super(Star_Net, self).apply_flatten_layer(idx_layer)

        if self.use_overapproximation_only:
            return
        milp_variables_prev = self.milp_variables_at_layer[idx_layer-1][0]
        self.milp_variables_at_layer[idx_layer] = (milp_variables_prev, )
        # milp_variables_prev = self.milp_variables_at_layer[-1]

        # milp_variables_new = list()

        # for neurons_dim0 in milp_variables_prev:
        #     for neurons_dim1 in neurons_dim0:
        #         for neurons_dim2 in neurons_dim1:
        #             milp_variables_new.append(neurons_dim2)

        # self.milp_variables_at_layer.append(milp_variables_new)
        pass

    def run_optimization(self, true_label, label_maximization=True):
        self.setup_optimization(true_label, label_maximization)
        self.optimize_milp()
        return self.evaluate_optimization()

    def setup_optimization(self, true_label, label_maximization=True):
        self.milp_model.update()

        if isinstance(true_label, torch.Tensor):
            true_label = true_label.item()

        var_name_last_layer = 'x_{}'.format(len(self.net.layers) - 1)

        milp_variables_prev = [
            v for v in self.milp_model.getVars() if var_name_last_layer in v.VarName]

        s = self.relaxation_at_layers[-1]

        a0_diff = s.a0.flatten() - s.a0[0, true_label]
        A_diff = s.A - s.A[:, [true_label]]
        A_diff_abs = torch.sum(A_diff.abs_(), 0)

        y_diff_max = a0_diff + A_diff_abs
        y_diff_min = a0_diff - A_diff_abs
        y_diff_max_np = y_diff_max.numpy()
        y_diff_min_np = y_diff_min.numpy()

        # check which labels violated zonotope overapproximation
        if label_maximization:
            potential_violation_labels = y_diff_max > 0
        else:
            potential_violation_labels = y_diff_min < 0

        potential_violating_labels_zono = [
            x.item() for x in potential_violation_labels.nonzero(as_tuple=False)]

        # Check violating labels using LP
        # if self.use_tighter_bounds and not self.use_lp:
        if False:
            self.intermediate_lp_model.update()
            lp_variables_prev = [
                v for v in self.intermediate_lp_model.getVars() if var_name_last_layer in v.VarName]

            potential_violating_labels_lp = []
            for idx_label in potential_violating_labels_zono:
                expr = lp_variables_prev[idx_label] - \
                    lp_variables_prev[true_label]
                self.intermediate_lp_model.setObjective(expr, GRB.MAXIMIZE)
                self.intermediate_lp_model.optimize()
                score = self.intermediate_lp_model.objVal

                # logger.info(idx_label, score)
                if score > 0:
                    potential_violating_labels_lp.append(idx_label)
            logger.info('Violating labels lp:', potential_violating_labels_lp)
        else:
            potential_violating_labels_lp = potential_violating_labels_zono

        milp_variables_new = []
        # calculate difference between the labels and the true label
        if label_maximization:

            for i, j in enumerate(potential_violating_labels_lp):
                var_name = 'y_diff[0,{}]'.format(j)
                if self.use_redundant_constraints:
                    var = self.milp_model.addVar(vtype=GRB.CONTINUOUS, lb=y_diff_min_np[j],
                                                 ub=y_diff_max_np[j], name=var_name)
                else:
                    var = self.milp_model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY,
                                                 ub=GRB.INFINITY, name=var_name)
                milp_variables_new.append(var)

                expr = milp_variables_new[i] - milp_variables_prev[j] + \
                    milp_variables_prev[true_label]
                self.milp_model.addConstr(expr, GRB.EQUAL, 0)

        else:
            for i, j in enumerate(potential_violating_labels_lp):
                var_name = 'y_diff[0,{}]'.format(j)
                if self.use_redundant_constraints:
                    var = self.milp_model.addVar(vtype=GRB.CONTINUOUS, lb=-y_diff_max_np[j],
                                                 ub=-y_diff_min_np[j], name=var_name)
                else:
                    var = self.milp_model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY,
                                                 ub=GRB.INFINITY, name=var_name)

                milp_variables_new.append(var)

                expr = milp_variables_new[i] + milp_variables_prev[j] - \
                    milp_variables_prev[true_label]
                self.milp_model.addConstr(expr, GRB.EQUAL, 0)

        y_diff_max = self.milp_model.addVar(
            lb=-GRB.INFINITY, ub=GRB.INFINITY, name='y_worst_case')

        self.milp_model.addConstr(y_diff_max == gp.max_(milp_variables_new))

        self.milp_model.setObjective(y_diff_max, GRB.MAXIMIZE)

    def optimize_milp(self):

        self.milp_model.optimize()

    def evaluate_optimization(self):

        status_codes = {GRB.OPTIMAL: 'OPTIMAL', GRB.INFEASIBLE: 'INFEASIBLE',
                        GRB.INF_OR_UNBD: 'INF_OR_UNBD', GRB.UNBOUNDED: 'UNBOUNDED',
                        GRB.TIME_LIMIT: 'TIME_LIMIT', GRB.USER_OBJ_LIMIT: 'USER_OBJ_LIMIT',
                        GRB.SUBOPTIMAL: 'SUBOPTIMAL', GRB.INTERRUPTED: 'INTERRUPTED'}

        status = self.milp_model.Status

        if status in [GRB.OPTIMAL, GRB.USER_OBJ_LIMIT, GRB.SUBOPTIMAL]:

            # isVerified = self.milp_model.ObjBound <= 0
            isVerified = self.milp_model.objVal <= 0

            num_solutions = self.milp_model.SolCount

            worst_cases = []
            for idx_solution in range(num_solutions):
                self.milp_model.Params.SolutionNumber = idx_solution

                if self.milp_model.objVal > 0:
                    worst_case = torch.Tensor([
                        v.x for v in self.milp_model.getVars() if 'x_init' in v.VarName])

                    worst_cases.append(worst_case)

            return isVerified, worst_cases

        elif status == GRB.INFEASIBLE:
            logger.warn('MILP Status: {}'.format(status_codes[status]))
            return True, None
        elif status == GRB.INF_OR_UNBD:
            status_prev = status + 0
            self.milp_model.Params.DualReductions = 0
            self.milp_model.optimize()

            logger.warn('MILP Status: {} --> {}'.format(
                status_codes[status_prev], status_codes[self.milp_model.Status]))
            return False, None

        elif status == GRB.INTERRUPTED:
            logger.warn('MILP Status: {}'.format(status_codes[status]))
            raise RuntimeError

        else:
            logger.warn('MILP Status:', status)
            return False, None

    def rerun_with_additional_constraint(self, C, d):

        num_constraints = d.numel()
        C = C.view([num_constraints, -1])
        d = d.view(-1)

        self.relaxation_at_layers[0].add_linear_constraints(C, d)

        def add_new_constraint_to_model(model):

            milp_variables_init = [
                v for v in model.getVars() if 'x_init' in v.VarName]

            num_dimensions = len(milp_variables_init)

            for i in range(num_constraints):
                expr = gp.LinExpr()
                for j in range(num_dimensions):
                    expr.addTerms(C[i, j], milp_variables_init[j])
                model.addConstr(expr, GRB.LESS_EQUAL, d[i])

        add_new_constraint_to_model(self.milp_model)

        if self.use_warm_start:

            milp_variables_init = [
                v for v in self.milp_model.getVars() if 'x_init' in v.VarName]

            center = self.relaxation_at_layers[0].a0.flatten()
            violation = torch.Tensor([v.x for v in milp_variables_init])
            d_center = (C[0, :] * center).sum()
            d_violation = (C[0, :] * violation).sum()
            k_violation = (d - d_center) / (d_violation - d_center + 1E-3)
            k_center = 1 - k_violation
            new_point = k_center * center + k_violation * violation
            new_point = new_point.view_as(self.relaxation_at_layers[0].a0)

            new_point_rot = new_point.matmul(
                self.relaxation_at_layers[0].U_rot)
            for i, v in enumerate(self.milp_variables_at_layer[0]):
                v.Start = new_point_rot[0, i]

            for i, v in enumerate(self.milp_variables_at_layer[1]):
                v.Start = new_point[0, i]

            x = new_point
            start_layer = self.milp_variables_at_layer[2][0].VarName.split('[')[
                0][-1]
            start_layer = int(start_layer)

            idx_variables = 2
            for idx_layer in range(start_layer, len(self.net.layers)):
                layer = self.net.layers[idx_layer]

                x = layer(x)
                vars = self.milp_variables_at_layer[idx_variables]

                x_flat = x.view(-1)

                if isinstance(layer, torch.nn.ReLU):
                    x_indicator = x_flat > 0.0
                    for i, v in enumerate(vars):
                        v.Start = x_indicator[i]

                    idx_variables += 1
                    vars = self.milp_variables_at_layer[idx_variables]

                for i, v in enumerate(vars):
                    v.Start = x_flat[i]

                idx_variables += 1

            # for v in self.milp_variables_at_layer:
            #     logger.info(len(v), v[0], v[0].x, v[0].Start)

        if self.use_tighter_bounds and self.use_retightening:

            if self.use_tighter_bounds_using_milp:
                intermediate_model = self.intermediate_milp_model
                use_lp = False
            else:
                intermediate_model = self.intermediate_lp_model
                use_lp = True
            intermediate_model.update()
            add_new_constraint_to_model(intermediate_model)

            for idx_layer in self.relu_bounds.keys():
                isLastReLU = idx_layer == list(self.relu_bounds.keys())[-1]

                lower_bound, upper_bound = self.relu_bounds[idx_layer]
                vars, indicators = self.milp_variables_at_layer[idx_layer]
                vars_prev = self.milp_variables_at_layer[idx_layer-1][0]
                layer_constraints = self.milp_relu_constraints[idx_layer]

                splits_removed = 0
                splits_total = 0

                for j, v_new in enumerate(vars):

                    v_prev = vars_prev[j]
                    varname_prev = v_prev.VarName
                    varname_new = v_new.VarName

                    if lower_bound[j] >= 0.0 or upper_bound[j] <= 0.0:
                        continue

                    splits_total += 1

                    num_splits_removed = self.refine_relu_bounds(
                        intermediate_model, varname_prev, j, idx_layer)

                    splits_removed += num_splits_removed

                    # update_model = num_splits_removed == 1 or self.use_lp
                    update_model = True
                    update_intermediate = (not isLastReLU) and \
                        (num_splits_removed == 1 or use_lp)

                    for c in layer_constraints[j]:

                        if isinstance(c, gp.Constr):
                            constraint_name = c.ConstrName
                        elif isinstance(c, gp.GenConstr):
                            constraint_name = c.GenConstrName
                        else:
                            logger.error('Unknown constraint', c)
                            raise RuntimeError

                        if update_model:
                            self.milp_model.remove(c)

                        if update_intermediate:

                            intermediate_model_c = intermediate_model.getConstrByName(
                                constraint_name)

                            if intermediate_model_c is not None:

                                try:
                                    intermediate_model.remove(
                                        intermediate_model_c)
                                except Exception:
                                    logger.info('Removal of constr failed',
                                                c, intermediate_model_c)

                    if update_model:
                        new_constraints_j = self.encode_relu_transformer(
                            self.milp_model, varname_prev, varname_new, j, idx_layer, self.use_lp)
                        layer_constraints[j] = new_constraints_j

                    if update_intermediate:
                        self.encode_relu_transformer(
                            intermediate_model, varname_prev, varname_new, j, idx_layer, use_lp)

                # logger.info('Layer {}, splits Removed/Total: {}/{}'.format(
                #     idx_layer, splits_removed, splits_total))

        self.milp_model.optimize()

        return self.evaluate_optimization()


class Box_Net(Zonotope_Net):

    def __init__(self, net, device='cpu', *_):
        self.device = device
        self.relaxation_type = Box

        self.relaxation_at_layers = []
        self.net = net
        self.remove_gradient_of_net()

        if net is not None:
            if self.net.absolute_layers_without_bias is None:
                self.net.create_absolute_weights()

    def apply_linear_layer(self, idx_layer):
        layer = self.net.layers[idx_layer]
        z = self.relaxation_at_layers[-1]

        a0_new = layer(z.a0)
        A_compact = (z.ub - z.lb)/2
        A_new = self.net.absolute_layers_without_bias[idx_layer](
            A_compact).abs()
        lb = a0_new - A_new
        ub = a0_new + A_new

        z_new = self.relaxation_type(a0_new, lb=lb, ub=ub)

        self.relaxation_at_layers.append(z_new)

    def apply_relu_layer(self, *_):
        z = self.relaxation_at_layers[-1]

        lx, ux = z.get_bounds()
        ux = ux.clamp_min(0)
        lx = lx.clamp_min(0)

        z_new = self.relaxation_type(lb=lx, ub=ux)
        self.relaxation_at_layers.append(z_new)

    def apply_normalization_layer(self, idx_layer):
        layer = self.net.layers[idx_layer]
        z = self.relaxation_at_layers[-1]

        sigma = layer.sigma
        mean = layer.mean

        lb = z.lb.sub(mean).div(sigma)
        ub = z.ub.sub(mean).div(sigma)
        z_new = self.relaxation_type(lb=lb, ub=ub)

        self.relaxation_at_layers.append(z_new)

    def apply_flatten_layer(self, idx_layer):
        layer = self.net.layers[idx_layer]
        z = self.relaxation_at_layers[-1]

        lb = layer(z.lb)
        ub = layer(z.ub)
        z_new = self.relaxation_type(lb=lb, ub=ub)

        self.relaxation_at_layers.append(z_new)

    def initialize_from_bounds(self, lower_bound, upper_bound, *_):
        z_new = self.relaxation_type(lb=lower_bound, ub=upper_bound)

        self.relaxation_at_layers = [z_new]

    def calculate_worst_case(self, true_label, label_maximization=True):

        a0 = self.relaxation_at_layers[-1].a0
        A = (self.relaxation_at_layers[-2].ub -
             self.relaxation_at_layers[-2].lb) / 2
        W_last = self.net.layers[-1].weight.data

        num_labels = a0.numel()

        W_select = torch.eye(num_labels)
        W_select[:, true_label] -= 1

        a0_new = W_select.matmul(a0.transpose(0, 1))
        A_new = W_select.matmul(W_last).abs_().matmul(A.transpose(0, 1))

        if label_maximization:
            self.y = a0_new + A_new
            most_likely_label = torch.argmax(self.y)
        else:
            self.y = a0_new - A_new
            most_likely_label = torch.argmin(self.y)

        return (most_likely_label == true_label).item()
