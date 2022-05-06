import argparse
import torch
from time import time

import config
import utils
import templates
from relaxations import Zonotope_Net


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_net(netname, dataset):
    path = 'examples/{}_nets/{}'.format(dataset, netname)
    file_format = netname.rsplit('.')[1]
    if file_format in ['pth']:
        return utils.load_net_from_patch_attacks(path)
    elif file_format in ['tf', 'pyt']:
        return utils.load_net_from_eran_examples(path)
    else:
        logger.error('Unknown file ending: .{}'.format(file_format))
        raise RuntimeError


if __name__ == "__main__":

    config.init_logger()
    logger = config.logger

    conf = config.config

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-l', '--l_infinity', action='store_true',
                       help='Verify against l_Infinity perturbations')
    group.add_argument('-p', '--patches', action='store_true',
                       help='Verify against patch perturbations')
    group.add_argument('-pt', '--patches_timing', action='store_true',
                       help='same as -p with additonal timing/loggin')
    group.add_argument('-g', '--geometric', action='store_true',
                       help='Verify against geometric perturbations')
    parser.add_argument('--netname', type=str,
                        default=conf.netname[0], help=conf.netname[1])
    parser.add_argument('--dataset', type=str,
                        default=conf.dataset[0], help=conf.dataset[1])
    parser.add_argument('--num_tests', type=int,
                        default=conf.num_tests[0], help=conf.num_tests[1])
    parser.add_argument('--relu_transformer', type=str,
                        default=conf.relu_transformer[0], help=conf.relu_transformer[1])
    parser.add_argument('--epsilon', type=float,
                        default=conf.epsilon[0], help=conf.epsilon[1])
    parser.add_argument('--label', type=int,
                        default=conf.label[0], help=conf.label[1])
    parser.add_argument('--patch_size', type=int,
                        default=conf.patch_size[0], help=conf.patch_size[1])
    parser.add_argument('--data_dir', type=str,
                        default=conf.data_dir[0], help=conf.data_dir[1])
    parser.add_argument('--template_method', type=str,
                        default=conf.template_method[0], help=conf.template_method[1])
    parser.add_argument('--template_layers', type=int, nargs='+',
                        default=conf.template_layers[0], help=conf.template_layers[1])
    parser.add_argument('--template_domain', type=str,
                        default=conf.template_domain[0], help=conf.template_domain[1])
    parser.add_argument('--template_with_hyperplanes', type=str2bool,
                        default=conf.template_with_hyperplanes[0], help=conf.template_with_hyperplanes[1])
    parser.add_argument('--template_dir', type=str, nargs='+',
                        default=conf.template_dir[0], help=conf.template_dir[1])
    parser.add_argument('--num_templates', type=int,
                        default=conf.num_templates[0], help=conf.num_templates[1])
    parser.add_argument('--template_max_eps', type=str2bool,
                        default=False, help='maximize epsilon when creating l-infinity templates')

    args = parser.parse_args()

    net = load_net(args.netname, args.dataset)
    # utils.print_layers_and_sizes(net)

    template_layers = [i for i, l in enumerate(net.layers)
                       if isinstance(l, torch.nn.ReLU)]
    template_layers = [i for j, i in enumerate(template_layers)
                       if j in args.template_layers]

    if args.l_infinity:

        if args.template_dir is not None:
            t = templates.OfflineTemplates(net, template_layers, args.label,
                                           args.template_domain, args.relu_transformer)
            path_to_net = 'examples/{}_nets/{}'.format(
                args.dataset, args.netname)
            t.load_templates(path_to_net, args.template_dir,
                             args.num_templates)
        else:
            dataset = utils.load_dataset_selected_labels_only(
                args.dataset, [args.label], test_set=False)

            t = templates.OfflineTemplates(net, template_layers, args.label,
                                           args.template_domain, args.relu_transformer)
            path_to_net = 'examples/{}_nets/{}'.format(
                args.dataset, args.netname)
            t.create_templates(dataset, args.epsilon, path_to_net,
                               args.template_with_hyperplanes, args.num_templates, args.template_max_eps)

        dataset = utils.load_dataset_selected_labels_only(args.dataset, [args.label],
                                                          num_elements=args.num_tests, test_set=True)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                  shuffle=False, num_workers=2)

        count_total = dataset.targets.shape[0]
        count_true = 0
        count_verified = 0
        count_submatched = 0

        start_time = time()

        for inputs, labels in data_loader:
            label = labels.item()

            if not torch.argmax(net(inputs), 1) == labels:
                continue
            count_true += 1

            z_net = Zonotope_Net(
                net, relu_transformer=args.relu_transformer)
            z_net.initialize(inputs, args.epsilon)

            isSubmatch = False
            for idx_layer in range(len(net.layers)):
                z_net.apply_layer(idx_layer)

                if idx_layer in template_layers:
                    z = z_net.relaxation_at_layers[-1]
                    isSubmatch = t.submatching(z, idx_layer)

                if isSubmatch:
                    count_verified += 1
                    count_submatched += 1
                    break

            if not isSubmatch:
                isVerified = z_net.calculate_worst_case(label)

                if isVerified:
                    count_verified += 1

        logger.info('Net: {}, Dataset: {}, Label: {}'.format(
            args.netname, args.dataset, args.label))
        logger.info('Images Submatched/Verified/Predicted/Total: {}/{}/{}/{}'.format(
            count_submatched, count_verified, count_true, count_total))
        logger.info('Time spent: {:.3f}'.format(time() - start_time))

    elif args.geometric:

        count_total = args.num_tests
        count_verified = 0
        count_verified_splits = 0
        count_submatched_splits = 0

        start_time = time()

        for idx_test in range(args.num_tests):

            inputs, lower_bounds, upper_bounds, specs = utils.load_deepg_specs(
                idx_test, args.data_dir)

            label = net(inputs)[0].argmax().item()

            t = templates.OnlineTemplates(
                net, template_layers, label, args.template_domain, args.relu_transformer)
            t.create_templates(inputs, args.template_method)

            num_splits = lower_bounds.shape[0]

            isVerified_All = True
            num_submatches = 0

            for idx_split in range(num_splits):
                z_net = Zonotope_Net(
                    net, relu_transformer=args.relu_transformer)
                z_net.initialize_from_bounds(
                    lower_bounds[[idx_split]], upper_bounds[[idx_split]])

                isSubmatch = False
                for idx_layer in range(len(net.layers)):
                    z_net.apply_layer(idx_layer)

                    if idx_layer in template_layers:
                        z = z_net.relaxation_at_layers[-1]
                        isSubmatch = t.submatching(z, idx_layer)

                    if isSubmatch:
                        count_verified_splits += 1
                        count_submatched_splits += 1
                        break

                if not isSubmatch:
                    isVerified = z_net.calculate_worst_case(label)

                    if isVerified:
                        count_verified_splits += 1
                    else:
                        isVerified_All = False

            count_verified += int(isVerified_All)

        count_splits_total = num_splits * count_total

        logger.info('Images Verified/Total: {}/{}'.format(
            count_verified, count_total))
        logger.info('Patches Submatched/Verified: {:.3f}/{:.3f}'.format(
            count_submatched_splits / count_splits_total,
            count_verified_splits / count_splits_total))
        logger.info('Time spent: {:.2f}'.format(time() - start_time))
    elif args.patches:
        dataset = utils.load_dataset_selected_labels_only(args.dataset, num_elements=args.num_tests,
                                                          test_set=True)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                  shuffle=False, num_workers=2)

        count_total = dataset.targets.shape[0]
        count_true = 0
        count_verified = 0
        count_verified_patches = 0
        count_submatched_patches = 0

        start_time = time()

        for inputs, labels in data_loader:
            label = labels.item()

            if not torch.argmax(net(inputs), 1) == labels:
                continue
            count_true += 1

            t = templates.OnlineTemplates(
                net, template_layers, label, args.template_domain, args.relu_transformer)
            t.create_templates(inputs, args.template_method)

            num_image_dim = inputs.shape[-1]
            num_patches_per_dimension = num_image_dim - args.patch_size + 1

            isVerified_All = True
            num_submatches = 0

            for idx in range(num_patches_per_dimension):
                for idy in range(num_patches_per_dimension):
                    z_net = Zonotope_Net(
                        net, relu_transformer=args.relu_transformer)

                    x_lb = inputs.clone()
                    x_ub = inputs.clone()

                    x_lb[:, :, idx:(idx+args.patch_size),
                         idy:(idy+args.patch_size)] = 0
                    x_ub[:, :, idx:(idx+args.patch_size),
                         idy:(idy+args.patch_size)] = 1

                    z_net.initialize_from_bounds(x_lb, x_ub)

                    isSubmatch = False
                    for idx_layer in range(len(net.layers)):
                        z_net.apply_layer(idx_layer)

                        if idx_layer in template_layers:
                            z = z_net.relaxation_at_layers[-1]
                            isSubmatch = t.submatching(z, idx_layer)

                        if isSubmatch:
                            count_verified_patches += 1
                            count_submatched_patches += 1
                            break

                    if not isSubmatch:
                        isVerified = z_net.calculate_worst_case(label)

                        if isVerified:
                            count_verified_patches += 1
                        else:
                            isVerified_All = False

            count_verified += int(isVerified_All)

        count_patches_total = (num_patches_per_dimension ** 2) * count_total

        logger.info('Images Verified/Predicted/Total: {}/{}/{}'.format(
            count_verified, count_true, count_total))
        logger.info('Patches Submatched/Verified: {:.3f}/{:.3f}'.format(
            count_submatched_patches / count_patches_total,
            count_verified_patches / count_patches_total))
        logger.info('Time spent: {:.2f}'.format(time() - start_time))
    elif args.patches_timing:
        import numpy as np

        tgen = 0
        tmatch = 0
        tmatch_cnt = 0
        tS = 0

        dataset = utils.load_dataset_selected_labels_only(args.dataset, num_elements=args.num_tests,
                                                          test_set=True)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                  shuffle=False, num_workers=2)

        count_total = dataset.targets.shape[0]
        count_true = 0
        count_verified = 0
        count_verified_patches = 0
        count_submatched_patches = 0
        assert(len(template_layers) == 1)


        start_time = time()

        for inputs, labels in data_loader:
            label = labels.item()

            if not torch.argmax(net(inputs), 1) == labels:
                continue
            count_true += 1

            tgen0 = time()
            t = templates.OnlineTemplates(
                net, template_layers, label, args.template_domain, args.relu_transformer)
            t.create_templates(inputs, args.template_method)
            tgen += time() - tgen0

            num_image_dim = inputs.shape[-1]
            num_patches_per_dimension = num_image_dim - args.patch_size + 1

            isVerified_All = True
            num_submatches = 0

            for idx in range(num_patches_per_dimension):
                for idy in range(num_patches_per_dimension):
                    t0 = time()


                    z_net = Zonotope_Net(
                        net, relu_transformer=args.relu_transformer)

                    x_lb = inputs.clone()
                    x_ub = inputs.clone()

                    x_lb[:, :, idx:(idx+args.patch_size),
                         idy:(idy+args.patch_size)] = 0
                    x_ub[:, :, idx:(idx+args.patch_size),
                         idy:(idy+args.patch_size)] = 1

                    z_net.initialize_from_bounds(x_lb, x_ub)

                    isSubmatch = False
                    for idx_layer in range(len(net.layers)):
                        z_net.apply_layer(idx_layer)

                        if idx_layer in template_layers:
                            tS += time() - t0
                            ts0 = time()
                            z = z_net.relaxation_at_layers[-1]
                            isSubmatch = t.submatching(z, idx_layer)
                            tmatch += time() - ts0
                            tmatch_cnt += 1

                        if isSubmatch:
                            count_verified_patches += 1
                            count_submatched_patches += 1
                            break

                    if not isSubmatch:
                        isVerified = z_net.calculate_worst_case(label)
                        if isVerified:
                            count_verified_patches += 1
                        else:
                            isVerified_All = False

            count_verified += int(isVerified_All)

        count_patches_total = (num_patches_per_dimension ** 2) * count_total
        print('tS', tS/count_patches_total)
        print('tgen', tgen/count_total)
        print('tmatch', tmatch/(tmatch_cnt + 1e-5))

        logger.info('Images Verified/Predicted/Total: {}/{}/{}'.format(
            count_verified, count_true, count_total))
        logger.info('Patches Submatched/Verified: {:.3f}/{:.3f}'.format(
            count_submatched_patches / count_patches_total,
            count_verified_patches / count_patches_total))
        logger.info('Time spent: {:.2f}'.format(time() - start_time))
