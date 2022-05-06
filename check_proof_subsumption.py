import config
import utils
import templates
from relaxations import Box_Net
import argparse
from tqdm import tqdm, trange
import random

def load_net(netname, dataset):
    path = 'examples/{}_nets/{}'.format(dataset, netname)
    file_format = netname.rsplit('.')[1]
    if file_format in ['pth']:
        return utils.load_net_from_patch_attacks(path)
    elif file_format in ['tf', 'pyt']:
        return utils.load_net_from_eran_examples(path)
    else:
        raise RuntimeError

def get_verified_samples(args, N, model, return_activations=False):
    model.to('cpu')
    dataset  = utils.load_dataset_selected_labels_only('mnist', labels=None)
    idx = list(range(len(dataset)))
    random.shuffle(idx)
    cnt = 0
    for i in idx:
        bn = Box_Net(model)
        inp, label = dataset[i]
        inp = inp.unsqueeze(0)
        verify = bn.process_input_once(inp, args.eps, label)
        if verify:
            if return_activations:
                Z = {}
                if verify:
                    relaxations = bn.relaxation_at_layers[1:]
                    Z = dict(enumerate(relaxations))
                yield (inp, label), Z
            else:
                yield (inp, label)
            cnt += 1
        if cnt == N:
            break

def get_relaxation_patch(model, inp, label, a, b, c, d):
    model.to('cpu')
    bn = Box_Net(model)
    lb = inp.clone()
    ub = inp.clone()
    lb[:, :, a:b, c:d] = 0
    ub[:, :, a:b, c:d] = 1
    bn.initialize_from_bounds(lb, ub)
    bn.forward_pass()
    verify = bn.calculate_worst_case(label, label_maximization=True)
    relaxations = bn.relaxation_at_layers[1:]
    Z = dict(enumerate(relaxations))
    return verify, Z

def verify(args):
    model = load_net(args.model, 'mnist')
    dataset  = utils.load_dataset_selected_labels_only('mnist', labels=None)
    model.to('cpu')
    cnt = len(dataset)
    cnt_verif = 0
    cnt_corr = 0
    for i in trange(cnt):
        bn = Box_Net(model)
        inp, label = dataset[i]
        inp = inp.unsqueeze(0)
        cnt_corr += int(model(inp).argmax().item() == label)
        cnt_verif += int(bn.process_input_once(inp, args.eps, label))
    print(f'Standard Accuracy: {100*cnt_corr/cnt:.2f}')
    print(f'Certified l-infty accuracy at eps={args.eps}: {100*cnt_verif/cnt:.2f}')


def check_subsumpition(args):
    random.seed(0) # ensure determinism
    model = load_net(args.model, 'mnist')
    model.to('cpu')
    zs = get_verified_samples(args, args.N, model,  return_activations=True)
    l_info = {i:str(l).split('(')[0] for i, l in enumerate(model.layers)}
    random.seed(0) # ensure determinism
    out = {}
    for (inp, label), z in tqdm(zs, total=args.N):
        for j in range(2):
            x, y = random.randint(0, 28-2), random.randint(0, 28-2)
            _, zp = get_relaxation_patch(model, inp, label, x, x+2, y,y+2)
            for l, li in l_info.items():
                if li != 'ReLU': continue
                zl = z[l]
                zpl = zp[l]
                zl_lb, zl_ub = zl.get_bounds()
                zpl_lb, zpl_ub = zpl.get_bounds()
                subsumbed = ((zl_lb <= zpl_lb) & (zpl_ub <= zl_ub)).all().item()
                if l not in out:
                    out[l] = 0
                out[l] += int(subsumbed)
    for i, key in enumerate(sorted(out.keys())):
        print(f'Patch proofs subsumed in l-infinity proofs for eps={args.eps} at layer {i+1}:', f'{100*out[key]/(2*args.N):.2f}')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='which model to run on')
    parser.add_argument('-N', type=int, default=1000, help='number of points')    
    parser.add_argument('--eps', type=float, default=0.05, help='layer at which to perform matching and union')
    args = parser.parse_args()

    verify(args)
    check_subsumpition(args)

