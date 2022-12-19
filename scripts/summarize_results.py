from datetime import time
import pandas as pd
import numpy as np
import argparse

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def increase_number_fn(c):
    try:
        n = int(c)
        return str(n+1)
    except ValueError:
        return c


def print_explanation_timing():
    print("Expect Result: This table contains timing results that will vary between computers.")
    print("               The important result is the relative improvement of from 'baseline' (sometimes 'BL') to the other results.")
    print("               In the paper we used the same code to produce these results on the Hardware specified in the paper, but without docker.")
    print("               Thus some difference is to be expected.")


def print_explanation_deterministic():
    print("Expect Result: This table contains deterministically reproducible results.")
    print("               The result is expected to be the same as in the paper.")

def print_table(table_nr, timing, df):
    print()
    print(f"Table {table_nr}")
    print()
    if timing:
        print_explanation_timing()
    else:
        print_explanation_deterministic()
    if table_nr == 5: print('               The non-timing result ("patches matched") is deterministic and expected to be the same as in the paper.')
    if table_nr == 6: print('               The non-timing result ("verif. acc.", "patch mat.", "patch verif.") is deterministic and expected to be the same as in the paper.')
    if table_nr  == 7:
        print('               This table shows speed-ups over the baseline. As both the baseline runtime and the proof sharing runtime are timing results,')
        print('               difference is to be expected. The results are expected to show the same trend though.')
    if table_nr == 8: print('               The non-timing result ("verif.", "splits verif.", "splits matched") is deterministic and expected to be the same as in the paper.')
    if table_nr == 9:
        print('               Here the baseline is m=1 and m>1 is our method. Non-timing result ("splits matched") are deterministic and expected to be the same as in the paper.')
        print()
        print("Note: In the paper the column 't' is reported by a factor 10 too high. Here this is fixed. This does not change our claim, as also the baseline is off by the same factor.")
    if table_nr == 10:
        print('               The non-timing results ("shapes matched") are deterministic and expected to be the same as in the paper if the same number samples (2000 per class) are used.')
        print('               If less are used (the default; see README.md) the "shapes matched" will trend towards the results in the paper as the number of samples increases.')
        print('               For the timing results the rough trend between the baseline and the timing values in the tables should hold, although depending on the number of samples the absolute values may be much lower.')
        print('               Reducing the number of samples by a large amount (e.g., to 5) will obscure many subtle tends.')
    print()
    print(df)


def read_file(fn, read_table7=False):
    verified, predicted, total, p_submatched, p_verified, time = None, None, None, None, None, None
    with open(fn, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Images Verified/Predicted/Total:"):
                line = line.replace("Images Verified/Predicted/Total:", "").strip()
                verified, predicted, total = line.split('/')
                verified, predicted, total = float(verified), float(predicted), float(total)
            elif line.startswith("Images Submatched/Verified/Predicted/Total:"):
                line = line.replace("Images Submatched/Verified/Predicted/Total:", "").strip()
                submatched, verified, predicted, total = line.split('/')
                p_submatched, verified, predicted, total = float(submatched), float(verified), float(predicted), float(total)
            elif line.startswith("Images Verified/Total:"):
                line = line.replace("Images Verified/Total:", "").strip()
                verified, total = line.split('/')
                verified, total = float(verified), float(total)
            elif line.startswith("Patches Submatched/Verified:"):
                line = line.replace("Patches Submatched/Verified:", "").strip()
                p_submatched, p_verified = line.split('/')
                p_submatched, p_verified = float(p_submatched), float(p_verified)
            elif line.startswith("Time spent:"):
                line = line.replace("Time spent:", "").strip()
                time = float(line)
    if read_table7:
        tS, tgen, tmatch = None, None, None
        for line in lines:
            if line.startswith("tS "):
                line = line.replace("tS ", "").strip()
                tS = float(line)
            elif line.startswith("tgen "):
                line = line.replace("tgen ", "").strip()
                tgen = float(line)
            elif line.startswith("tmatch "):
                line = line.replace("tmatch ", "").strip()
                tmatch = float(line)
        return verified, predicted, total, p_submatched, p_verified, time, tS, tgen, tmatch
    else:
        return verified, predicted, total, p_submatched, p_verified, time 


parser = argparse.ArgumentParser()
parser.add_argument('--table', type=int, required=True, help='which table to create')
args = parser.parse_args()

assert args.table in [3, 4, 5, 6, 7, 8, 9, 10]

if args.table == 3:
    n_layers = 7
    net = '7x200'
    method = 'linfinity'
    datasets = ['mnist', 'cifar']
    layers = list(range(n_layers))
    r = 1
    table = np.zeros((2, n_layers+1))
    for ds_index, dataset in enumerate(datasets):
        for layer in range(n_layers):
            fn=f"results/patches_{dataset}_{net}_{layer}_{method}_{r}.txt"
            verified, predicted, total, p_submatched, p_verified, time = read_file(fn)
            table[ds_index][layer] = p_submatched
            table[ds_index][n_layers] = p_verified
    table = pd.DataFrame(table, columns=list(range(1, n_layers+1))+['patches verified'])
    table.index = datasets
    table = 100*table
    print_table(args.table, False, table.round(1))

elif args.table == 4:
    datasets = ['mnist', 'cifar']
    net = '7x200'
    method = 'linfinity'
    layers = ['base', 0, 1, 2, 3, '0 2', '1 2', '1 3', '1 2 3']
    layers = list(map(str, layers))
    rep = list(range(1, 3+1))
    table = np.zeros((len(datasets), len(layers), len(rep)))
    for r in rep:
        for d, dataset in enumerate(datasets):
            for l, layer  in enumerate(layers):
                layer = layer.replace(' ', '+')
                method = 'base' if layer == 'base' else 'linfinity'
                fn=f"results/patches_{dataset}_{net}_{layer}_{method}_{r}.txt"
                verified, predicted, total, p_submatched, p_verified, time = read_file(fn)
                table[d, l, r-1] = time
    reps = np.repeat(np.array(rep), len(datasets)).reshape((-1, 1))
    table = np.concatenate([table[:, :, i-1] for i in rep], axis=0)
    table = np.concatenate([table, reps], axis=1)

    increase_layer_nr_fn = lambda x: "".join(list(map(increase_number_fn, x)))
    layers = map(increase_layer_nr_fn, layers)
    layers = list(map(lambda x: x.replace(' ', '+'), layers))
    if layers[0] == 'base': layers[0] = 'baseline'

    df = pd.DataFrame(table, columns=layers+['rep'])/100
    df['dataset'] = datasets*len(rep)
    df_mean = df.groupby('dataset').mean().sort_values('dataset', ascending=False)
    df_mean.drop(labels=['rep'], axis=1, inplace=True)
    df_mean = df_mean.round(2)
    df_std = df.groupby('dataset').std().sort_values('dataset', ascending=False)
    df_std.drop(labels=['rep'], axis=1, inplace=True)
    df_std = df_std.round(2)
    df_mean = df_mean.astype(str) + ' +/- '
    df_std = df_std.astype(str)
    print_table(args.table, True, df_mean + df_std)

elif args.table == 5:
    dataset = 'mnist'
    net = '7x200'
    method = 'linfinity'
    layer = '1 2'
    methods = ['base', 'linfinity', 'centerandborder', 'grid2']
    m = ['-', '1', '2', '4']
    rep = list(range(1, 3+1))
    table = np.zeros((len(methods), 2, len(rep)))
    for r in rep:
        for i, method in enumerate(methods):
            l = layer.replace(' ', '+')
            if method == 'base': l = 'base'
            fn=f"results/patches_{dataset}_{net}_{l}_{method}_{r}.txt"
            verified, predicted, total, p_submatched, p_verified, time = read_file(fn)
            table[i, 0, r-1] = p_submatched * 100
            table[i, 1, r-1] = time / 100 

    reps = np.repeat(np.array(rep), len(methods)).reshape((-1, 1))
    table = np.concatenate([table[:, :, i-1] for i in rep], axis=0)
    table = np.concatenate([table, reps], axis=1)
    df = pd.DataFrame(table, columns=['patches matched', 'time', 'rep'])
    df['method'] = list(range(len(methods)))*len(rep)
    df_mean = df.groupby('method').mean()
    df_mean['m'] = m

    df_mean.time = df_mean.time.round(2)
    df_mean.index = df_mean.index.map(lambda x: methods[x])
    df_mean = df_mean[['m', 'patches matched', 'time']]
    df_std = df.groupby('method').std()
    df_std.index = df_std.index.map(lambda x: methods[x])
    df = df_mean.copy()
    df_std.time = df_std.time.round(2).astype(str)
    df.time = df_mean.time.astype(str) + ' +/- '
    df.time = df.time + df_std.time
    print_table(args.table, True, df)

elif args.table == 6:
    datasets = ['mnist', 'cifar']
    nets = ['7x200', '9x500']
    method = 'linfinity'
    layer = '1 2'
    rep = list(range(1, 3+1))
    nrows = len(datasets)*len(nets)
    table = np.zeros((nrows, 5, len(rep)))
    for r in rep:
        for d, dataset in enumerate(datasets):
            for n, net in enumerate(nets):
                i = n + d*len(datasets)

                l = layer.replace(' ', '+')
                fn=f"results/patches_{dataset}_{net}_{l}_{method}_{r}.txt"
                verified, predicted, total, p_submatched, p_verified, time = read_file(fn)
                table[i, 0, r-1] = verified
                table[i, 2, r-1] = time / total
                table[i, 3, r-1] = p_submatched 
                table[i, 4, r-1] = p_verified 
                
                fn=f"results/patches_{dataset}_{net}_base_base_{r}.txt"
                verified, predicted, total, p_submatched, p_verified, time = read_file(fn)
                table[i, 1, r-1] = time / total

    reps = np.repeat(np.array(rep), nrows).reshape((-1, 1))
    table = np.concatenate([table[:, :, i-1] for i in rep], axis=0)
    table = np.concatenate([table, reps], axis=1)
    df = pd.DataFrame(table, columns=['verif. acc.', 'time BL', 'time PS', 'patch mat.', 'patch verif.', 'rep'])
    df['dataset'] = np.repeat(np.array(datasets), len(nets)).tolist()*len(rep)
    df['model'] = (nets*len(datasets))*len(rep)
    df_mean = df.groupby(['dataset', 'model']).mean()
    df_mean = df_mean.sort_values(['dataset', 'model'], ascending=[False, True])
    df_mean.drop(labels=['rep'], axis=1, inplace=True)
    df_mean['patch mat.'] *= 100
    df_mean['patch verif.'] *= 100
    df_mean['time BL']  = df_mean['time BL'].round(2)
    df_mean['time PS']  = df_mean['time PS'].round(2)
    df_std= df.groupby(['dataset', 'model']).std()
    df = df_mean.copy()
    df_std['time BL'] = df_std['time BL'].round(2).astype(str)
    df_std['time PS'] = df_std['time PS'].round(2).astype(str)
    df['time BL'] = df_mean['time BL'].astype(str) + ' +/- '
    df['time BL'] = df['time BL'] + df_std['time BL']
    df['time PS'] = df_mean['time PS'].astype(str) + ' +/- '
    df['time PS'] = df['time PS'] + df_std['time PS']
    print_table(args.table, True, df)

elif args.table == 7:
    n_layers = 4 
    net = '7x200'
    method = 'linfinity'
    dataset = 'mnist'
    layers = list(range(n_layers))
    r = 1
    n_spec = 28*28
    table = np.zeros((4, n_layers))

    fn=f"results/patches_{dataset}_{net}_base_base_{r}.txt"
    _, _, _, _, _, tBL = read_file(fn)
    for i, layer in enumerate(layers):
        fn=f"results/patchestiming_{dataset}_{net}_{layer}_{method}_{r}.txt"
        _, _, total, _, _, tPS, tS, tgen, tmatch = read_file(fn, True)
        table[0, i] = tBL/tPS
        table[1, i] = tBL/(total*(tgen + n_spec*tS + n_spec*tmatch))
        table[2, i] = tBL/(total*(tgen + n_spec*tS))
        table[3, i] = tBL/(total*(n_spec*tS))
    table = pd.DataFrame(table, columns=list(range(1, n_layers+1)))
    table.index = ['realized', 'optimal', 'optimal, no match.', 'optimal, no gen., no match.']
    print_table(args.table, True, table.round(2))

elif args.table == 8:
    dataset = 'mnist'
    net = '7x200'
    method = 'linfinity'
    layer = '1 2'
    splits = [4, 6, 8, 10]
    rep = list(range(1, 3+1))
    table = np.zeros((len(splits), 6, len(rep)))

    for r in rep:
        for s, split in enumerate(splits):
            l = 'base'
            fn=f"results/geometrics{split}_{dataset}_{net}_{l}_base_{r}.txt"
            verified, predicted, total, p_submatched, p_verified, time = read_file(fn)
            table[s, 4, r-1] = time / total
            l = layer.replace(' ', '+')
            fn=f"results/geometrics{split}_{dataset}_{net}_{l}_{method}_{r}.txt"
            verified, predicted, total, p_submatched, p_verified, time = read_file(fn)
            table[s, 0, r-1] = split
            table[s, 1, r-1] = verified
            table[s, 2, r-1] = p_verified * 100 
            table[s, 3, r-1] = p_submatched * 100
            table[s, 5, r-1] = time / total 
    reps = np.repeat(np.array(rep), len(splits)).reshape((-1, 1))
    table = np.concatenate([table[:, :, i-1] for i in rep], axis=0)
    table = np.concatenate([table, reps], axis=1)
    df = pd.DataFrame(table, columns=['r', 'verif.', 'splits verif.', 'splits matched', 'time BL', 'time PS', 'rep'])
    df_mean = df.groupby('r').mean()
    df_mean.drop(labels=['rep'], axis=1, inplace=True)
    df_mean['time BL'] = df_mean['time BL'].round(2)
    df_mean['time PS'] = df_mean['time PS'].round(2)
    df_std= df.groupby(['r']).std()
    df = df_mean.copy()
    df_std['time BL'] = df_std['time BL'].round(2).astype(str)
    df_std['time PS'] = df_std['time PS'].round(2).astype(str)
    df['time BL'] = df_mean['time BL'].astype(str) + ' +/- '
    df['time BL'] = df['time BL'] + df_std['time BL']
    df['time PS'] = df_mean['time PS'].astype(str) + ' +/- '
    df['time PS'] = df['time PS'] + df_std['time PS']
    print_table(args.table, True, df)

elif args.table == 9:
    dataset = 'mnist'
    net = '7x200'
    methods = ['base', 'l_infinity', 'rotation2_40', 'rotation3_40']
    m = ['-', '1', '2', '3']
    layer = '1 2'
    rep = list(range(1, 3+1))
    table = np.zeros((len(methods), 2, len(rep)))

    for r in rep:
        for i, method in enumerate(methods):
            l = 'base' if method == 'base' else layer.replace(' ', '+')
            method = method.replace('_', '')
            fn=f"results/geometric40_{dataset}_{net}_{l}_{method}_{r}.txt"
            verified, predicted, total, p_submatched, p_verified, time = read_file(fn)
            table[i, 0, r-1] = p_submatched * 100
            table[i, 1, r-1] = time/total
            verified = verified/total
            p_verified = p_verified
    reps = np.repeat(np.array(rep), len(methods)).reshape((-1, 1))
    table = np.concatenate([table[:, :, i-1] for i in rep], axis=0)
    table = np.concatenate([table, reps], axis=1)
    df = pd.DataFrame(table, columns=['splits matched', 't', 'rep'])
    df['m'] = m*len(rep)
    df_mean = df.groupby('m').mean()
    df_mean.drop(labels=['rep'], axis=1, inplace=True)
    df_mean['t'] = df_mean['t'].round(2)
    df_std= df.groupby(['m']).std()
    df = df_mean.copy()
    df_std['t'] = df_std['t'].round(2).astype(str)
    df['t'] = df_mean['t'].astype(str) + ' +/- '
    df['t'] = df['t'] + df_std['t']
    print_table(args.table, True, df)
    print('Samples verified', verified*100)
    print('Splits verified', p_verified*100)


elif args.table == 10:
    dataset = 'mnist'
    rep = [1, 2, 3]
    def print_linf_table(method, epsilon):
        layers = ['2', '3', '2 3']
        ms = [1, 3, 25] 
        table = np.zeros((3, 2*len(ms), len(rep)))
        for r in rep:
            for i, layer in enumerate(layers):
                for j, m in enumerate(ms):
                    verified, predicted, total, submatched, time = 0, 0, 0, 0, 0
                    for label in range(10):
                        l = layer.replace(' ', '+')
                        fn = f"results/linf{epsilon:.2f}_{dataset}{label}_5x100_{l}_{method}{m}_{r}.txt"
                        _verified, _predicted, _total, _submatched, _, _time = read_file(fn)
                        verified += _verified
                        predicted += _predicted
                        total += _total
                        submatched += _submatched
                        time += _time
                    table[i, j, r-1] = submatched/total*100
                    table[i, len(ms)+j, r-1] = time
        reps = np.repeat(np.array(rep), len(layers)).reshape((-1, 1))
        table = np.concatenate([table[:, :, i-1] for i in rep], axis=0)
        table = np.concatenate([table, reps], axis=1)
        df = pd.DataFrame(table, columns=['shapes matched m='+str(m) for m in ms]+['t m='+str(m) for m in ms]+['rep'])
        increase_layer_nr_fn = lambda x: "".join(list(map(increase_number_fn, x)))
        layers = map(increase_layer_nr_fn, layers)
        layers = list(map(lambda x: x.replace(' ', '+'), layers))
        df['layer'] = layers*(len(rep))
        df.drop(labels=['rep'], axis=1, inplace=True)
        df_mean = df.groupby('layer').mean()
        df_std = df.groupby('layer').std()
        for m in ms:
            df_mean['shapes matched m='+str(m)] = df_mean['shapes matched m='+str(m)].round(2)
            df_mean['t m='+str(m)] = df_mean['t m='+str(m)].round(2).astype(str) + ' +/- ' + df_std['t m='+str(m)].round(2).astype(str)
        print(f"{method}, eps={epsilon}")
        print(df_mean)


    print_table(args.table, True, '')
    for eps in [0.05, 0.1]:
        ts = []
        for r in rep:
            verified, predicted, total, submatched, time = 0, 0, 0, 0, 0
            for label in range(10):
                fn = f"results/linf{eps:.2f}_{dataset}{label}_5x100_base_base_{r}.txt"
                _verified, _predicted, _total, _, _, _time = read_file(fn)
                verified += _verified
                predicted += _predicted
                total += _total
                time += _time
            verified = verified/total
            ts.append(time)
        t_mean = np.mean(ts)
        t_std = np.std(ts)
        print(f'At eps={eps}:')
        print(f'\tVerified: {verified}')
        print(f'\tBaseline: {t_mean} +/- {t_std}')
        print()

    for method in ['box', 'boxTE', 'star', 'starTE']:
        for eps in [0.05, 0.1]:
            print_linf_table(method, eps)
            print()
