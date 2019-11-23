import pdb
import matplotlib.pyplot as plt
import os
import matplotlib
import argparse
import numpy as np
from scripts.plot_configs import Configs
matplotlib.use('Agg')  # in order to be able to save figure through ssh
# matplotlib.use('TkAgg')  # Can change to 'Agg' for non-interactive mode


def configure_plot(fontsize, usetex):
    fontsize = fontsize
    matplotlib.rc("text", usetex=usetex)
    matplotlib.rcParams['axes.linewidth'] = 0.1
    matplotlib.rc("font", family="Times New Roman")
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = "Times"
    matplotlib.rcParams["figure.figsize"] = 10, 8
    matplotlib.rc("xtick", labelsize=fontsize)
    matplotlib.rc("ytick", labelsize=fontsize)
    params = {'text.latex.preamble': [r'\usepackage{amsfonts}']}
    plt.rcParams.update(params)


def truncate_to_same_len(arrs):
    min_len = np.min([x.size for x in arrs])
    arrs_truncated = [x[:min_len] for x in arrs]
    return arrs_truncated


def read_attr(csv_path, attr):
    def get_items(line):
        return line.rstrip('\n').split('\t')
    iters = []
    is_continuous = True
    has_started = False
    with open(csv_path) as f:
        lines = f.readlines()
        if len(lines) == 0:
            return None, None, None
        attrs = get_items(lines[0])
        if attr not in attrs:
            return None, None, None
        idx = attrs.index(attr)
        vals = []
        for i in range(1, len(lines)):
            d = get_items(lines[i])
            if d[idx] != '':
                vals.append(d[idx])
                iters.append(i-1)
                has_started = True
            elif has_started:  # has holes in the data
                is_continuous = False
    return np.array(vals, dtype=np.float64), iters, is_continuous


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--output', type=str, default='', help='output file name')
    parser.add_argument('--style', type=str, default='', help='icml_piccolo_final')
    parser.add_argument('--y_higher', nargs='?', type=float)
    parser.add_argument('--y_lower', nargs='?', type=float)
    parser.add_argument('--n_iters', nargs='?', type=int)
    parser.add_argument('--legend_loc', type=int, default=0)

    attrs = ['sigma_s_mc', 'sigma_a_mc', 'sigma_tau_mc']
    args = parser.parse_args()
    args.dir = args.logdir
    conf = Configs(args.style)
    attrs = conf.sort_dirs(attrs)
    fontsize = 32 if args.style else 4  # for non style plots, exp name can be quite long
    usetex = True if args.style else False
    configure_plot(fontsize=fontsize, usetex=usetex)

    linewidth = 8
    markersize = 25
    low_thresh = 50
    high_thresh = 900
    for root, _, files in os.walk(args.logdir):
        if 'log.txt' in files:
            for attr in attrs:
                d, iters, is_continuous = read_attr(os.path.join(root, 'log.txt'), attr)
                color = conf.color(attr)
                mark = '-' if is_continuous else 'o-'
                data = np.array(d)
                plt.plot(iters, data,  mark, markersize=markersize,
                         label=conf.label(attr), color=color, linewidth=linewidth)
                plt.yscale('log')
            # Plot rewards.
            d, iters, is_continuous = read_attr(os.path.join(root, 'log.txt'), 'MeanSumOfRewards')
            lower = np.argmax(np.array(d) > low_thresh)
            higher = np.argmax(np.array(d) > high_thresh) - 1

    plt.axvline(x=lower, color='grey', linestyle='--', linewidth=3.0)
    plt.axvline(x=higher, color='grey', linestyle='--', linewidth=3.0)

    legend = plt.legend(loc=args.legend_loc, fontsize=fontsize, frameon=False)
    plt.autoscale(enable=True, tight=True)
    plt.tight_layout()
    plt.ylim(args.y_lower, args.y_higher)
    for line in legend.get_lines():
        line.set_linewidth(10.0)
    if not args.output:
        args.output = '{}.pdf'.format(args.attr)
    output_path = os.path.join(args.dir, args.output)
    plt.savefig(output_path)
    plt.clf()


if __name__ == "__main__":
    main()
