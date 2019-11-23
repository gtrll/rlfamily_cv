import os
import argparse


def main(exp, style):

    attrs = [
        'MeanSumOfRewards',
        'MinSumOfRewards',
        'MeanRolloutLens',
        'std',
        'ExplainedVarianceBefore\(value_function_approximator\)',
        'ExplainedVarianceAfter\(value_function_approximator\)',
        'ExplainedVarianceBefore\(or_dyn\)',
        'ExplainedVarianceAfter\(or_dyn\)',
        'NumSamplesThisBatch',
        'norm_g',
        'norm_mc_g',
        'norm_ac_os',
        'norm_tau_os',
        'online_learner_stepsize',
        'std_g',
        'std_mc_g',
        'std_ac_os',
        'std_tau_os',
        'norm_avg_g',
        'norm_avg_mc_g',
        'norm_avg_ac_os',
        'norm_avg_au_os',
        'sigma_s_mc',
        'sigma_a_mc',
        'polstd_old',
        'polstd_in_log',
        'sigma_tau_mc',
        'n_sts',
        'n_ros_in_total',
    ]

    logdir = os.path.join('/home/robotsrule/Dropbox/control_variate_results', exp)
    os.makedirs(logdir, exist_ok=True)
    for sublogdir in next(os.walk(logdir))[1]:
        ld = os.path.join(logdir, sublogdir)

        for a in attrs:
            print('Plotting {}'.format(a))
            output = os.path.join(ld, '{}.pdf'.format(a))
            if 'Err' in a:
                flags = '--y_higher {}'.format(3.0)
            else:
                flags = ''
            if not style:
                cmd = 'python3 scripts/plot.py --logdir {} --curve percentile --value {} --output {} {} &'.format(
                    ld, a, output, flags)
            else:
                cmd = 'python3 scripts/plot.py --logdir {} --curve percentile value {} --output {} --style {} {}&'.format(
                    ld, a, output, style, flags)
            os.system(cmd)
        """Example for setting the limits of axis.

        y_limit = ''
        y_higher = y_lower = None

        # Set y_higher and y_lower...
        if y_higher is not None:
            y_limit += '--y_higher {} '.format(y_higher)
        if y_lower is not None:
            y_limit += '--y_lower {} '.format(y_lower)

        # add y_limit to the plot command.
        """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str, default='.')
    parser.add_argument('--style', type=str, default='')
    args = parser.parse_args()
    main(args.exp, args.style)
