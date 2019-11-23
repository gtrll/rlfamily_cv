import sys
import time
import collections
import copy
import numpy as np
import tensorflow as tf
from contextlib import contextmanager
from rl.tools.utils import logz

#
def safe_assign(obj, *args):
    assert any([isinstance(obj, cls) for cls in args])
    return obj


# To be compatible with python3.4.
def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z


def deepcopy_from_list(old, new, attrs):
    [setattr(old, attr, copy.deepcopy(getattr(new, attr))) for attr in attrs]


def copy_from_list(old, new, attrs):
    [setattr(old, attr, copy.copy(getattr(new, attr))) for attr in attrs]


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


def cprint(string, color='red', bold=False, highlight=False):
    """Print with color."""
    print(colorize(string, color, bold, highlight))


@contextmanager
def timed(msg):
    print(colorize(msg, color='magenta'), end='', flush=True)
    tstart = time.perf_counter()
    yield
    t = time.perf_counter() - tstart
    print(colorize(" in %.3f seconds" % (t), color='magenta'))
    logz.log_tabular(msg + ' Time', t)


def dict_update(d, u):
    """Update dict d based on u recursively."""
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def check_required_params(params, required_params):
    for param in required_params:
        assert param in params, '{} is not included in params'.format(param)


def flatten(vs):
    return np.concatenate([np.reshape(v, [-1]) for v in vs], axis=0)


def unflatten(v, template=None, shapes=None):
    """Shape a flat v in to a list of array with shapes as in template, or with shapes specified by shapes.
    Args:
        v: a np array.
        template: a list of arrays.
        shapes: a list of tuples.
    """
    assert template is not None or shapes is not None
    start = 0
    vs = []
    if template:
        for w in template:
            vs.append(np.reshape(v[start:start + w.size], w.shape))
            start += w.size
    else:
        for shape in shapes:
            size = np.prod(shape)
            vs.append(np.reshape(v[start:start + size], shape))
            start += size
    return vs


def zipsame(*seqs):
    length = len(seqs[0])
    assert all(len(seq) == length for seq in seqs[1:])
    return zip(*seqs)


# TODO do we still need these???
def build_cartpole_feature_map(sy_ob_no):
    """Compute the features based on Ching-An's RKHS IK paper."""
    theta, dp, dtheta = sy_ob_no[:, 1], sy_ob_no[:, 2], sy_ob_no[:, 3]
    s, c = tf.sin(theta), tf.cos(theta)
    ones = tf.ones_like(s, dtype=tf.float32)
    f1 = tf.stack([ones, dp, dtheta, dp * dtheta, dp**2, dtheta**2], axis=1)
    f2 = tf.stack([ones, s, c, s * c, s**2, c**2], axis=1)
    f = tf.reshape(tf.matmul(tf.expand_dims(f1, 2), tf.expand_dims(f2, 1)), [-1, f1.shape[1] * f2.shape[1]])
    return f


def build_model_feature_map(sy_inputs, st_dim):
    """Concatenate (the features for obs), and (actions)."""
    sy_ob_inputs = sy_inputs[:, :st_dim]
    sy_ac_inputs = sy_inputs[:, st_dim:]
    sy_ob_features = sy_ob_inputs
    sy_features = tf.concat([sy_ob_features, sy_ac_inputs], axis=1)
    return sy_features
