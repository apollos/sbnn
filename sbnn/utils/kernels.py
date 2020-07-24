from operator import mul, add
from functools import reduce
import gpflowSlim.kernels as gfsk
import gpflowSlim as gfs
import tensorflow as tf


def SpectralMixture(params, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        sm = 0.
        for i in range(len(params)):
            w = gfs.Param(params[i]['w'], transform=gfs.transforms.positive, name='w' + str(i))
            sm = gfsk.RBF(**params[i]['rbf']) * gfsk.Cosine(**params[i]['cos']) * w.value + sm
        return sm


_KERNEL_DICT=dict(
    White=gfsk.White,
    Constant=gfsk.Constant,
    ExpQuad=gfsk.RBF,
    RBF=gfsk.RBF,
    Matern12=gfsk.Matern12,
    Matern32=gfsk.Matern32,
    Matern52=gfsk.Matern52,
    Cosine=gfsk.Cosine,
    ArcCosine=gfsk.ArcCosine,
    Linear=gfsk.Linear,
    Periodic=gfsk.Periodic,
    RatQuad=gfsk.RatQuad,
    
    SM=SpectralMixture,
)

def KernelWrapper(hparams):
    assert len(hparams) > 0, 'At least one kernel should be provided.'
    with tf.variable_scope('KernelWrapper'):
        return [_KERNEL_DICT[k['name']](**k['params']) for k in hparams]

