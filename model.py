import theano
import theano.tensor as T
import lasagne
import lasagne.layers as  ll
import pdb
import numpy as np
import numpy.random as rng
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from utils import * 


def encoder_DCGAN(z_dim=100, input_var=None, num_units=512, batch_size=64):
    encoder = []
    elu = lasagne.nonlinearities.elu

    encoder.append(ll.InputLayer(shape=(batch_size, 3, 64, 64), input_var=input_var))

    encoder.append(ll.Conv2DLayer(encoder[-1], 
                                                      num_filters=num_units/8,
                                                      filter_size=(5,5),
                                                      stride=2, 
                                                      pad=2, 
                                                      nonlinearity=elu))
    # no bn 
    encoder.append(ll.batch_norm(ll.Conv2DLayer(encoder[-1], 
                                                      num_filters=num_units/4,
                                                      filter_size=(5,5),
                                                      stride=2, 
                                                      pad=2, 
                                                      nonlinearity=elu)))
    # no bn 
    encoder.append(ll.batch_norm(ll.Conv2DLayer(encoder[-1], 
                                                      num_filters=num_units/2,
                                                      filter_size=(5,5),
                                                      stride=2, 
                                                      pad=2, 
                                                      nonlinearity=elu)))
    # no bn 
    encoder.append(ll.batch_norm(ll.Conv2DLayer(encoder[-1], 
                                                      num_filters=num_units,
                                                      filter_size=(5,5),
                                                      stride=2, 
                                                      pad=2, 
                                                      nonlinearity=elu)))

    encoder.append(ll.FlattenLayer(encoder[-1]))

    encoder.append(ll.DenseLayer(encoder[-1], num_units=z_dim, nonlinearity=None))
        

    for layer in encoder : 
        print layer.output_shape
    print ""

    return encoder

def encoder(z_dim=100, input_var=None, num_units=512, batch_size=64):
    encoder = []
    elu = lasagne.nonlinearities.elu

    encoder.append(ll.InputLayer(shape=(batch_size, 3, 64, 64), input_var=input_var))

    encoder.append(ll.Conv2DLayer(encoder[-1], 
                                                      num_filters=num_units/8,
                                                      filter_size=(3,3),
                                                      stride=2, 
                                                      pad=2, 
                                                      nonlinearity=elu))
    # no bn 
    encoder.append(ll.batch_norm(ll.Conv2DLayer(encoder[-1], 
                                                      num_filters=num_units/4,
                                                      filter_size=(5,5),
                                                      stride=2, 
                                                      pad=2, 
                                                      nonlinearity=elu)))
    # no bn 
    encoder.append(ll.batch_norm(ll.Conv2DLayer(encoder[-1], 
                                                      num_filters=num_units/2,
                                                      filter_size=(5,5),
                                                      stride=2, 
                                                      pad=2, 
                                                      nonlinearity=elu)))
    # no bn 
    encoder.append(ll.batch_norm(ll.Conv2DLayer(encoder[-1], 
                                                      num_filters=num_units,
                                                      filter_size=(5,5),
                                                      stride=2, 
                                                      pad=2, 
                                                      nonlinearity=elu)))

    encoder.append(ll.FlattenLayer(encoder[-1]))

    encoder.append(ll.DenseLayer(encoder[-1], num_units=z_dim, nonlinearity=None))
        

    for layer in encoder : 
        print layer.output_shape
    print ""

    return encoder




# generates tensors of shape (None, 3, 80, 160)
def generator_DCGAN(z_dim=100, num_units=128, input_var=None, batch_size=64):
    generator = []

    theano_rng = RandomStreams(rng.randint(2 ** 15))
    noise = theano_rng.uniform(size=(batch_size, z_dim))
    input_var = noise if input_var is None else input_var
    elu = lasagne.nonlinearities.elu

    generator.append(ll.InputLayer(shape=(batch_size, z_dim), input_var=input_var))

    generator.append(ll.DenseLayer(generator[-1], num_units*8*4*4, nonlinearity=None))

    generator.append(ll.ReshapeLayer(generator[-1], shape=(-1, num_units*8, 4, 4)))

    # no bn 
    generator.append(ll.batch_norm(ll.TransposedConv2DLayer(generator[-1],
                                                            num_filters=num_units*4, 
                                                            filter_size=(4,4), 
                                                            stride=(2,2), 
                                                            crop=1,
                                                            nonlinearity=elu)))
    # no bn
    generator.append(ll.batch_norm(ll.TransposedConv2DLayer(generator[-1],
                                                            num_filters=num_units*2, 
                                                            filter_size=(4,4), 
                                                            stride=(2,2), 
                                                            crop=1,
                                                            nonlinearity=elu)))
    # no bn 
    generator.append(ll.batch_norm(ll.TransposedConv2DLayer(generator[-1],
                                                            num_filters=num_units, 
                                                            filter_size=(4,4), 
                                                            stride=(2,2), 
                                                            crop=1,
                                                            nonlinearity=elu)))

    generator.append(ll.TransposedConv2DLayer(generator[-1],
                                                            num_filters=3, 
                                                            filter_size=(4,4), 
                                                            stride=(2,2), 
                                                            crop=1, 
                                                            nonlinearity=T.tanh))

    for layer in generator : 
        print layer.output_shape
    print ""

    return generator


