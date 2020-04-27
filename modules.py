from keras.layers import RepeatVector, Dense, Activation,Concatenate,Dot
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np

Tx = 1  #sample_value
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor1 = Dense(units = 10, activation = 'tanh')
densor2 = Dense(units = 1, activation = 'relu')
activator = Activation('softmax',name = 'attention_weights')
dot_layer = Dot(axes=1)

def one_attention(a,s_prev):
    
    s_prev = repeator(s_prev)
    concat = concatenator([a,s_prev])
    e = densor1(concat)
    energies = densor2(e)
    alphas = activator(energies)
    context = dot_layer([alphas,a])
    
    return context

#def model()    
    
