from keras.layers import RepeatVector, Dense, Activation,Concatenate,Dot,Bidirectional,LSTM,Input
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np

m= 1000 # training example size
Tx = 30 #max len of input
Ty = 10 #max len of output
n_a = 32 # number of units for the pre-attention, bi-directional LSTM's hidden state 'a'
n_s = 64 # number of units for the post-attention LSTM's hidden state "s"
output_vocab_size = 26
input_vocab_size = 26


repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor1 = Dense(units = 10, activation = 'tanh')
densor2 = Dense(units = 1, activation = 'relu')
activator = Activation('softmax',name = 'attention_weights')
dot_layer = Dot(axes=1)

post_activation_LSTM_cell = LSTM(n_s, return_state = True)
output_layer = Dense(output_vocab_size, activation='softmax')


def one_attention(a,s_prev):
    
    s_prev = repeator(s_prev)
    concat = concatenator([a,s_prev])
    e = densor1(concat)
    energies = densor2(e)
    alphas = activator(energies)
    context = dot_layer([alphas,a])
    
    return context

def model(Tx,Ty,n_a,n_s,input_vocab_size,output_vocab_size):
    
    X = Input(shape = (Tx,input_vocab_size))
    s0 = Input(shape = (n_s,),name='s0')
    c0 = Input(shape = (n_s,),name='c0')
    s = s0
    c = c0
    
    outputs = []
    
    a = Bidirectional(LSTM(n_a,return_sequences =True))(X)
    
    for t in range(Ty):
        
        context = one_attention(a,s)
        s,_,c = post_activation_LSTM_cell(inputs = context, initial_state = [s,c])
        output = output_layer(inputs = s)
        outputs.append(output)
    
    model = Model(inputs = [X,s0,c0],outputs = outputs)
    
    return model

model = model(Tx, Ty, n_a, n_s, input_vocab_size, output_vocab_size)
model.summary()
opt = Adam(lr=0.005,beta_1=0.9,beta_2=0.999,decay=0.01)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
model.fit([Xoh,s0,c0],outputs,epochs=4,batch_size=64)
    
        
        
    
