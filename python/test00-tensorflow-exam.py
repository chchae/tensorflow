import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers




def make_LR_samples_linear(num_sample) :
    np.random.seed(320)
    x = np.linspace( -1, 1, num_sample )
    f = lambda x: 0.5 * x + 1.0
    y = f(x) + 0.4 * np.random.rand( len(x) )
    return np.array(x).reshape(-1,1), np.array(y).reshape(-1,1)

def make_LR_samples_cubic(num_sample) :
    np.random.seed(320)
    x = np.linspace( -1, 1, num_sample )
    y = x**3 + 0.1 * x**2 -0.15 * x + 1.0 + 0.5 * np.random.rand(len(x))
    return np.array(x).reshape(-1,1), np.array(y).reshape(-1,1)


def func_LR_Keras( x_data, y_data ) :
    model = Sequential();
    model.add( Dense( 1, input_dim=x_data.shape[1], activation='linear' ) )
    #model.add( Dense( 64, activation='relu', input_dim=1 ) )
    #model.add( Dense( 64, activation='relu' ) )
    #model.add( Dense(1) )
    #opt = optimizers.SGD( learning_rate=0.01 )
    opt = optimizers.RMSprop(0.001)
    opt = optimizers.Adam(0.001)
    model.compile( loss='mse', optimizer=opt , metrics=['accuracy'] )
    model.fit( x_data, y_data, epochs=20, batch_size=1, shuffle=False )
    return model



def scatterplot( x_data, y_data ) :
    fmin = np.minimum( np.min( x_data ), np.min(y_data) )
    fmax = np.maximum( np.max( x_data ), np.max(y_data) )
    plt.figure( figsize=(8, 8) )
    plt.xlim( fmin, fmax )
    plt.ylim( fmin, fmax )
    plt.scatter( x_data, y_data )
    plt.savefig( 'test00.png' )




def linear_regression_keras() :
    data = pd.read_csv('~/work/delaney-processed.csv')
    x_data = ( data.iloc[:, 2:8] )
    y_data = ( data['measured log solubility in mols per litre'] )
    print ( np.c_[ x_data, y_data ] )

    model = func_LR_Keras( x_data, y_data )
    # print( model.score( x_data,y_data ) )
    print( model.summary() )

    print( model.evaluate( x_data, y_data ) )
    y_pred = model.predict( x_data )
    scatterplot( y_data, y_pred )






def func_loss( w, x, y ) :
    val = 0.0
    for i in range( len(x) ) :
        val += 0.5 * ( w[0] * x[i] + w[1] - y[i] ) ** 2
    return val / len(x)

def func_loss_gradient( w, x, y ) :
    val = np.zeros( len(w) )
    for i in range( len(x) ) :
        er = w[0] * x[i] + w[1] - y[i]
        val += er * np.array( [x[i], 1.0 ] )
    return val / len(w)

def linear_regression_gradient() :
    x, y = make_LR_samples_linear( 100 )
    print( (x,y) )
    return





def exam_tensorflow() :
    x = tf.constant([10, 20])
    y = tf.constant([20, 40])
    z = x * y
    w = z - 1
    print( x.numpy(), y.numpy(), z.numpy(), w.numpy() )








def linear_regression() :
    func_loss = lambda y, yhat : tf.reduce_mean( tf.square( y - yhat ) )
    data_x, data_y = make_LR_samples_linear( 50 )


    class MyModel( tf.keras.Model ) :
        def __init__( self, dim=1, **kwargs ) :
            super().__init__( **kwargs )
            self.w = tf.Variable( tf.ones([dim,1]) )
            self.b = tf.Variable( tf.ones([1]) )

        def call( self, x ) :
            return tf.matmul( x, self.w ) + self.b


    model = MyModel()
    maxepoch = 250
    lr = 0.25
    optimizer = optimizers.Adam(lr)

    for epoch in range( maxepoch ) :
        with tf.GradientTape() as tape :
            curr_loss = func_loss( data_y, model(data_x) )
            gradients = tape.gradient( curr_loss, model.trainable_variables )
            if epoch % 5 == 0:
                print(model.w.numpy(), model.b.numpy(), curr_loss.numpy())
            optimizer.apply_gradients( zip(gradients, model.trainable_variables) )







if __name__ == "__main__" :
    # linear_regression_keras()    
    # exam_tensorflow()

    # linear_regression_gradient()    

    linear_regression()
