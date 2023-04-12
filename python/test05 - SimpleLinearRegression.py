import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class SimpleLinearRegression:
    def __init__( self, initializer='random' ) :
        if initializer == 'ones' :
            self.var = 1.
        elif initializer == 'zeros' :
            self.var = 0.
        elif initializer == 'random' :
            self.var = tf.random.uniform( shape=[], minval=0., maxval=1. )
            
        self.m = tf.Variable( 1., shape=tf.TensorShape(None) )
        self.b = tf.Variable( self.var )
        
        
    def mse( self, true, predicted ) :
        return tf.reduce_mean( tf.square( true - predicted ) )
            

    def predict( self, x ) :
        return tf.reduce_sum( self.m * x, 1 ) + self.b
    
    
    def update( self, X, y, learning_rate=0.01 ) :
        with tf.GradientTape( persistent=True ) as g :
            loss = self.mse( y, self.predict(X) )




def do_LRClass( x, y ) :
    slr = SimpleLinearRegression()
    for epoch in range( 100 ) :
        slr.update( x, y )





def plot( X, Y ) :
    plt.plot( X, Y, 'bo' )
    # plt.show()
    plt.savefig( 'test05.png' )



def do_SimpleLinearRegression( x_train, y_train ) :

    W = tf.Variable( np.random.random() )
    b = tf.Variable( np.random.random() )

    def compute_loss():
        Hyp = x_train * W + b
        cost = tf.reduce_sum( [ tf.square( Hyp - y_train ) ] )
        return cost


    learningrate = 0.01
    optimizer = tf.optimizers.SGD( learningrate, name='SGD' )

    for epoch in range( 100 ) :
        optimizer.minimize( compute_loss, var_list=[W,b] )
        if 0 == epoch % 2 :
            print( epoch, 'a:', W.numpy(), 'b:', b.numpy(), 'loss:', compute_loss().numpy() )

    plot( x_train, y_train )




x_train = [  1.3, -0.78,  1.26,  0.03,  2.11,  0.24, -0.24, -0.47, -0.77, -0.37,  0.85, -0.41,  1.27,  1.02, -0.76,  2.66]
y_train = [15.27, 17.44, 14.87, 16.75, 14.52, 16.37, 17.78, 17.51, 17.65, 16.74, 16.72, 17.94, 15.83, 15.51, 17.14, 14.42]

if __name__ == '__main__' :
    do_SimpleLinearRegression( x_train, y_train )
    # do_LRClass( x_train, y_train )



