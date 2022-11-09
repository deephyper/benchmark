from math import pi as PI
from math import exp as exp
import random
import numpy as np
import tensorflow as tf



def analytical_solution(NT, NX, TMAX, XMAX, NU):
   """
   Returns the velocity field and distance for the analytical solution
   """

   # Increments
   DT = TMAX/(NT-1)
   DX = XMAX/(NX-1)

   # Initialise data structures
   import numpy as np
   u_analytical = np.zeros((NX,NT))
   x = np.zeros(NX)
   t = np.zeros(NT)

   # Distance
   for i in range(0,NX):
       x[i] = i*DX

   # Analytical Solution
   for n in range(0,NT):
       t = n*DT

       for i in range(0,NX):
           phi = exp( -(x[i]-4*t)**2/(4*NU*(t+1)) ) + exp( -(x[i]-4*t-2*PI)**2/(4*NU*(t+1)) )

           dphi = ( -0.5*(x[i]-4*t)/(NU*(t+1))*exp( -(x[i]-4*t)**2/(4*NU*(t+1)) )
               -0.5*(x[i]-4*t-2*PI)/(NU*(t+1))*exp( -(x[i]-4*t-2*PI)**2/(4*NU*(t+1)) ) )

           u_analytical[i,n] = -2*NU*(dphi/phi) + 4

   return u_analytical, x


def convection_diffusion(NT, NX, TMAX, XMAX, NU):
    DT = TMAX/(NT-1)
    DX = XMAX/(NX-1)

    # Initialise data structures
    u = np.zeros((NX,NT))
    u_analytical = np.zeros((NX,NT))
    x = np.zeros(NX)
    t = np.zeros(NT)
    ipos = np.zeros(NX, dtype=int)
    ineg = np.zeros(NX, dtype=int)

   # Periodic boundary conditions
    for i in range(0,NX):
        x[i] = i*DX
        ipos[i] = i+1
        ineg[i] = i-1

    ipos[NX-1] = 0
    ineg[0] = NX-1

   # Initial conditions
    for i in range(0,NX):
        phi = exp( -(x[i]**2)/(4*NU) ) + exp( -(x[i]-2*PI)**2 / (4*NU) )
        dphi = -(0.5*x[i]/NU)*exp( -(x[i]**2) / (4*NU) ) - (0.5*(x[i]-2*PI) / NU )*exp(-(x[i]-2*PI)**2 / (4*NU) )
        u[i,0] = -2*NU*(dphi/phi) + 4
        #u[i, 0] = -np.sin(PI*x[i])
    
   # Numerical solution
    for n in range(0,NT-1):
        for i in range(0,NX):
            u[i,n+1] = (u[i,n]-u[i,n]*(DT/DX)*(u[i,n]-u[ineg[i],n])+ NU*(DT/DX**2)*(u[ipos[i],n]-2*u[i,n]+u[ineg[i],n]))

    return u, x



class PINN(tf.keras.Model):
    def __init__(self, num_layers, hidden_dim, output_dim, act_fn):
        super().__init__()

        if act_fn == "relu":
            self.act_fn = tf.nn.relu
        elif act_fn == "leaky_relu":
            self.act_fn = tf.nn.leaky_relu
        elif act_fn == "elu":
            self.act_fn = tf.nn.elu
        elif act_fn == "gelu":
            self.act_fn = tf.nn.gelu
        elif act_fn == "tanh":
            self.act_fn = tf.nn.tanh
        elif act_fn == "sigmoid":
            self.act_fn = tf.nn.sigmoid
        else:
            raise ValueError("%s is not in the activation function list" % act_fn)

        self.lys = [tf.keras.layers.Dense(units=hidden_dim, activation=self.act_fn, kernel_initializer='glorot_uniform',bias_initializer='zeros')]
        for n in range(num_layers):
            self.lys.append(tf.keras.layers.Dense(units=hidden_dim, activation=self.act_fn, kernel_initializer='glorot_uniform',bias_initializer='zeros'))
        self.lys.append(tf.keras.layers.Dense(units=output_dim, activation=None, kernel_initializer='glorot_uniform',bias_initializer='zeros'))
    
    def call(self, x, t):
        x = tf.cast(x, tf.float32)
        t = tf.cast(t, tf.float32)

        x_concat = tf.stack((x,t), axis=1)
        for layer in self.lys:
            x_concat = layer(x_concat)
        return x_concat

class BurgerSupervisor:
    def __init__(self, nu, net, epochs, batch_size, lr, alpha):
        self.nu = nu
        self.net = net # the net predicting u from x, t pairs.
        self.epochs = epochs
        self.batch_size = batch_size

        self.alpha = alpha
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self.loss = tf.losses.MeanSquaredError()
    

    # def u_net(self, x, t):
    #     u = self.net(x, t)
    #     return u


    def f_net(self, x, t):
        x = tf.cast(x, tf.float32)
        t = tf.cast(t, tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(t)
            u_ = tf.squeeze(self.net(x, t))
            u_x = tape.gradient(u_, x)
            u_t = tape.gradient(u_, t)
            u_xx = tape.gradient(u_x, x)
        del tape
        f = u_t + (u_ * u_x) - (self.nu * u_xx)
        return f
    
    # @tf.function
    # def loss(self, true, pred):
    #     true = tf.cast(true, tf.float32)
    #     ls = tf.reduce_mean((tf.square(true - pred)))
    #     return ls
    
    def train(self, x_train, y_train, x_val):
        x_u_train = x_train[0]
        x_f_train = x_train[1]

        x_u = x_u_train[:, 0]
        t_u = x_u_train[:, 1]

        x_f = x_f_train[:, 0]
        t_f = x_f_train[:, 1]
        for i in range(self.epochs):
            train_ls = []
            # for j in range(len(x_f_train) // self.batch_size):
            #     x_train_batch = x_u_train#[j*self.batch_size : (j+1)*self.batch_size]
            #     y_train_batch = y_train#[j*self.batch_size : (j+1)*self.batch_size]
            #     x_f_train_batch = x_f_train[j*self.batch_size : (j+1)*self.batch_size]
            with tf.GradientTape() as t:
                u_out = self.net(x_u, t_u)
                f_out = self.f_net(x_f, t_f)
                loss_u = self.loss(y_train, u_out)
                loss_f = self.loss(0, f_out) 
                tot_loss = loss_u + self.alpha * loss_f
                train_ls.append(tot_loss)
            grads = t.gradient(tot_loss, self.net.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.net.trainable_weights))
        
            # validation loss
            #val_r2 = self.r2(y_val, self.net(x_val[:, 0], x_val[:, 1]))
            val_f_loss = self.loss(0, self.f_net(x_val[:, 0], x_val[:, 1])).numpy()
            print("Epoch %d, training loss: (%.4f, %.4f), validiation PDE loss %.4f" % (i+1, loss_u.numpy(), loss_f.numpy(), val_f_loss))
        return val_f_loss # return the objective.
                
    def r2(self, true, pred):
        true = tf.cast(true, tf.float32)
        rss = tf.reduce_sum((true - pred)**2)  
        tss = tf.reduce_sum((true - tf.reduce_mean(true))**2)
        return 1 - rss/tss

    def test(self, x_test, y_test):
        out = self.net(x_test)
        r2 = self.r2(y_test, out)
        f_loss = self.f_loss(x_test)
        return r2, f_loss



def get_data(NT, NX, TMAX, XMAX, NU):
    """
    NT: number of points in time.
    NX: number of points in space.
    TMAX: the maximum time span.
    XMAX: the maximum space span.
    NU: viscosity
    """
    u,x = convection_diffusion(NT, NX, TMAX, XMAX, NU) # shape of [x, t]
    x = np.linspace(0, XMAX, NX)
    t = np.linspace(0, TMAX, NT)
    xv, tv = np.meshgrid(x, t)
    x_= np.stack((xv.flatten(), tv.flatten()), axis=1)
    y_= u.T.flatten()

    # Get the indices of points on the boundaries and initial conditions.
    b_up_idx = np.where(x_[:,0]==x[-1])[0]
    b_low_idx = np.where(x_[:,0]==x[0])[0]
    ic_idx = np.where(x_[:,1]==t[0])[0]
  
    u_train_idx = np.concatenate([b_up_idx, b_low_idx, ic_idx]).reshape(-1)
    x_u_train = x_[u_train_idx]
    y_u_train = y_[u_train_idx]

    NUM_BC_PT = 500
    u_idx = np.random.permutation(len(x_u_train))[:NUM_BC_PT]
    NUM_COL_PT = 2000#int(.1 * len(x_)) # 10% of the points in domain.
    print("Generating data... total size %d, number of collocation points for trianing %d" % (len(x_), NUM_COL_PT))
    all_idx = np.arange(len(x_)).tolist()
    col_idx = [i for i in all_idx if (i not in u_train_idx)]
    f_idx = random.sample(col_idx, NUM_COL_PT)

    x_f_train = x_[f_idx]
    # y_f_train = y_[f_idx]

    x_val = x_[col_idx][:2048] # keep a small number for now for testing
    y_val = y_[col_idx][:2048]
    return (x_u_train[u_idx], x_f_train), y_u_train[u_idx], (x_val, y_val), (x_, y_)

def plotter(x_, u_, name, NUMX, NUMT):
    """
    x_: shape=[num_points, 2]. First dim being x and second t.
    u_: shape=[num_points,]
    NUMX: number of points the in x-direction.
    NUMT: number of points the in t-direction.
    """
    import matplotlib.animation as animation

    u_ = u_.reshape(NUMT, NUMX).T
    xv = x_[:,0].reshape(NUMT, NUMX)
    tv = x_[:, 1].reshape(NUMT, NUMX)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1)
    ax[0].imshow(u_, interpolation='nearest',
                      cmap='rainbow', 
                      extent=[tv.min(), tv.max(), xv.min(), xv.max()], 
                      origin='lower', aspect='auto' )
    ax[0].set_xlabel('t')
    ax[0].set_ylabel('x')

    line,= ax[1].plot(xv[0], u_[:, 0])
    text = ax[1].text(0.8, 0.9, ' ', transform=ax[1].transAxes)
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('u')
    def animate(t):
        line.set_ydata(u_[:, t])
        text.set_text('t=%.2f' % x_[:, 1][t])
        return line, text
    
    ani = animation.FuncAnimation(fig, animate, frames=u_.shape[1], interval=50)
    plt.tight_layout()
    ani.save('./Burgers_' + name + '.gif', writer='imagemagick', fps=60)



def sanityCheckData():
    nu = 0.01         # constant in the diff. equation
    N_u = 1000                 # number of data points in the boundaries
    N_f = 10000               # number of collocation points

    SEED = 0
    random_state = np.random.RandomState(SEED)
    x_ = np.linspace(-1, 1, 200)
    t_ = np.linspace(0, 1, 100)
    xx, tt = np.meshgrid(x_, t_)
    x_domain = np.stack((xx.flatten(), tt.flatten()), axis=1) 

    x_upper = np.ones((N_u//4, 1), dtype=float)
    x_lower = np.ones((N_u//4, 1), dtype=float) * (-1)
    t_zero = np.zeros((N_u//2, 1), dtype=float)

    t_upper = np.random.rand(N_u//4, 1)
    t_lower = np.random.rand(N_u//4, 1)
    x_zero = (-1) + np.random.rand(N_u//2, 1) * (1 - (-1))

    # stack uppers, lowers and zeros:
    X_upper = np.hstack( (x_upper, t_upper) )
    X_lower = np.hstack( (x_lower, t_lower) )
    X_zero = np.hstack( (x_zero, t_zero) )

    # each one of these three arrays haS 2 columns, 
    # now we stack them vertically, the resulting array will also have 2 
    # columns and 100 rows:
    X_u_train = np.vstack( (X_upper, X_lower, X_zero) )

    # shuffle X_u_train:
    index = np.arange(0, N_u)
    random_state.shuffle(index)
    X_u_train = X_u_train[index, :]
    
    # make X_f_train:
    X_f_train = np.zeros((N_f, 2), dtype=float)
    for row in range(N_f):
        x = random.uniform(-1, 1)  # x range
        t = random.uniform( 0, 1)  # t range

        X_f_train[row, 0] = x 
        X_f_train[row, 1] = t

    # add the boundary points to the collocation points:
    X_f_train = np.vstack( (X_f_train, X_u_train) )

    # make u_train
    u_upper =  np.zeros((N_u//4, 1), dtype=float)
    u_lower =  np.zeros((N_u//4, 1), dtype=float) 
    u_zero = -np.sin(np.pi * x_zero)  

    # stack them in the same order as X_u_train was stacked:
    u_train = np.vstack( (u_upper, u_lower, u_zero) )

    # match indices with X_u_train
    u_train = u_train[index, :]
    return (X_u_train, X_f_train), u_train, x_domain