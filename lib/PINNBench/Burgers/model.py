import numpy as np
import tensorflow as tf
import phi.flow as pl


def BurgersSolver(NT: int,
            NX: int,
            X_U: float,
            X_L: float,
            T_max: float,
            Nu: float,
            IN_cond):
    """
    Burgers equation solver with Phiflow,
    partially adopted from
    https://physicsbaseddeeplearning.org/overview-burgers-forw.html

    args:
        NT: number of points in the t-direction.
        NX: number of points in the x-direction.
        X_U: the upper bound of x range.
        X_L: the lower bound of x range.
        T_max: the maximum time lenght.
        Nu: viscosity.
        IN_cond: the initial condition function.
    """
    DX = (X_U - X_L) / NX
    DT = T_max / NT
    x_ = np.linspace(X_L, X_U, NX)
    t_ = np.linspace(0, T_max, NT)
    xx, tt = np.meshgrid(x_, t_)
    input_domain = np.stack((xx.flatten(), tt.flatten()), axis=1)

    x_range = np.linspace(X_L + DX/2., X_U - DX/2., NX)
    INITIAL = IN_cond(x_range)
    INITIAL = pl.math.tensor(INITIAL, pl.spatial('x') ) # convert to phiflow tensor
    velocity = pl.CenteredGrid(INITIAL, pl.extrapolation.PERIODIC,
                                x=NX, bounds=pl.Box(x=(X_L,X_U)))
    vt = pl.advect.semi_lagrangian(velocity, velocity, DT)

    velocities = [velocity]
    age = 0.
    for i in range(NT - 1):
        v1 = pl.diffuse.explicit(velocities[-1], Nu, DT)
        v2 = pl.advect.semi_lagrangian(v1, v1, DT)
        age += DT
        velocities.append(v2)
    vels = np.array([v.values.numpy('x,vector') for v in velocities])
    u = np.squeeze(vels)
    return x_, t_, u

def split_data(x_, t_, u):
    """
    Split data into train, val, and test.
        train: input coordinates on the boundaries and its solution (boundary conditions)
                and the input coordinates of collocation points
        val: the collocation points (in training) with solution
        test: the entire domain and its solution.
    """
    rd = np.random.RandomState(0) # set local random seed

    NUM_IC_PT = 500
    NUM_BC_PT = 500
    NUM_COL_PT = 10000

    x_ic_idx = np.array(rd.uniform(size=NUM_IC_PT) * len(x_), dtype=int)
    x_ic = x_[x_ic_idx]
    input_ic = np.stack((x_ic, np.zeros(len(x_ic))), axis=1)
    u_ic = u[0, x_ic_idx].reshape(-1,1)

    t_bc_idx = np.array(rd.uniform(size=NUM_BC_PT//2) * len(t_), dtype=int)
    t_bc = t_[t_bc_idx]

    input_bc_upper = np.stack([np.repeat(x_[-1], len(t_bc)), t_bc], axis=1)
    input_bc_lower = np.stack([np.repeat(x_[0], len(t_bc)), t_bc], axis=1)
    input_bc = np.vstack([input_bc_lower, input_bc_upper])

    u_bc_upper = u[t_bc_idx, -1]
    u_bc_lower = u[t_bc_idx, 0]
    u_bc = np.hstack([u_bc_lower, u_bc_upper]).reshape(-1, 1)


    t_col_idx = np.array(rd.uniform(size=NUM_COL_PT) * len(t_), dtype=int)
    x_col_idx = np.array(rd.uniform(size=NUM_COL_PT) * len(x_), dtype=int)
    input_col = np.stack((x_[x_col_idx], t_[t_col_idx]), axis=1)
    u_col = u[t_col_idx, x_col_idx]

    x_u_train = np.vstack([input_ic, input_bc])
    y_u_train = np.vstack([u_ic, u_bc])
    idx = np.arange(0, NUM_IC_PT+NUM_BC_PT)
    rd.shuffle(idx)
    x_u_train = x_u_train[idx]
    y_u_train = y_u_train[idx]

    x_f_train = input_col

    # prepare val data
    rd2 = np.random.RandomState(42)
    v_t_col_idx = np.array(rd2.uniform(size=NUM_COL_PT) * len(t_), dtype=int)
    v_x_col_idx = np.array(rd2.uniform(size=NUM_COL_PT) * len(x_), dtype=int)
    x_val = np.stack((x_[v_x_col_idx], t_[v_t_col_idx]), axis=1)
    y_val = u[v_t_col_idx, v_x_col_idx]

    # prepare testing data
    xx, tt = np.meshgrid(x_, t_)
    input_domain = np.stack([xx.flatten(), tt.flatten()], axis=1)
    y_test = u.flatten()

    train_data = {"x_u": x_u_train, "y_u": y_u_train, "x_f": x_f_train}
    val_data = {"x": x_val, "y": y_val}
    test_data = {"x": input_domain, "y": y_test}
    return train_data, val_data, test_data

def get_data(NT, NX, X_U, X_L, T_max, Nu):
    def init_cond(x):
        return -np.sin(np.pi*x)
    x_, t_, u = BurgersSolver(NT=NT, NX=NX, X_U=X_U, X_L=X_L, T_max=T_max,
                            Nu=Nu, IN_cond=init_cond)
    train, val, test = split_data(x_, t_, u)
    return train, val, test


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
    def __init__(self, nu, net, epochs, lr, alpha):
        self.nu = nu
        self.net = net # the net predicting u from x, t pairs.
        self.epochs = epochs

        self.alpha = alpha
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self.loss = tf.losses.MeanSquaredError()

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

    def train(self, train, val):
        x_u_train = train["x_u"]
        x_f_train = train["x_f"]
        y_train = train["y_u"]

        x_val = val["x"]
        x_u = x_u_train[:, 0]
        t_u = x_u_train[:, 1]

        x_f = x_f_train[:, 0]
        t_f = x_f_train[:, 1]
        for i in range(self.epochs):
            train_ls = []
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
