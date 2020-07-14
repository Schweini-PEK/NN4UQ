import numpy as np
import models
import time
import copy
import matlab.engine
import warnings
import torch
import utils
import torch.optim as optim
from trainable import train, val
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

warnings.filterwarnings('ignore')

seed = 0
np.random.seed(seed)

eng = matlab.engine.start_matlab()

n_pool = 1000

in_dim = 4
out_dim = 2

x_pool = np.zeros((n_pool, in_dim))
bounds = np.zeros((in_dim, 2))
bounds[0, 0] = -4
bounds[0, 1] = 4  # State bounds for T
bounds[1, 0] = -5
bounds[1, 1] = 8  # State bounds for ?
bounds[2, 0] = 0.5
bounds[2, 1] = 1.5  # Absolute value of lower bound (abs. works due to "-" sign in the code)
bounds[3, 0] = 0.5
bounds[3, 1] = 1.5  # Absolute value of upper bound

for j in range(in_dim):
    x_pool[:, j] = np.random.uniform(bounds[j, 0], bounds[j, 1], n_pool)

y_init = []
x_init = []
n_init = 350
for k in range(n_init):
    x_init.append(x_pool[k, :])
    y_i1 = x_pool[k, 0].item()
    y_i2 = x_pool[k, 1].item()
    w_l1 = x_pool[k, 2].item()
    w_u1 = x_pool[k, 3].item()
    u_opt = eng.ocp_solve(y_i1, y_i2, w_l1, w_u1)
    y_init.append(u_opt)

y_init = np.array(y_init)[:, :, 0]
x_init = np.array(x_init)

dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform',
                         name='learning_rate')
dim_num_dense_layers = Integer(low=2, high=6, name='num_layers')
dim_num_dense_nodes = Integer(low=4, high=15, name='num_nodes')
dim_batch_size = Integer(low=2, high=20, name='batch_size')
dim_epochs = Integer(low=100, high=300, prior='log-uniform', name='epochs')

dimensions = [dim_learning_rate,
              dim_num_dense_layers,
              dim_num_dense_nodes,
              dim_batch_size,
              dim_epochs
              ]
default_parameters = [1e-3, 3, 7, 5, 150]

data = utils.data.loader.LoadDataset(path='dataset/data_72000_x3a5.pkl')


@use_named_args(dimensions=dimensions)
def loss_obj(learning_rate, num_layers, num_nodes, batch_size, epochs, dataset):
    train_loader, test_loader = utils.data.loader.get_data_loaders(dataset,
                                                                   batch_size=batch_size,
                                                                   num_workers=4)

    model = models.RSResNet(h_dim=num_nodes, n_h_layers=num_layers)
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate)
    for i in range(epochs):
        train(model, optimizer, train_loader, torch.device("cpu"))

    score = val(model, test_loader)

    print('New GP iteration with scores : ', score)

    # Print the classification accuracy.
    print()
    print("Accuracy: {0:.2%}".format(score))
    print()

    # the optimizer aims for the lowest scores, so we return our negative accuracy
    return score


gp_result = gp_minimize(func=loss_obj,
                        dimensions=dimensions,
                        n_calls=11,
                        noise=1e-8,
                        n_jobs=-1,
                        acq_func="EI",
                        x0=default_parameters)

opt_h = gp_result.x[0], gp_result.x[1], gp_result.x[2], gp_result.x[3], gp_result.x[4]
print(opt_h)
print(gp_result.fun)

x_pool = x_pool[n_init:]
x_pool_rnd = copy.deepcopy(x_pool)

x = x_init
y = y_init

n_previous = n_init
n_samples = [n_init]
scores = []
al_iter = 0
iter_max = 4

ts = 5
n_add = 100
start_time = time.time()

while len(x_pool) > n_add and al_iter < iter_max:
    n_samples.append(n_previous + n_add)
    al_iter += al_iter
    p_var = []

    for j in range(x_pool.shape[0]):
        xs = x_pool[j, :].reshape((1, 4))
        print('--------------------------')
        q1 = []  # for each sample in the pool, q1 holds the preds in qoi1
        q2 = []  # for each sample in the pool, q1 holds the preds in qoi2
        for k in range(ts):
            print(myDropModel.predict(xs))
            q1.append(myDropModel.predict(xs)[0, 0])
            q2.append(myDropModel.predict(xs)[0, 1])

        # A.append(myDropModel.predict(xs000))
        p_var.append(np.var(q2))
    print("--- %s seconds ---" % (time.time() - start_time))

    varr = np.array(p_var)  # Define array with the variance(s)
    ind = varr.argsort()[-n_add:][::-1]  # get Nadd max variance elements
    Xadd = Xpool[ind, :]  # Create a new array with the most uncertain elements
    Xpool = np.delete(Xpool, ind, 0)  # Redefine the pool
    X = np.vstack((X, Xadd))  # Add new dataset to the previous dataset set

    Yadd = []
    for k in range(len(Xadd)):
        # X.append(x[k,:])
        yi1 = Xadd[k, 0]
        yi2 = Xadd[k, 1]
        wl1 = Xadd[k, 2]
        wu1 = Xadd[k, 3]
        yi1 = yi1.item()
        yi2 = yi2.item()
        wl1 = wl1.item()
        wu1 = wu1.item()  # Convert to native python types
        print(yi1, yi2, wl1, wu1)
        ret = eng.ocp_solve(yi1, yi2, wl1, wu1)
        Yadd.append(ret)

    Yadd = np.array(Yadd)
    Y = np.vstack((Y, Yadd[:, :, 0]))

    # Train a model with the augmented dataset
    data = utils.data.loader.LoadDataset(path='dataset/data_72000_x3a5.pkl')
    train_loader, test_loader = utils.data.loader.get_data_loaders(data,
                                                                   batch_size=bs,
                                                                   num_workers=4)
    myDropModel = get_model(training=True)
    myDropModel.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, Y_test))
    myDetModel = get_model()  # note: the `training` is `None` by default
    # transfer the weights from the trained model to this model
    myDetModel.set_weights(myDropModel.get_weights())
    tloss = myDetModel.evaluate(X_test, Y_test, verbose=0)
    scores.append(tloss)

print(scores)