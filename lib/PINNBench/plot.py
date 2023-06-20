import pickle
import numpy as np
import matplotlib.pyplot as plt

data = pickle.load(open('./Diffusion-reaction/history.pkl', 'rb'))
train = np.asarray(data.loss_train)
val = np.asarray(data.loss_test)

plt.plot(train.sum(axis=1), label='train')
plt.plot(val.sum(axis=1), label='val')
plt.legend()
plt.show()