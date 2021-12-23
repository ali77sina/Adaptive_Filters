import tensorflow as tf
import matplotlib.pyplot as plt
from training import plot, fir_freqz
from scipy.fft import fft, fftfreq

test_ranges = [
               [10,11],
               [15,20],
               [20,25],
               [20,40],
               [20,50],
               [50,60],
               [80,90],
               [100,150],
               [150,200],
               [0,200]

]
test_ranges = np.array(test_ranges)
for i in test_ranges:
  low_lim = i[0]
  high_lim = i[1]
  freqs = (np.random.random(l) - 0.5)*200 + 100
  phase = (np.random.random(l) - 0.5)*360
  x_train = []
  y_train = []
  times = []
  #next variable for testing only
  times_pos = []
  #next variable for testing only
  times_neg = []
  for i in range(l):
    dum = np.sin(2*np.pi*freqs[i]*np.linspace(0,length_of_input/fs,length_of_input)+phase[i]) + (np.random.random(length_of_input)-0.5)
    x_train.append(dum)
    if freqs[i] < high_lim and freqs[i] > low_lim:
      y_train.append(1)
    else:
      y_train.append(0)
  x_train = np.array(x_train)
  y_train = np.array(y_train)
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Input(shape=(60,1)))
  model.add(tf.keras.layers.Conv1D(filters=1,kernel_size=6))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(16,activation='relu'))
  model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
  model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
  model.fit(x_train, y_train, batch_size = 100, epochs=5)
  coeffs = []
  for j in model.trainable_variables[0]:
    coeffs.append(float(j[0]))
  plot(coeffs,high_lim,low_lim)
