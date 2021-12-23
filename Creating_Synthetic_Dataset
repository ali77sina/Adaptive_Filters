import numpy as np
import matplotlib.pyplot as plt 

#global variables 
l = 50000
low_lim = 100
high_lim = 150
fs = 512
sep_ind = int(0.8*l)
length_of_input = 60

freqs = (np.random.random(l) - 0.5)*200 + 100
plt.hist(freqs, bins = 70)
plt.xlabel('frequency(Hz)')
plt.title('frequency distribution')
plt.show()

phase = (np.random.random(l) - 0.5)*360
plt.hist(phase, bins = 70)
plt.title('phase distribution')
plt.xlabel('Phase (degree)')
plt.show()

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

x_trian = np.expand_dims(x_train,-1)
x_trian = np.expand_dims(x_train,-1)
print(x_trian.shape)

