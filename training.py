from Creating_Synthetic_Dataset import x_train, y_train
import tensorflow as tf
from scipy.fft import fft, fftfreq

#global variables 
l = 50000
low_lim = 100
high_lim = 150
fs = 512
sep_ind = int(0.8*l)
length_of_input = 60
# Size of FFT analysis
N = 60

def fir_freqz(b):
    # Get the frequency response
    X = np.fft.fft(b, N)
    # Take the magnitude
    Xm = np.abs(X)
    # Convert the magnitude to decibel scale
    Xdb = 20*np.log10(Xm/Xm.max())
    # Frequency vector
    f = np.arange(N)*fs/N        

    return Xdb, f

def plot(coeffs,high_lim,low_lim):

    # FIR filter coefficients
    #b = np.array(list(reversed(coeffs)))
    b = np.array(coeffs)

    # Window to be used
    win = np.kaiser(len(b), 15)
    # Windowed filter coefficients
    b_win = win*b

    # Get frequency response of filter
    Xdb, f = fir_freqz(b)
    # ... and it mirrored version
    Xdb_win, f = fir_freqz(b_win)


    # Plot the impulse response
    plt.subplot(211)
    plt.stem(b, linefmt='b-', markerfmt='bo', basefmt='k-', label='Orig. coeff.')
    plt.grid(True)



    plt.title('Impulse reponse')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    # Plot the frequency response
    plt.subplot(212)
    plt.plot(f, Xdb, 'b', label='Orig. coeff.')
    plt.grid(True)


    plt.title('Frequency reponse for range {} - {} Hz'.format(low_lim,high_lim))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.xlim((0, fs/2)) # Set the frequency limit - being lazy
    plt.tight_layout()
    plt.show()

#creating and training the CNN
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(60,1)))
model.add(tf.keras.layers.Conv1D(filters=1,kernel_size=6, use_bias=False))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(16,activation='relu'))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = 100, epochs=5)

#getting the convulting filters weights and plotting the frequency and step response
coeffs = []
for j in model.trainable_variables[0]:
  coeffs.append(float(j[0]))
plot(coeffs,high_lim,low_lim)
