import numpy as np
import matplotlib.pyplot as plt

from scipy.io.wavfile import write

def sigmoid(x):
    return 1 - 1 / (1 + np.exp(x))

def relu(x):
    return np.maximum(0,x)

def arsin(x):
    return np.log(x + np.sqrt(np.power(x,2) + 1))

def regexp(x):
    return np.exp(x) - 1

def identity(x):
    return x

""" CPPN class for transforming arrays."""
class Cppn():

    """ Initialize network.
        - n_in: Int, number input neurons
        - n_out: Int, number input neurons
        - n_in: [Int], size of hidden layers
        - weightscale: Float, scales the random weights
        - fun: [Float] -> [Float], activation function
        - out_fun: [Float] -> [Float], output activation function
        - z_dim: Int, dimension of the latent vector"""
    def __init__(self, n_in, n_out, layers, 
                 weightscale=1.0, fun=np.tanh, out_fun=sigmoid,
                 z_dim=32):

        assert layers != [], "At least one hidden layer."

        self.fun = fun
        self.out_fun = out_fun

        self.z_dim = z_dim
        self.z_weight = weightscale * np.random.randn(layers[0],z_dim)

        self.dims = [n_in] + layers + [n_out]
        self.weights = []
        for dim1, dim2 in zip(self.dims[:-1], self.dims[1:]):
            self.weights.append(weightscale * np.random.randn(dim2,dim1))

    def get_output(self, x):
        # First layer and latent vector
        y = self.fun(
                np.dot(self.weights[0], x) + \
                np.dot(self.z_weight, self.z))

        # Hidden layers
        for weight in self.weights[1:-1]:
            y = self.fun(np.dot(weight, y))

        # Output layer
        y = self.out_fun(np.dot(self.weights[-1], y))
        return y

    """ Uses the network to transform array.
        - xs: [n_in, ...] float array
        - z: [z_dim], latent vector
        - z_scale: float, scaling of latent vector"""
    def transform(self, xs, z=None, z_scale=1.0):
        if z is None:
            z = np.random.uniform(-1,1,self.z_dim)
        self.z = z_scale*z
        return np.apply_along_axis(self.get_output, 0, xs)


def create_image(imsize=(200,200), scale=10.0):
    net = Cppn(3, 3, [32,10,32,3])
    aspect = imsize[0]/imsize[1]
    x,y = np.meshgrid(np.linspace(-scale*aspect,scale*aspect,imsize[0]),
                      np.linspace(-scale,scale,imsize[1]))
    r = np.sqrt(np.power(x,2)+np.power(y,2))
    inp = np.stack((x,y,r))
    out = net.transform(inp, z_scale=0.3*scale)

    plt.imshow(out.transpose(1,2,0))
    plt.show()

def create_sound():
    net = Cppn(2, 1, [200,200,3], weightscale=0.4, 
               fun=np.sin, out_fun=np.sin)
    c = np.linspace(-1,1,44100)
    x = np.sin(10*2*np.pi*c)*0.001
    y = np.sin(40*2*np.pi*c)
    z = np.random.uniform(-1,1,32)
    inp = np.stack((c,y))
    out = net.transform(inp, z=z)

    data = out[0]
    plt.plot(data, label="data")
    plt.plot(inp.T, label="input")
    plt.legend()
    plt.show()
    scaled = np.int16(data/np.max(np.abs(data)) * 32767)
    write('test.wav', 44100, scaled)

def create_sound2():
    net = Cppn(2, 1, [200,200,50,50,20,20,3], weightscale=0.5, 
               fun=np.sin, out_fun=np.sin)
    c = np.linspace(-1,1,44100)
    x = np.sin(10*2*np.pi*c)*0.001
    y = np.sin(2*np.pi*c)
    inp = np.stack((c,y))
    out = net.transform(inp)

    data = out[0]
    data = np.convolve(data, np.ones(20))
    plt.plot(data, label="data")
    plt.plot(inp.T, label="input")
    plt.legend()
    plt.show()
    scaled = np.int16(data/np.max(np.abs(data)) * 32767)
    write('test.wav', 44100, scaled)

  
if __name__== "__main__":
    create_image(imsize=(600,400))
    #create_sound()


