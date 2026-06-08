import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        # He initialization
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2 / in_features)
        self.b = np.zeros((1, out_features))
        
        self.dW = None
        self.db = None
        self._input = None  # cache for backward pass

    def forward(self, x):
        """
        x: (batch_size, in_features)
        returns: (batch_size, out_features)
        """
        self._input = x
        return x @ self.W + self.b    # y = xW + b

    def backward(self, dout):
        """
        dout: (batch_size, out_features)
        returns dx: (batch_size, in_features)
        """
        self.dW = self._input.T @ dout
        self.db = np.sum(dout, axis=0, keepdims=True) 
        return dout @ self.W.T

class Conv2D:
    def __init__(self, in_channels, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.in_channels = in_channels
        
        # He initialization
        self.W = np.random.randn(filter_size, filter_size, in_channels, num_filters) * np.sqrt(2 / (filter_size * filter_size * in_channels))
        self.b = np.zeros((1, 1, 1, num_filters))
        
        self.dW = None
        self.db = None
        self._input = None

    def forward(self, x):
        """
        x: (batch_size, height, width, in_channels)
        returns: (batch_size, out_h, out_w, num_filters)
        """
        self._input = x
        batch_size, h, w, _ = x.shape
        fh = fw = self.filter_size
        
        out_h = h - fh + 1
        out_w = w - fw + 1
        
        out = np.zeros((batch_size, out_h, out_w, self.num_filters))
        
        for i in range(out_h):
            for j in range(out_w):
                patch = x[:, i:i+fh, j:j+fw, :]  # (batch, fh, fw, in_channels)
                out[:, i, j, :] = np.tensordot(patch, self.W, axes=([1,2,3], [0,1,2])) + self.b
        
        return out

    def backward(self, dout):
        """
        dout: (batch_size, out_h, out_w, num_filters)
        returns dx: (batch_size, height, width, in_channels)
        """
        x = self._input
        fh = fw = self.filter_size
        
        batch_size, out_h, out_w, _ = dout.shape

        self.dW = np.zeros_like(self.W)
        self.db = np.sum(dout, axis=(0,1,2), keepdims=True)
        dx = np.zeros_like(x)

        for i in range(out_h):
            for j in range(out_w):
                patch = x[:, i:i+fh, j:j+fw, :]
                self.dW += np.tensordot(patch, dout[:, i, j, :], axes=([0], [0]))
                dx[:, i:i+fh, j:j+fw, :] += np.tensordot(dout[:, i, j, :], self.W, axes=([1], [3]))

        return dx
    
class MaxPool2D:
    def __init__(self, pool_size=2):
        self.pool_size = pool_size
        self._input = None
        self._mask = None  # stores where the max values were

    def forward(self, x):
        """
        x: (batch_size, height, width, channels)
        returns: (batch_size, height//pool_size, width//pool_size, channels)
        """
        self._input = x
        batch_size, h, w, c = x.shape
        ps = self.pool_size
        out_h = h // ps
        out_w = w // ps

        out = np.zeros((batch_size, out_h, out_w, c))
        self._mask = np.zeros_like(x)

        for i in range(out_h):
            for j in range(out_w):
                patch = x[:, i*ps:(i+1)*ps, j*ps:(j+1)*ps, :]
                max_val = np.max(patch, axis=(1,2), keepdims=True)
                out[:, i, j, :] = max_val[:, 0, 0, :]
                mask = (patch == max_val)
                mask = mask / np.sum(mask, axis=(1,2), keepdims=True)
                self._mask[:, i*ps:(i+1)*ps, j*ps:(j+1)*ps, :] = mask
        return out

    def backward(self, dout):
        """
        dout: (batch_size, height//pool_size, width//pool_size, channels)
        returns dx: (batch_size, height, width, channels)
        """
        ps = self.pool_size
        dx = np.zeros_like(self._input)
        _, out_h, out_w, _ = dout.shape

        for i in range(out_h):
            for j in range(out_w):
                d = dout[:, i, j, :][:, np.newaxis, np.newaxis, :]
                dx[:, i*ps:(i+1)*ps, j*ps:(j+1)*ps, :] += d * self._mask[:, i*ps:(i+1)*ps, j*ps:(j+1)*ps, :]

        return dx

class Flatten:
    def __init__(self):
        self._input_shape = None

    def forward(self, x):
        """
        x: (batch_size, height, width, channels)
        returns: (batch_size, height*width*channels)
        """
        self._input_shape = x.shape
        return x.reshape(x.shape[0], -1)


    def backward(self, dout):
        """
        dout: (batch_size, height*width*channels)
        returns: (batch_size, height, width, channels)
        """
        return dout.reshape(self._input_shape)
    
class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.training = True  # toggle this for inference
        self._mask = None

    def forward(self, x):
        """
        x: any shape
        returns: same shape as x
        """
        if self.training:
            self._mask = np.random.rand(*x.shape) > self.p
            x = x * self._mask
            x *= (1 / (1 - self.p))

        return x

    def backward(self, dout):
        """
        dout: same shape as x
        returns: same shape as x
        """
        return dout * self._mask
    
class BatchNorm:
    def __init__(self, num_features, momentum=0.9, epsilon=1e-5):
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        self.momentum = momentum
        self.epsilon = epsilon
        self.training = True

        self.running_mean = np.zeros((1, num_features)) 
        self.running_var = np.ones((1, num_features))

        # cache for backward
        self._x_norm = None
        self._var = None
        self._mean = None

        # gradients
        self.dgamma = None
        self.dbeta = None

        self._input = None

    def forward(self, x):
        """
        x: (batch_size, num_features)
        returns: (batch_size, num_features)
        """
        self._input = x
        if self.training:
            self._mean = np.mean(x, axis=0)
            self._var = np.var(x, axis=0)
            self._x_norm = (x - self._mean) / np.sqrt(self._var + self.epsilon)
            out = self.gamma * self._x_norm + self.beta
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self._mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self._var
        else:
            self._x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            out = self.gamma * self._x_norm + self.beta

        return out

    def backward(self, dout):
        """
        dout: (batch_size, num_features)
        returns dx: (batch_size, num_features)
        """
        self.dgamma = np.sum(dout * self._x_norm, axis=0, keepdims=True)
        self.dbeta = np.sum(dout, axis=0, keepdims=True)
        dx_norm = dout * self.gamma
        dvar = np.sum(dx_norm * (self._input - self._mean) * -0.5 * (self._var + self.epsilon)**(-3/2), axis=0)
        dmean = np.sum(dx_norm * -1/np.sqrt(self._var + self.epsilon), axis=0) + dvar * np.mean(-2*(self._input - self._mean), axis=0)
        dx = dx_norm/np.sqrt(self._var+self.epsilon) + dvar * 2*(self._input-self._mean)/dout.shape[0] + dmean/dout.shape[0]

        return dx