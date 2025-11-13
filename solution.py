import numpy as np

from interface import *


# ================================= 1.4.1 SGD ================================
class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter

        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam

            :return: np.array, new parameter values
            """
            # your code here \/
            return parameter - self.lr * parameter_grad
            # your code here /\

        return updater


# ============================= 1.4.2 SGDMomentum ============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter

        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam

            :return: np.array, new parameter values
            """
            # your code here \/

            updater.inertia = self.momentum * updater.inertia + self.lr * parameter_grad

            return parameter - updater.inertia
            # your code here /\

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values

        :return: np.array((n, ...)), output values

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/

        zeros = np.zeros_like(inputs)

        return np.where(inputs >= zeros, inputs, zeros)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

        :return: np.array((n, ...)), dLoss/dInputs

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/

        ones = np.ones_like(self.forward_inputs)
        zeros = np.zeros_like(self.forward_inputs)

        grad_func = np.where(self.forward_inputs >= zeros, ones, zeros)
        grad_inputs = grad_outputs * grad_func

        return grad_inputs
        # your code here /\


# =============================== 2.1.2 Softmax ==============================
class Softmax(Layer):
    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d)), input values

        :return: np.array((n, d)), output values

            n - batch size
            d - number of units
        """
        # your code here \/

        # fixing E FloatingPointError('overflow encountered in exp')
        stable_data = inputs - np.max(inputs, axis=1, keepdims=True)

        exp_data = np.exp(stable_data)

        return exp_data / np.sum(exp_data, axis=1, keepdims=True)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d)), dLoss/dOutputs

        :return: np.array((n, d)), dLoss/dInputs

            n - batch size
            d - number of units
        """
        # your code here \/

        dot_product_vec = np.sum(self.forward_outputs * grad_outputs, axis=1, keepdims=True)
        grad_inputs = grad_outputs * self.forward_outputs - self.forward_outputs * dot_product_vec

        return grad_inputs
        # your code here /\


# ================================ 2.1.3 Dense ===============================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_units = units

        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        (input_units,) = self.input_shape
        output_units = self.output_units

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name="weights",
            shape=(output_units, input_units),
            initializer=he_initializer(input_units),
        )

        self.biases, self.biases_grad = self.add_parameter(
            name="biases",
            shape=(output_units,),
            initializer=np.zeros,
        )

        self.output_shape = (output_units,)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d)), input values

        :return: np.array((n, c)), output values

            n - batch size
            d - number of input units
            c - number of output units
        """
        # your code here \/
        return inputs @ self.weights.T + self.biases
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c)), dLoss/dOutputs

        :return: np.array((n, d)), dLoss/dInputs

            n - batch size
            d - number of input units
            c - number of output units
        """
        # your code here \/

        # if self.is_training:

        self.weights_grad = grad_outputs.T @ self.forward_inputs
        self.biases_grad = np.sum(grad_outputs, axis=0)

        self.update_parameters()

        return grad_outputs @ self.weights
        # your code here /\


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def value_impl(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values

        :return: np.array((1,)), mean Loss scalar for batch

            n - batch size
            d - number of units
        """
        # your code here \/

        stable_y_pred = np.where(y_pred >= eps, y_pred, eps)
        batch_of_cross_entropy = -np.sum(y_gt * np.log(stable_y_pred), axis=1)

        return np.array([np.mean(batch_of_cross_entropy)])
        # your code here /\

    def gradient_impl(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values

        :return: np.array((n, d)), dLoss/dY_pred

            n - batch size
            d - number of units
        """
        # your code here \/

        stable_y_pred = np.where(y_pred >= eps, y_pred, eps)
        batch_size = y_gt.shape[0]

        return -y_gt / stable_y_pred / batch_size
        # your code here /\


# ======================== 2.3 Train and Test on MNIST =======================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    model = Model(loss=CategoricalCrossentropy(), optimizer=SGDMomentum(lr=0.01, momentum=0.9))

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Dense(input_shape=(784,), units=392))
    model.add(ReLU())
    model.add(Dense(units=196))
    model.add(ReLU())
    model.add(Dense(units=98))
    model.add(ReLU())
    model.add(Dense(units=10))
    model.add(Softmax())

    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(
        x_train, y_train,
        batch_size=64,
        epochs=10,
        x_valid=x_valid,
        y_valid=y_valid
    )

    # your code here /\
    return model


# ============================== 3.3.2 convolve ==============================
def convolve(inputs, kernels, padding=0):
    """
    :param inputs: np.array((n, d, ih, iw)), input values
    :param kernels: np.array((c, d, kh, kw)), convolution kernels
    :param padding: int >= 0, the size of padding, 0 means 'valid'

    :return: np.array((n, c, oh, ow)), output values

        n - batch size
        d - number of input channels
        c - number of output channels
        (ih, iw) - input image shape
        (oh, ow) - output image shape
    """
    # !!! Don't change this function, it's here for your reference only !!!
    assert isinstance(padding, int) and padding >= 0
    assert inputs.ndim == 4 and kernels.ndim == 4
    assert inputs.shape[1] == kernels.shape[1]

    if os.environ.get("USE_FAST_CONVOLVE", False):
        return convolve_pytorch(inputs, kernels, padding)
    else:
        return convolve_numpy(inputs, kernels, padding)


def convolve_numpy(inputs, kernels, padding):
    """
    :param inputs: np.array((n, d, ih, iw)), input values
    :param kernels: np.array((c, d, kh, kw)), convolution kernels
    :param padding: int >= 0, the size of padding, 0 means 'valid'

    :return: np.array((n, c, oh, ow)), output values

        n - batch size
        d - number of input channels
        c - number of output channels
        (ih, iw) - input image shape
        (oh, ow) - output image shape
    """
    # your code here \/

    n, d, ih, iw = inputs.shape
    c, _, kh, kw = kernels.shape
    res_shape = (ih - kh + 2 * padding + 1, iw - kw + 2 * padding + 1)

    if padding > 0:
        inputs_padded = np.pad(inputs,
                               ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                               mode='constant', constant_values=0)
    else:
        inputs_padded = inputs

    res = np.zeros((n, c, res_shape[0], res_shape[1]))

    # (x * kernel)[n, m] = sum_{i, j}(x[i, j]kernel[n - i, m - j]) => have to flip kernel because [n - i, m - j]
    flipped_kernels = np.flip(kernels, axis=(2, 3))

    for i in range(res_shape[0]):
        for j in range(res_shape[1]):
            # window.shape == (n, d, kh, kw)
            window = inputs_padded[:, :, i: i + kh, j: j + kw]

            # expanded_window.shape == (n, 1, d, kh, kw)
            expanded_window = window[:, None, :, :, :]
            # expanded_kernels.shape == (1, c, d, kh, kw)
            expanded_kernels = flipped_kernels[None, :, :, :, :]

            # tmp.shape == (n, c, d, kh, kw)
            tmp = expanded_window * expanded_kernels
            res[:, :, i, j] = np.sum(tmp, axis=(2, 3, 4))

    return res
    # your code here /\


# =============================== 4.1.1 Conv2D ===============================
class Conv2D(Layer):
    def __init__(self, output_channels, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2, "Kernel size should be odd"

        self.output_channels = output_channels
        self.kernel_size = kernel_size

        self.kernels, self.kernels_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        output_channels = self.output_channels
        kernel_size = self.kernel_size

        self.kernels, self.kernels_grad = self.add_parameter(
            name="kernels",
            shape=(output_channels, input_channels, kernel_size, kernel_size),
            initializer=he_initializer(input_h * input_w * input_channels),
        )

        self.biases, self.biases_grad = self.add_parameter(
            name="biases",
            shape=(output_channels,),
            initializer=np.zeros,
        )

        self.output_shape = (output_channels,) + self.input_shape[1:]

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, c, h, w)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (h, w) - image shape
        """
        # your code here \/

        # SAME convolve strategy
        p = (self.kernels.shape[2] - 1) // 2
        expanded_biases = self.biases[None, :, None, None]

        return convolve(inputs, self.kernels, p) + expanded_biases
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of input channels
            c - number of output channels
            (h, w) - image shape
        """
        # your code here \/

        p = (self.kernels.shape[2] - 1) // 2
        c, _, kh, kw = self.kernels.shape

        self.biases_grad = np.sum(grad_outputs, axis=(0, 2, 3))

        inputs_padded = np.pad(self.forward_inputs,
                               ((0, 0), (0, 0), (p, p), (p, p)),
                               mode='constant', constant_values=0)

        kernels_grad_tmp = np.zeros_like(self.kernels)

        for i in range(kh):
            for j in range(kw):
                windows = inputs_padded[:, :, i: i + grad_outputs.shape[2], j: j + grad_outputs.shape[3]]

                # expanded_window.shape == (n, 1, d, h, w)
                expanded_windows = windows[:, None, :, :, :]
                # expanded_grad.shape == (1, c, d, h, w)
                expanded_grad = grad_outputs[:, :, None, :, :]

                # tmp.shape == (n, c, d, h, w)
                tmp = expanded_windows * expanded_grad
                kernels_grad_tmp[:, :, i, j] = np.sum(tmp, axis=(0, 3, 4))

        self.kernels_grad = np.flip(kernels_grad_tmp, axis=(2, 3))

        self.update_parameters()

        flipped_kernels = np.flip(self.kernels, axis=(2, 3)).transpose(1, 0, 2, 3)
        grad_inputs = convolve(grad_outputs, flipped_kernels, p)

        return grad_inputs
        # your code here /\


# ============================== 4.1.2 Pooling2D =============================
class Pooling2D(Layer):
    def __init__(self, pool_size=2, pool_mode="max", *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert pool_mode in {"avg", "max"}

        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.forward_idxs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        channels, input_h, input_w = self.input_shape
        output_h, rem_h = divmod(input_h, self.pool_size)
        output_w, rem_w = divmod(input_w, self.pool_size)
        assert not rem_h, "Input height should be divisible by the pool size"
        assert not rem_w, "Input width should be divisible by the pool size"

        self.output_shape = (channels, output_h, output_w)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, ih, iw)), input values

        :return: np.array((n, d, oh, ow)), output values

            n - batch size
            d - number of channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
        """
        # your code here \/

        p = self.pool_size
        n, d, ph, pw = inputs.shape
        h, w = np.array([ph, pw]) // np.array([p, p])

        # if self.pool_mode == 'max':
        #     self.forward_idxs = np.zeros((n, d, h, w, 2))
        #
        # # cinema coding
        # for batch in range(n):
        #     for channel in range(d):
        #         for i in range(h):
        #             for j in range(w):
        #
        #                 window = inputs[batch, channel, p * i: p * (i + 1), p * j: p * (j + 1)]
        #
        #                 if self.pool_mode == 'avg':
        #                     res[batch, channel, i, j] = np.mean(window)
        #                 else:
        #                     res[batch, channel, i, j] = np.max(window)
        #
        #                     max_idx = np.unravel_index(np.argmax(window), window.shape)
        #                     self.forward_idxs[batch, channel, i, j] = [max_idx[0], max_idx[1]]

        # reshaped_inputs.shape == (n, d, h, p, w, p)
        reshaped_inputs = inputs.reshape(n, d, h, p, w, p)
        # inputs_blocks.shape == (n, d, h, w, p, p)
        inputs_blocks = reshaped_inputs.transpose(0, 1, 2, 4, 3, 5)

        if self.pool_mode == 'avg':

            res = inputs_blocks.mean(axis=(4, 5))

        else:

            res = inputs_blocks.max(axis=(4, 5))

            tmp_idxs = np.argmax(inputs_blocks.reshape(n, d, h, w, p*p), axis=-1)
            row_idxs = tmp_idxs // p
            col_idxs = tmp_idxs % p

            self.forward_idxs = np.stack([row_idxs, col_idxs], axis=-1)

        return res
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d, oh, ow)), dLoss/dOutputs

        :return: np.array((n, d, ih, iw)), dLoss/dInputs

            n - batch size
            d - number of channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
        """

        # your code here \/

        # again cinema coding
        def avg_mode():

            res = np.repeat(np.repeat(grad_outputs, p, axis=2), p, axis=3) / (p * p)

            return res

        def max_mode():

            res = np.zeros((n, d, ph, pw))

            max_row_idxs = self.forward_idxs[:, :, :, :, 0].astype(int)
            max_col_idxs = self.forward_idxs[:, :, :, :, 1].astype(int)

            batch_idxs = np.arange(n)[:, None, None, None]
            channel_idxs = np.arange(d)[None, :, None, None]
            i_idxs = np.arange(h)[None, None, :, None]
            j_idxs = np.arange(w)[None, None, None, :]

            res[batch_idxs, channel_idxs, p * i_idxs + max_row_idxs, p * j_idxs + max_col_idxs] = grad_outputs

            return res

        n, d, h, w = grad_outputs.shape
        p = self.pool_size
        ph, pw = h * p, w * p

        if self.pool_mode == 'avg':

            grad_inputs = avg_mode()
        else:

            grad_inputs = max_mode()

        return grad_inputs
        # your code here /\


# ============================== 4.1.3 BatchNorm =============================
class BatchNorm(Layer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

        self.running_mean = None
        self.running_var = None

        self.beta, self.beta_grad = None, None
        self.gamma, self.gamma_grad = None, None

        self.forward_inverse_std = None
        self.forward_centered_inputs = None
        self.forward_normalized_inputs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        self.running_mean = np.zeros((input_channels,))
        self.running_var = np.ones((input_channels,))

        self.beta, self.beta_grad = self.add_parameter(
            name="beta",
            shape=(input_channels,),
            initializer=np.zeros,
        )

        self.gamma, self.gamma_grad = self.add_parameter(
            name="gamma",
            shape=(input_channels,),
            initializer=np.ones,
        )

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, d, h, w)), output values

            n - batch size
            d - number of channels
            (h, w) - image shape
        """
        # your code here \/

        # correct broadcasting
        gamma_ = self.gamma[None, :, None, None]
        beta_ = self.beta[None, :, None, None]

        if self.is_training:

            # correct broadcasting
            mu = np.mean(inputs, axis=(0, 2, 3), keepdims=True)
            nu = np.var(inputs, axis=(0, 2, 3), keepdims=True)

            self.forward_centered_inputs = inputs - mu
            self.forward_inverse_std = 1.0 / np.sqrt(eps + nu)
            self.forward_normalized_inputs = self.forward_centered_inputs * self.forward_inverse_std

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu.reshape(-1)
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * nu.reshape(-1)

            res = gamma_ * self.forward_normalized_inputs + beta_

        else:

            # correct broadcasting
            running_mean_ = self.running_mean[None, :, None, None]
            running_var_ = self.running_var[None, :, None, None]

            inverse_std = 1.0 / np.sqrt(eps + running_var_)
            centered_inputs = inputs - running_mean_
            normalized_inputs = centered_inputs * inverse_std

            res = gamma_ * normalized_inputs + beta_

        return res
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d, h, w)), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of channels
            (h, w) - image shape
        """
        # your code here \/

        # correct broadcasting
        gamma_ = self.gamma[None, :, None, None]

        n, _, h, w = grad_outputs.shape

        centered_inputs = self.forward_centered_inputs
        inverse_std = self.forward_inverse_std
        normalized_inputs = self.forward_normalized_inputs

        self.beta_grad = np.sum(grad_outputs, axis=(0, 2, 3))
        self.gamma_grad = np.sum(grad_outputs * normalized_inputs, axis=(0, 2, 3))

        self.update_parameters()

        norm_x_grad = grad_outputs * gamma_

        nu_grad = np.sum(-0.5 * norm_x_grad * centered_inputs * (inverse_std ** 3), axis=(0, 2, 3), keepdims=True)

        coef = n * h * w
        mu_grad = np.sum(norm_x_grad * (-inverse_std), axis=(0, 2, 3), keepdims=True) + \
                  nu_grad * np.sum(-2 * centered_inputs, axis=(0, 2, 3), keepdims=True) / coef

        grad_inputs = norm_x_grad * inverse_std + 2 * centered_inputs * nu_grad / coef + mu_grad / coef

        return grad_inputs
        # your code here /\


# =============================== 4.1.4 Flatten ==============================
class Flatten(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = (int(np.prod(self.input_shape)),)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, (d * h * w))), output values

            n - batch size
            d - number of input channels
            (h, w) - image shape
        """
        # your code here \/

        n, d, h, w = inputs.shape
        self._input_shape = inputs.shape

        # res = np.zeros(shape=(n, d * h * w))
        # for batch in range(n):
        #     res[batch, :] = np.ravel(inputs[batch, :, :, :])

        res = inputs.reshape(n, d * h * w)

        return res
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, (d * h * w))), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of units
            (h, w) - input image shape
        """
        # your code here \/

        grad_inputs = grad_outputs.reshape(self._input_shape)

        return grad_inputs
        # your code here /\


# =============================== 4.1.5 Dropout ==============================
class Dropout(Layer):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.forward_mask = None

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values

        :return: np.array((n, ...)), output values

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        if self.is_training:

            # P(u > p) = 1 - P(u <= p) = 1 - p => P(True) == 1 - p, P(False) = p
            uniform_ = np.random.uniform(0, 1, inputs.shape)
            self.forward_mask = (uniform_ > self.p).astype(np.float64)

            res = self.forward_mask * inputs

        else:

            res = (1 - self.p) * inputs

        return res
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

        :return: np.array((n, ...)), dLoss/dInputs

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/

        grad_inputs = grad_outputs * self.forward_mask

        return grad_inputs
        # your code here /\


# ====================== 2.3 Train and Test on CIFAR-10 ======================
def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    model = Model(loss=CategoricalCrossentropy(), optimizer=SGDMomentum(lr=0.01, momentum=0.9))

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)

    model.add(Conv2D(input_shape=(3, 32, 32), output_channels=24))
    model.add(BatchNorm())
    model.add(ReLU())
    model.add(Pooling2D(pool_size=2))

    model.add(Conv2D(output_channels=40))
    model.add(BatchNorm())
    model.add(ReLU())
    model.add(Pooling2D(pool_size=2))
    model.add(Dropout(p=0.5))

    model.add(Flatten())

    model.add(Dense(units=1024))
    model.add(ReLU())
    model.add(Dense(units=10))
    model.add(Softmax())

    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=6,
        x_valid=x_valid,
        y_valid=y_valid
    )

    # your code here /\
    return model

# ============================================================================
