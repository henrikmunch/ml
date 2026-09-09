import math
import numpy as np
import random


# =================================================================


class Value:

    def __init__(self, data, _children=(), _operation=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._previous = set(_children)
        self._operation = _operation


    def __repr__(self): # print
        return f"Value(data={self.data}, grad={self.grad})"


    def __add__(self, other): # self + other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out


    def __radd__(self, other): # other + self
        return self + other


    def __neg__(self): # -self
        return self * -1


    def __sub__(self, other): # self - other
        return self + (-other)


    def __mul__(self, other): # self * other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out


    def __rmul__(self, other): # other * self
        return self * other


    def __pow__(self, other): # self ^ other
        assert isinstance(other, (int, float)), "only supporting int/float powers (other is not Value)"
        out = Value(self.data**other, (self, ), f'**{other}')

        def _backward():
            self.grad += other * (out.data**(other-1)) * out.grad
        out._backward = _backward

        return out


    def __truediv__(self, other): # self / other
        return self * other**(-1)



    def __rtruediv__(self, other): # other / self
        return other * self**-1


    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out


    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out


    def backward(self):

        # topological ordering
        topo = []
        visited = set()
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._previous:
                    build_topo(child)
                topo.append(node)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


# =================================================================


class Neuron:

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))


    def __call__(self, x): # activation = w.x + b
       # call  allows notation n(x) if n = Neuron(nin)
       act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
       out = act.tanh()
       return out


# =================================================================


class Layer:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]


    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs


# =================================================================


class MLP:

    def __init__(self, nin, nouts):
        size = [nin] + nouts
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(nouts))]


    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x