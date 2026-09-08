import math
import numpy as np
import matplotlib.pyplot as plt


class Value:

    def __init__(self, data, _children=(), _operation=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._previous = set(_children)
        self._operation = _operation

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
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


a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
e = a*b
d = e + c
f = Value(-2.0)
L = d*f
L.backward()
print(L)

a = Value(3.0)
b = a + a
b.backward()
print(a)