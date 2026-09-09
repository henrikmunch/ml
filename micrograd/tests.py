from micrograd import Value, Neuron, Layer, MLP

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
e = a*b
d = e + c
f = Value(-2.0)
f = f * 2
L = d*f
L.backward()
print(L)

a = Value(3.0)
b = a + a
b = b**2
b / 3
b.backward()
print(a)
print(b)

x = [2.0, 3.0]
n = Neuron(2)
print(n(x))

m = Layer(2, 3)
print(m(x))

# 3d input
# 2 hidden layers of dim 4
# 1d output
x = [2.0, -1.2, 3.0]
n = MLP(2, [4, 4, 1])
print("MPL:" )
print(n(x))