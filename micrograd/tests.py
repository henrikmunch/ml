from micrograd import Value, Neuron, Layer, MLP


# ============================= Misc =============================


a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
e = a*b
d = e + c
f = Value(-2.0)
f = f * 2
L = d*f
L.backward()
# print(L)

a = Value(3.0)
b = a + a
b = b**2
b / 3
b.backward()
# print(a)
# print(b)

x = [2.0, 3.0]
n = Neuron(2)
# print(n(x))

m = Layer(2, 3)
# print(m(x))


# ============================= Training example =============================

# 3d input
# 2 hidden layers of dim 4
# 1d output
n = MLP(3, [4, 4, 1])

# Data
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

# Target
ys = [1.0, -1.0, -1.0, 1.0]

# Gradient descent: 

learning_rate = 0.05
N_train = 20

for k in range(N_train):

    # Forward pass 
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # Number 3) in https://x.com/karpathy/status/1013244313327681536 :>
    # Reset grad to zero (like in its constructor), otherwise we keep piling gradient values
    for p in n.parameters():
        p.grad = 0.0

    # Backward pass 
    loss.backward()

    # Nudge parameters in negative gradient direction
    for p in n.parameters():
        p.data += (-1) * learning_rate * p.grad

    # Hopefully this number goes down
    print(k, loss.data)

print(f"ypred = {ypred}")