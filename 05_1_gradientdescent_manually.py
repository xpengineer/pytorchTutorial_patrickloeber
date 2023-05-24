import numpy as np 

# Compute every step manually

# Linear regression
# f = w * x 

# here : f = 2 * x
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 2.0

print(w*X)



# model output
def forward(x):
    return w * x

# loss = MSE
def loss(y, y_pred):
    return ((y_pred - y)**2).mean()  # same as: np.mean((y_pred - y)**2)

# J = MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N * 2x(w*x - y)
def gradient(x, y, y_pred):
    # wrong way
    # np.dot(2*x, y_pred - y).mean()
    dot = np.dot(2*x, y_pred - y)  # XP: it is sum. a single number.
    sum = (2*x * (y_pred - y)).sum()  # XP: it is sum. a single number.
    print(f'dot: {dot}')
    print(f'dot.mean(): {dot.mean()}')  # no effect
    print(f'sum: {sum}')  # correct
    assert abs(dot - sum) < 1e4

    # correct way
    # np.mean(2 * x * (y_pred - y))
    z = 2*x * (y_pred - y)
    print(f'z: {z}')
    print(f'z.mean(): {z.mean()}')
    return z.mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 20

for epoch in range(n_iters):
    # predict = forward pass
    y_pred = forward(X)

    # loss
    # l = loss(Y, y_pred)  # XP: not needed
    
    # calculate gradients
    dw = gradient(X, Y, y_pred)

    # update weights
    w -= learning_rate * dw

    if epoch % 2 == 0:
        # print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
        print(f'epoch {epoch+1}: w = {w:.3f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')
print(len(X))
