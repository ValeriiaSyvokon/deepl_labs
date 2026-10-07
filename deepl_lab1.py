import numpy as np
from sklearn.datasets import load_iris
import torch

torch.set_default_dtype(torch.float64)



iris = load_iris()

X = iris.data.astype(np.float64)
y = iris.target

# print("X shape:", X.shape)
# print("y shape:", y.shape)

#---------------------------------------------

rng = np.random.default_rng(0)

train_indices = []
test_indices = []

# окрема обробка кожного класу
for cls in range(3):
    indices = np.where(y == cls)[0]

    rng.shuffle(indices)

    train_indices.extend(indices[:35])

    test_indices.extend(indices[35:])

train_indices = np.array(train_indices)
test_indices = np.array(test_indices)

X_train = X[train_indices]
y_train = y[train_indices]

X_test = X[test_indices]
y_test = y[test_indices]

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

print("\nКількість класів у train:", np.bincount(y_train))
print("Кількість класів у test:", np.bincount(y_test))

#---------------------------------------------

mean = X_train.mean(axis=0)
std = X_train.std(axis=0, ddof=0)

# стандартизація
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

print("\nСереднє:", mean)
print("Стандартне відхилення:", std)

print("\nПеревірка середніх train:")
print(X_train.mean(axis=0))

print("\nПеревірка стандартних відхилень train:")
print(X_train.std(axis=0, ddof=0))

#---------------------------------------------

rng_init = np.random.default_rng(0)

# шари
input_size = 4
hidden_size = 8
output_size = 3

# He initialization для W1
he_std = np.sqrt(2.0 / input_size)
W1 = rng_init.normal(
    loc=0.0,
    scale=he_std,
    size=(input_size, hidden_size)
).astype(np.float64)

# нульовий bias першого шару
b1 = np.zeros(hidden_size, dtype=np.float64)

# Xavier initialization для W2
xavier_std = np.sqrt(2.0 / (hidden_size + output_size))
W2 = rng_init.normal(
    loc=0.0,
    scale=xavier_std,
    size=(hidden_size, output_size)
).astype(np.float64)

# нульовий bias другого шару
b2 = np.zeros(output_size, dtype=np.float64)

#---------------------------------------------

print("\nПараметри мережі:")

print("W1 shape:", W1.shape)
print("b1 shape:", b1.shape)
print("W2 shape:", W2.shape)
print("b2 shape:", b2.shape)

print("\nТипи даних:")
print("X_train:", X_train.dtype)
print("W1:", W1.dtype)
print("b1:", b1.dtype)
print("W2:", W2.dtype)
print("b2:", b2.dtype)

#---------------------------------------------

def relu(x):
    return np.maximum(0, x)


def log_softmax(z):
    z_max = np.max(z, axis=1, keepdims=True)
    shifted = z - z_max

    log_sum_exp = np.log(
        np.sum(np.exp(shifted), axis=1, keepdims=True)
    )

    return shifted - log_sum_exp


def cross_entropy(log_probs, y):
    n = y.shape[0]

    loss = -np.mean(log_probs[np.arange(n), y])

    return loss


def forward(X, y, W1, b1, W2, b2):
    # 1-ший лінійний шар
    Z1 = X @ W1 + b1

    A1 = relu(Z1)

    # 2-гий лінійний шар
    Z2 = A1 @ W2 + b2

    log_probs = log_softmax(Z2)

    loss = cross_entropy(log_probs, y)

    cache = {
        "X": X,
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "log_probs": log_probs
    }

    return loss, cache


# перевірка прямого проходу
loss, cache = forward(
    X_train,
    y_train,
    W1,
    b1,
    W2,
    b2
)

print("\nРезультат прямого проходу:")
print("Loss:", loss)

print("\nРозмірності проміжних значень:")
print("X:", cache["X"].shape)
print("Z1:", cache["Z1"].shape)
print("A1:", cache["A1"].shape)
print("Z2:", cache["Z2"].shape)
print("log_probs:", cache["log_probs"].shape)

#---------------------------------------------

def backward(y, W2, cache, divide_by_n=True):
    X = cache["X"]
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    log_probs = cache["log_probs"]

    N = X.shape[0]

    # градієнт за логітами для softmax + cross-entropy
    probs = np.exp(log_probs)

    dZ2 = probs.copy()
    dZ2[np.arange(N), y] -= 1
    if divide_by_n:
        dZ2 /= N

    # градієнти другого шару
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)

    dA1 = dZ2 @ W2.T

    dZ1 = dA1 * (Z1 > 0)

    # градієнти першого шару
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    return dW1, db1, dW2, db2


dW1, db1, dW2, db2 = backward(
    y_train,
    W2,
    cache
)

print("\nГрадієнти:")

print("dW1 shape:", dW1.shape)
print("db1 shape:", db1.shape)
print("dW2 shape:", dW2.shape)
print("db2 shape:", db2.shape)

print("\nПеревірка скінченності:")
print("dW1:", np.all(np.isfinite(dW1)))
print("db1:", np.all(np.isfinite(db1)))
print("dW2:", np.all(np.isfinite(dW2)))
print("db2:", np.all(np.isfinite(db2)))

#---------------------------------------------


model = torch.nn.Sequential(
    torch.nn.Linear(4, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 3)
)

with torch.no_grad():
    model[0].weight.copy_(torch.tensor(W1.T))
    model[0].bias.copy_(torch.tensor(b1))

    model[2].weight.copy_(torch.tensor(W2.T))
    model[2].bias.copy_(torch.tensor(b2))

X_train_torch = torch.tensor(X_train)
y_train_torch = torch.tensor(y_train, dtype=torch.long)

logits = model(X_train_torch)

criterion = torch.nn.CrossEntropyLoss()
loss_torch = criterion(logits, y_train_torch)

loss_torch.backward()


dW1_torch = model[0].weight.grad.detach().numpy().T
db1_torch = model[0].bias.grad.detach().numpy()
dW2_torch = model[2].weight.grad.detach().numpy().T
db2_torch = model[2].bias.grad.detach().numpy()



print("\nПорівняння NumPy та PyTorch:")
print("Loss NumPy:   ", loss)
print("Loss PyTorch: ", loss_torch.item())
print("\nМаксимальні абсолютні різниці:")

print("W1:",
    np.max(np.abs(dW1 - dW1_torch)))

print("b1:",
    np.max(np.abs(db1 - db1_torch)))

print("W2:",
    np.max(np.abs(dW2 - dW2_torch)))

print("b2:",
    np.max(np.abs(db2 - db2_torch)))


tolerance = 1e-12

loss_diff = abs(loss - loss_torch.item())

w1_diff = np.max(np.abs(dW1 - dW1_torch))
b1_diff = np.max(np.abs(db1 - db1_torch))
w2_diff = np.max(np.abs(dW2 - dW2_torch))
b2_diff = np.max(np.abs(db2 - db2_torch))

print("\nПеревірка:")
print("Loss:",
    np.isfinite(loss) and
    np.isfinite(loss_torch.item()) and
    loss_diff <= tolerance)

print("dW1:",
    np.all(np.isfinite(dW1)) and
    np.all(np.isfinite(dW1_torch)) and
    w1_diff <= tolerance)

print("db1:",
    np.all(np.isfinite(db1)) and
    np.all(np.isfinite(db1_torch)) and
    b1_diff <= tolerance)

print("dW2:",
    np.all(np.isfinite(dW2)) and
    np.all(np.isfinite(dW2_torch)) and
    w2_diff <= tolerance)

print("db2:",
    np.all(np.isfinite(db2)) and
    np.all(np.isfinite(db2_torch)) and
    b2_diff <= tolerance)

#---------------------------------------------

# чисельна перевірка градієнтів

epsilon = 1e-6


def numerical_gradient(param, index, loss_function):
    original_value = param[index]

    param[index] = original_value + epsilon
    loss_plus = loss_function()

    param[index] = original_value - epsilon
    loss_minus = loss_function()

    param[index] = original_value

    return (loss_plus - loss_minus) / (2 * epsilon)


def calculate_loss():
    loss, _ = forward(
        X_train,
        y_train,
        W1,
        b1,
        W2,
        b2
    )
    return loss


num_W1 = numerical_gradient(
    W1,
    (0, 0),
    calculate_loss
)

num_b1 = numerical_gradient(
    b1,
    0,
    calculate_loss
)

num_W2 = numerical_gradient(
    W2,
    (0, 0),
    calculate_loss
)

num_b2 = numerical_gradient(
    b2,
    0,
    calculate_loss
)


# ручні градієнти для тих самих параметрів
manual_W1 = dW1[0, 0]
manual_b1 = db1[0]
manual_W2 = dW2[0, 0]
manual_b2 = db2[0]


print("\nЧисельна перевірка градієнтів:")
print("epsilon:", epsilon)

print("\nПараметр      Чисельний       Ручний          Різниця")
print(
    f"W1[0,0]   {num_W1:.12f}   {manual_W1:.12f}   "
    f"{abs(num_W1 - manual_W1):.12e}"
)
print(
    f"b1[0]     {num_b1:.12f}   {manual_b1:.12f}   "
    f"{abs(num_b1 - manual_b1):.12e}"
)
print(
    f"W2[0,0]   {num_W2:.12f}   {manual_W2:.12f}   "
    f"{abs(num_W2 - manual_W2):.12e}"
)
print(
    f"b2[0]     {num_b2:.12f}   {manual_b2:.12f}   "
    f"{abs(num_b2 - manual_b2):.12e}"
)



numerical_tolerance = 1e-7

print("\nПеревірка:")
print(
    "W1[0,0]:",
    abs(num_W1 - manual_W1) <= numerical_tolerance
)
print(
    "b1[0]:",
    abs(num_b1 - manual_b1) <= numerical_tolerance
)
print(
    "W2[0,0]:",
    abs(num_W2 - manual_W2) <= numerical_tolerance
)
print(
    "b2[0]:",
    abs(num_b2 - manual_b2) <= numerical_tolerance
)