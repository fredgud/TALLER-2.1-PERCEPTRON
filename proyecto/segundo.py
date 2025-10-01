import math
import random

# ================================
# 1. Funciones de Activación
# ================================
def activation(name, x):
    """ Aplica la función de activación seleccionada """
    if name == "linear":   # Función lineal: salida directa
        return x
    if name == "step":     # Escalón: 0 o 1
        return 1 if x >= 0 else 0
    if name == "sigmoid":  # Sigmoide: probabilidad entre 0 y 1
        return 1 / (1 + math.exp(-x))
    if name == "relu":     # ReLU: rectificador (negativos a 0)
        return max(0, x)
    if name == "tanh":     # Tangente hiperbólica: entre -1 y 1
        return math.tanh(x)
    if name == "softmax":  # Softmax adaptada a binario (igual a sigmoide)
        return 1 / (1 + math.exp(-x))
    return x

# ================================
# 2. Clase Perceptrón
# ================================
class Perceptron:
    def __init__(self, input_size, activation="step", lr=0.1):
        # Pesos iniciales aleatorios
        self.w = [random.uniform(-1, 1) for _ in range(input_size)]
        self.b = random.uniform(-1, 1)
        self.lr = lr
        self.activation = activation

    def predict(self, x):
        """ Calcula la salida del perceptrón para una entrada x """
        z = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        return activation(self.activation, z)

    def train(self, X, y, epochs=20):
        """ Entrena el perceptrón ajustando pesos y sesgo """
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                y_pred = self.predict(xi)
                error = yi - y_pred
                # Regla de aprendizaje del perceptrón
                self.w = [wi + self.lr * error * xi_j for wi, xi_j in zip(self.w, xi)]
                self.b += self.lr * error

# ================================
# 3. Dataset: OR lógico
# ================================
# Entradas (dos valores binarios) y salidas esperadas
X = [[0,0], [0,1], [1,0], [1,1]]
y = [0, 1, 1, 1]  # OR: basta que al menos una entrada sea 1

# ================================
# 4. Entrenamiento y Evaluación
# ================================
activ_funcs = ["step", "sigmoid", "relu", "tanh", "linear", "softmax"]

for act in activ_funcs:
    print("\n============================")
    print(f"Función de activación: {act}")
    print("============================")

    # Creamos el perceptrón con la activación actual
    p = Perceptron(input_size=2, activation=act, lr=0.1)

    # Entrenamos por 20 iteraciones (>10 como pide la rúbrica)
    p.train(X, y, epochs=20)

    # Evaluamos con todas las combinaciones de entradas
    for xi in X:
        print(f"Entrada {xi} -> Predicción: {p.predict(xi)}")
