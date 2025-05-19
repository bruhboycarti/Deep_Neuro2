import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Считываем данные
df = pd.read_csv('data.csv')

# Выделим целевую переменную
y = df.iloc[:, 4].values
y = np.where(y == "Iris-setosa", 1, -1)  # Преобразуем в численные значения

# Берем три признака
X = df.iloc[:, [0, 2, 3]].values

# Создаем 3D-график
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Отображаем точки для двух классов
ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], color='red', marker='o', label='Class 1 (Iris-setosa)')
ax.scatter(X[y == -1, 0], X[y == -1, 1], X[y == -1, 2], color='blue', marker='x', label='Class -1')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')

plt.legend()
plt.show()

# Функция нейрона
def neuron(w, x):
    if (w[3]*x[2] + w[1]*x[0] + w[2]*x[1] + w[0]) >= 0:
        return 1
    else: 
        return -1

# Начальные веса
w = np.random.random(4)
eta = 0.01  # скорость обучения
w_iter = []  # Список для хранения весов

# Процесс обучения
for xi, target, j in zip(X, y, range(X.shape[0])):
    predict = neuron(w, xi)
    w[1:] += (eta * (target - predict)) * xi  # обновление весов
    w[0] += eta * (target - predict)
    
    # Сохраняем веса каждые 10 шагов
    if j % 10 == 0:
        w_iter.append(w.tolist())

# Посчитаем ошибки
sum_err = 0
for xi, target in zip(X, y):
    predict = neuron(w, xi) 
    sum_err += (target - predict) / 2

print("Всего ошибок: ", sum_err)

# Визуализация процесса обучения в 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Отображаем данные
ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], color='red', marker='o', label='Class 1 (Iris-setosa)')
ax.scatter(X[y == -1, 0], X[y == -1, 1], X[y == -1, 2], color='blue', marker='x', label='Class -1')

# Визуализируем разделяющую гиперплоскость для каждой итерации обучения
xl = np.linspace(min(X[:, 0]), max(X[:, 0]), 10)
yl = np.linspace(min(X[:, 1]), max(X[:, 1]), 10)
X1, Y1 = np.meshgrid(xl, yl)
Z1 = -(w[1] * X1 + w[2] * Y1 + w[0]) / w[3]  # Уравнение гиперплоскости

# Для каждой итерации обучения рисуем поверхность
for i, w in zip(range(len(w_iter)), w_iter):
    Z1 = -(w[1] * X1 + w[2] * Y1 + w[0]) / w[3]
    ax.plot_surface(X1, Y1, Z1, alpha=0.3, rstride=100, cstride=100, color='gray')
    ax.text(xl[-1], yl[-1], Z1[-1, -1], f'Iter {i}', color='black', size=10)
    plt.pause(1)

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')

plt.legend()
plt.show()
