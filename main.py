from matplotlib import pyplot as plt
from scipy.integrate import quad
import numpy as np
from scipy.interpolate import interp1d
from numpy.polynomial import polynomial as P
import pandas as pd


# исходная функция
def f(t):
    return np.sqrt(t) * np.sin(t)


def print_graph(x, y, title, id, count_graphs):
    plt.subplot(1, count_graphs, id)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.title(title)
    plt.plot(x, y, '-o')


x_values = np.arange(1, 3.2, 0.2)
xk_values = np.arange(1.1, 3.0, 0.2)
integral_values = np.zeros(len(x_values))
f_values = np.array([f(x) for x in x_values])

# вычислим значение функции при помощи QUANC8 и сохраним результат в integral_values
for i, x in enumerate(x_values):
    integral_values[i], error = quad(f, 0, x, limit=30)

# По полученным точкам построим сплайн-функцию и полином Лагранжа 10-й степени
spline_func = interp1d(x_values, integral_values, kind='cubic')
lagrange_poly = P.Polynomial.fit(x_values, integral_values, 10)

# далее вычисляем значение в точках xk=1.1+0.2k
quanc8_values = np.zeros(len(xk_values))
for i, x in enumerate(xk_values):
    quanc8_values[i], error = quad(f, 0, x, limit=50)

spline_values = spline_func(xk_values)
lagrange_values = lagrange_poly(xk_values)

# вычислим погрешность
spline_errors = np.abs(spline_values - quanc8_values)
lagrange_errors = np.abs(lagrange_values - quanc8_values)

# представим полученный результат
results = pd.DataFrame({
    'xk': xk_values,
    'QUANC8 value': quanc8_values,
    'Spline value': spline_values,
    'Lagrange value': lagrange_values,
    'Spline error': spline_errors,
    'Lagrange error': lagrange_errors
})

print(results.to_string(index=False))

plt.figure(figsize=(15, 5))
print_graph(xk_values, quanc8_values, 'QUANC8 с высокой точностью', 1, 3)
print_graph(xk_values, lagrange_values, 'График Лагранжем', 2, 3)
print_graph(xk_values, spline_values, 'График сплайном', 3, 3)
plt.savefig("Graphs.jpg")
plt.show()

plt.figure(figsize=(15, 5))
print_graph(xk_values, lagrange_errors, 'Погрешность Лагранжем', 1, 2)
print_graph(xk_values, spline_errors, 'Погрешность сплайном', 2, 2)
plt.savefig("Error.jpg")
plt.show()



