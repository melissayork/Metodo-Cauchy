import numpy as np
import math
from abc import ABC, abstractmethod

def regla_eliminacion(x1, x2, fx1, fx2, a, b) -> tuple[float, float]:
    if fx1 > fx2:
        return x1, b
    if fx1 < fx2:
        return a, x2
    return x1, x2 

def w_to_x(w: float, a, b) -> float:
    return w * (b - a) + a 

def gradiente(f, x, deltaX=0.001):
    grad = []
    for i in range(len(x)):
        xp = x.copy()
        xn = x.copy()
        xp[i] += deltaX
        xn[i] -= deltaX
        grad.append((f(xp) - f(xn)) / (2 * deltaX))
    return grad

class Optimizador(ABC):
    def __init__(self, funcion: callable) -> None:
        super().__init__()
        self.funcion = funcion

    @abstractmethod
    def optimizar(self, *args, **kwargs):
        pass

class MetodoUnivariable(ABC):
    @abstractmethod
    def optimizar(self, funcion, epsilon: float, a: float = None, b: float = None) -> float:
        pass

class GoldenSearch(MetodoUnivariable):
    def optimizar(self, funcion, epsilon: float, a: float = None, b: float = None) -> float:
        phi = (1 + math.sqrt(5)) / 2 - 1
        aw, bw = 0, 1
        Lw = 1
        while Lw > epsilon:
            w2 = aw + phi * Lw
            w1 = bw - phi * Lw
            aw, bw = regla_eliminacion(w1, w2, funcion(w_to_x(w1, a, b)), funcion(w_to_x(w2, a, b)), aw, bw)
            Lw = bw - aw
        return (w_to_x(aw, a, b) + w_to_x(bw, a, b)) / 2

class Fibonacci(MetodoUnivariable):
    def optimizar(self, funcion, epsilon: float, a: float = None, b: float = None) -> float:
        n = 10  # Número de iteraciones de Fibonacci, puede ajustarse
        L = b - a
        fib = [0, 1]
        while len(fib) <= n + 2:
            fib.append(fib[-1] + fib[-2])
        k = 2
        while k < n:
            Lk = (fib[n - k + 2] / fib[n + 2]) * L
            x1 = a + Lk
            x2 = b - Lk
            fx1 = funcion(x1)
            fx2 = funcion(x2)
            if fx1 < fx2:
                b = x2
            elif fx1 > fx2:
                a = x1
            elif fx1 == fx2:
                a, b = x1, x2
            k += 1
        return (a + b) / 2

class Cauchy(Optimizador):
    def __init__(self, funcion: callable) -> None:
        super().__init__(funcion)

    def optimizar(self, x0, epsilon1, epsilon2, M, metodo_univariable: MetodoUnivariable):
        terminar = False
        xk = x0
        k = 0

        while not terminar:
            grad = np.array(gradiente(self.funcion, xk))
            if np.linalg.norm(grad) < epsilon1 or k >= M:
                terminar = True
            else:
                def alpha_funcion(alpha):
                    return self.funcion(xk - alpha * grad)
                
                alpha = metodo_univariable.optimizar(alpha_funcion, epsilon=epsilon2, a=0.0, b=1.0)
                x_k1 = xk - alpha * grad

                if np.linalg.norm(x_k1 - xk) / (np.linalg.norm(xk) + 0.00001) <= epsilon2:
                    terminar = True
                else:
                    k += 1
                    xk = x_k1
        return xk

x0 = np.array([0.0, 0.0])
e1 = 0.001
e2 = 0.001
max_iter = 100

himmenblau = lambda x: (((x[0] ** 2) + x[1] - 11) ** 2) + ((x[0] + (x[1] ** 2) - 7) ** 2)

metodo_univariable_fib = Fibonacci()
metodo_univariable_golden = GoldenSearch()

optimizador_fib = Cauchy(himmenblau)
optimizador_golden = Cauchy(himmenblau)

resultado_fib = optimizador_fib.optimizar(x0, e1, e2, max_iter, metodo_univariable_fib)
resultado_golden = optimizador_golden.optimizar(x0, e1, e2, max_iter, metodo_univariable_golden)

print("Resultado con Fibonacci:", resultado_fib)
print("Resultado con Golden Search:", resultado_golden)
