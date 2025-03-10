import math


def func(magnitude, angle):
    x = magnitude * math.cos(angle * math.pi / 180)
    y = magnitude * math.sin(angle * math.pi / 180)
    return x + 1j * y


print(func(12.99410626, 79.9745932))
