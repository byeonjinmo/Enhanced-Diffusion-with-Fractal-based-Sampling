import numpy as np
import matplotlib.pyplot as plt

# Define the transformation functions
def transformation_1(p):
    return p/2

def transformation_2(p):
    return (p + np.array([1, 0]))/2

def transformation_3(p):
    return (p + np.array([0.5, np.sqrt(3)/2]))/2

# Initial point
p = np.array([0, 0])

# Number of iterations
iterations = 10000

# For storing the points
x = [p[0]]
y = [p[1]]

# Iterate to create the fractal
for i in range(iterations):
    # Randomly choose a transformation and apply it
    transformation = np.random.choice([transformation_1, transformation_2, transformation_3])
    p = transformation(p)
    x.append(p[0])
    y.append(p[1])

# Plot the fractal
plt.plot(x, y, 'k.', markersize=1)
plt.title('Sierpinski Triangle Fractal')
plt.show()
