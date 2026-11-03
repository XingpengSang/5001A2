# ============================================================
# Question 10: Plot the Estimated Regression Curve
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# 1. Define the x values
x = np.linspace(-2, 2, 500)

# 2. Calculate the estimated regression curve
# Y = 1 + X, when X < 1
# Y = 1 + X - 2(X - 1)^2, when X >= 1

y = np.where(
    x < 1,
    1 + x,
    1 + x - 2 * (x - 1)**2
)

# 3. Calculate important points
x_intercept = (-1, 0)
y_intercept = (0, 1)
knot = (1, 2)
vertex = (1.25, 2.125)
endpoint = (2, 1)

# 4. Plot the curve
plt.figure(figsize=(9, 6))

# First segment: linear function
x1 = np.linspace(-2, 1, 200)
y1 = 1 + x1
plt.plot(x1, y1, linewidth=2.5, label=r'$\hat{Y}=1+X,\ X<1$')

# Second segment: quadratic function
x2 = np.linspace(1, 2, 200)
y2 = 1 + x2 - 2 * (x2 - 1)**2
plt.plot(x2, y2, linewidth=2.5,
         label=r'$\hat{Y}=1+X-2(X-1)^2,\ X\geq1$')

# 5. Mark important points
points = [x_intercept, y_intercept, knot, vertex, endpoint]

for point in points:
    plt.scatter(point[0], point[1], s=55, zorder=3)

# 6. Add annotations
plt.annotate('X-intercept (-1, 0)',
             xy=x_intercept, xytext=(-1.9, 0.5),
             arrowprops=dict(arrowstyle='->'))

plt.annotate('Y-intercept (0, 1)',
             xy=y_intercept, xytext=(-0.9, 1.5),
             arrowprops=dict(arrowstyle='->'))

plt.annotate('Knot (1, 2)',
             xy=knot, xytext=(0.3, 2.5),
             arrowprops=dict(arrowstyle='->'))

plt.annotate('Maximum (1.25, 2.125)',
             xy=vertex, xytext=(1.35, 2.65),
             arrowprops=dict(arrowstyle='->'))

plt.annotate('(2, 1)',
             xy=endpoint, xytext=(1.65, 0.6),
             arrowprops=dict(arrowstyle='->'))

# 7. Set axes and figure properties
plt.axhline(y=0, linewidth=0.8)
plt.axvline(x=0, linewidth=0.8)

plt.xlim(-2, 2)
plt.ylim(-1.3, 3)

plt.xticks(np.arange(-2, 2.1, 0.5))
plt.yticks(np.arange(-1, 3.1, 0.5))

plt.xlabel('X', fontsize=12)
plt.ylabel(r'Estimated response $\hat{Y}$', fontsize=12)
plt.title('Question 10: Estimated Regression Curve', fontsize=14)

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

# 8. Save and display the figure
plt.savefig('Q10_estimated_curve.png', dpi=300, bbox_inches='tight')
plt.show()