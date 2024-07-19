# Laplace Equation Dirichlet Problem

The full solution can be found at <a href="https://github.com/rotmanfinhub/deep-macrofin/blob/develop/examples/basic_examples/basic_pdes.ipynb" target="_blank">basic_pdes.ipynb</a>.

## Problem Setup
$$\nabla^2 T = \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} = 0, T(x,0)=T(x,\pi)=0, T(0,y)=1$$

The solution is $T(x,y) = \frac{2}{\pi} \arctan\frac{\sin y}{\sinh x}$

## Implementation