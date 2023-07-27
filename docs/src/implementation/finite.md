# Finite Element Method

This is a short introduction to the finite element method mostyl based on the documentation of [Ferrite.jl](https://ferrite-fem.github.io/Ferrite.jl/stable/manual/fe_intro/) with direct applications to the continuous model of power grids.
In general, the dynamics of the continuous model are given by the following PDEs

```math
m(\mathbf{r})\frac{\partial^2 \theta(\mathbf{r}, t)}{\partial t^2} + d(\mathbf{r}) \frac{\partial{\theta(\mathbf{r}, t)}}{\partial t} = P(\mathbf{r}) + \nabla\left(
        \begin{bmatrix}b_x(\mathbf{r}) & 0 \\ 0 & b_y(\mathbf{r})\end{bmatrix} \nabla \theta(\mathbf{r}, t)\right),
```

subject to the Neumann boundary conditions

```math
\mathbf{n} \left(\begin{bmatrix}b_x(\mathbf{r}) & 0 \\ 0 & b_y(\mathbf{r})\end{bmatrix} \nabla \theta(\mathbf{r}, t)\right) = 0,
```

with ``\mathbf{r} \in \Omega``.
The finite element method will be motivated using the static solution to this problem.

## Static Solution

To find the static solution, *i.e.*, the solution to the continuous version of the power flow equations, we need to solve the following differential equation

```math
\nabla \left(\begin{bmatrix}b_x(\mathbf{r}) & 0 \\ 0 & b_y(\mathbf{r})\end{bmatrix} \nabla \theta_0(\mathbf{r})\right) = -P(\mathbf{r})
```

subject to the same boundary conditions.

We begin by multiplying the equations by an arbitrary test function ``u(\mathbf{r})`` and integrate the resulting equations over the surface on which the PDEs are defined.
The left hand side yields

```math
\begin{align*}
    &\int u(\mathbf{r})\nabla \left(\begin{bmatrix}b_x(\mathbf{r}) & 0 \\ 0 & b_y(\mathbf{r})\end{bmatrix} \nabla \theta_0(\mathbf{r})\right)\, d\Omega \\
    & = \int u(\mathbf{r})\left(\begin{bmatrix}b_x(\mathbf{r}) & 0 \\ 0 & b_y(\mathbf{r})\end{bmatrix} \nabla \theta_0(\mathbf{r})\right)\mathbf{n}\,d(\partial\Omega) - \int \nabla u(\mathbf{r})\left(\begin{bmatrix}b_x(\mathbf{r}) & 0 \\ 0 & b_y(\mathbf{r})\end{bmatrix} \nabla \theta_0(\mathbf{r})\right) \,d \Omega\\
    & = - \int \nabla u(\mathbf{r})\left(\begin{bmatrix}b_x(\mathbf{r}) & 0 \\ 0 & b_y(\mathbf{r})\end{bmatrix} \nabla \theta_0(\mathbf{r})\right) \,d \Omega.
\end{align*}
```

The right hand side simply becomes

```math
-\int P(\mathbf{r}) u(\mathbf{r}) \, d\Omega.
```

After discretization we expand the functions in a set of base functions ``\phi(\mathbf{r})`` with the property ``\phi_i(\mathbf{r}_j) = \delta_{ij}``, where ``\mathbf{r}_j`` is the ``j``th nodal point.
The expansions read

```math
u(\mathbf{r}) \approx \sum_i\hat{u}_i\phi_i(\mathbf{r}),\qquad \theta_0(\mathbf{r}) \approx \sum_i\hat{\theta}_i\phi_i(\mathbf{r}).
```

This turns the PDE into

```math
\sum_{i, j}\hat{u}_i\hat{\theta}_j\int\nabla\phi_i(\mathbf{r})
\begin{bmatrix}b_x(\mathbf{r}) & 0 \\ 0 & b_y(\mathbf{r})\end{bmatrix}
\nabla\phi_j(\mathbf{r})\, d \Omega = \sum_{i} \hat{u}_i\int\phi_i(\mathbf{r})P(\mathbf{r})\,d\Omega.
```

In matrix form this can be written as

```math
\mathbf{\hat{u}}^\top \mathbf{K} \mathbf{\hat{\theta}} = \mathbf{\hat{u}}^\top\mathbf{f}.
```

As this equation needs to hold for arbitrary ``\mathbf{\hat{u}}`` we need to solve the linear system

```math
\mathbf{K} \mathbf{\hat{\theta}} = \mathbf{f},
```

where the stiffness matrix

```math
\mathbf{K}_{ij} = \int\nabla\phi_i(\mathbf{r})
\begin{bmatrix}b_x(\mathbf{r}) & 0 \\ 0 & b_y(\mathbf{r})\end{bmatrix}
\nabla\phi_j(\mathbf{r})\, d \Omega
```

and force vector

```math
\mathbf{f}_i = \int\phi_i(\mathbf{r})P(\mathbf{r})\,d\Omega
```

have been introduced.

Lastly, we need to look at how the integrals are calculated.
This is done by turning the integral into a sum of integrals over the elements ``E`` (in our case traingles) of the grid.
The integral is then solved using Gaussian quadrature

```math
\int\nabla\phi_i(\mathbf{r})
\begin{bmatrix}b_x(\mathbf{r}) & 0 \\ 0 & b_y(\mathbf{r})\end{bmatrix}
\nabla\phi_j(\mathbf{r})\, d \Omega = \sum_E\sum_i\omega_k \nabla\phi_i(\mathbf{q}_k)
\begin{bmatrix}b_x(\mathbf{q}_k) & 0 \\ 0 & b_y(\mathbf{q}_k)\end{bmatrix}
\nabla\phi_j(\mathbf{q}_k),
```

where ``\omega_i`` are the quadrature weights and ``\mathbf{q}_i`` the quadrature points.
The final trick is to use the base functions of a reference element and then just do a coordinate transformation to the actual coordinates.
This yields

```math
\int\nabla\phi_i(\mathbf{r})\, d\Omega = \sum_E\sum_k\omega_k J^{-1^\top}\widetilde{\nabla\phi}_i(\tilde{\mathbf{q}}_k)
\begin{bmatrix}b_x(\mathbf{q}_k) & 0 \\ 0 & b_y(\mathbf{q}_k)\end{bmatrix}
J^{-1^\top}\widetilde{\nabla\phi}_j(\tilde{\mathbf{q}}_k)\det(J),
```

where the quantities marked with a tilde are the reference quantities and $J$ is the Jacobian of the coordinate transform.

Using these results we can create the stiffness matrix.
The force vector can be created similarly using

```math
\int\phi_i(\mathbf{r})P(\mathbf{r}) \, d\Omega = \sum_E\sum_k\omega_k J^{-1^\top}\tilde{\phi}_i(\tilde{\mathbf{q}}_k) P(\mathbf{q}_k)\det(\mathbf{J}).
```

## Dynamic Solution

We start by rewriting the second order differential equation as a system of two first order differential equations

```math
\begin{bmatrix}1&0\\0&m(\mathbf{r})\end{bmatrix}\frac{d}{dt}\begin{bmatrix}\theta(\mathbf{r})\\\omega(\mathbf{r}) \end{bmatrix}=
\begin{bmatrix}\omega(\mathbf{r})\\-d(\mathbf{r})\omega(\mathbf{r}) + \nabla(\mathbf{b}(\mathbf{r})\nabla\theta(\mathbf{r}))\end{bmatrix} + 
\begin{bmatrix}0\\P(\mathbf{r})\end{bmatrix}
```

After multiplying by a test function and integrating this equation can be written as

```math
\begin{bmatrix}\mathbf{M}_1 & 0 \\ 0 & \mathbf{M}_2\end{bmatrix} \frac{d}{dt} \begin{bmatrix}\hat{\theta}\\\hat{\omega}\end{bmatrix} =
\begin{bmatrix}0 & \mathbf{K}_1\\\mathbf{K}_2 & \mathbf{K}_3 \end{bmatrix}\begin{bmatrix}\hat{\theta}\\\hat{\omega}\end{bmatrix} + \begin{bmatrix}0\\ \mathbf{f}\end{bmatrix},
```

where

```math
\mathbf{M}_{1,ij} = \int\phi_i(\mathbf{r}) \phi_j(\mathbf{r}) \, d\Omega, \qquad\mathbf{M}_{2,ij} = \int m(\mathbf{r})\phi_i(\mathbf{r}) \phi_j(\mathbf{r}) \, d\Omega,
```

```math
\begin{align*}
\mathbf{K}_{1,ij} &= \int\phi_i(\mathbf{r}) \phi_j(\mathbf{r}) \, d\Omega, \qquad\mathbf{K}_{2,ij} = -\int d(\mathbf{r})\phi_i(\mathbf{r}) \phi_j(\mathbf{r}) \, d\Omega,\\
\mathbf{K}_{3,ij} &= -\int\nabla\phi_i(\mathbf{r})\begin{bmatrix}b_x(\mathbf{r}) & 0 \\ 0 & b_y(\mathbf{r}) \end{bmatrix} \nabla\phi_j(\mathbf{r})\, d\Omega,
\end{align*}
```

and

```math
\mathbf{f} = \int P(\mathbf{r})\phi_i(\mathbf{r})\, d\Omega.
```

The integrals can then be solved in the way described in the previous section.
