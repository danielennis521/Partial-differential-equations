Welcome! I keep things here divided up by equation, each folder will have code for one or more variants of the 
equation it's named after. In addition to functions for solving agiven equation there will usually be some classes 
that make it easy to run and view multiple simulations at the same time.


# Nonlinear-Heat
below is an animation of the nonlinear heat transfer variant: $$\frac{\partial u}{\partial t} = \frac{\partial}{\partial x}[\frac{\alpha}{\alpha + u^2} * \frac{\partial u}{\partial x}]$$
with $$\alpha$$ ranging from 0.5 to 3. As $$\alpha$$ increases this equations becomes more like the standard heat equation (blue line below)
while for smaller values of a the initial evolution of the solution differs greatly from the normal behavior.

![Alt Text](https://github.com/danielennis521/Partial-differential-equations/blob/main/nonlinear-heat/gifs/quadratic_limit_behavior.gif)

While the examples here are fairly exagerated most real world examples of heat transfer have some degree of nonlinearity since the thermal conductivity of most materials tends to go down as temperature increases. This makes the demo above somewhat unrealistic since $$(\alpha/(\alpha + u^2))$$ will decrease at higher and lower (far below 0) temperatures. I chose this function for the 
thermal conductivity as an example though because I really like how it looks :)

The demo below shows a more realistic scenario when the diffusivity is lower when the temperature is higher. Notice how the left hand side moves faster at first:

<img src="https://github.com/danielennis521/Partial-differential-equations/blob/main/nonlinear-heat/gifs/realistic_nonlinear_heat.gif" alt="Alt Text" width="450" height="325">

The weird bump on the right side that appears for a while. That's an artifact of the way this simulation was made, the spatial discretization was done using a "Spectral" method. These methods find linear combinations of global basis functions, usually fourier modes or orthogonal polynomials, that "nearly solve" the differential equation. Spectral methods can be super useful when you need higher accuracy or are working with real world data that is relatively sparse. You can also learn a lot about a given PDE from the effect it has on the spectral coefficients of some inital data.


# Burgers Equation
Burgers equation is a prototypical example when it comes to shock, that is, when a discontinuity appears in the solution of a PDE. 

![Alt Text](https://github.com/danielennis521/Partial-differential-equations/blob/main/Burgers-Equation/gifs/burger_eq_near_shock.gif)
