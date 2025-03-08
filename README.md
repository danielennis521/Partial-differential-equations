Welcome! I keep things here divided up by equation, each folder will have code for one or more variants of the 
equation it's named after. In addition to functions for solving agiven equation there will usually be some classes 
that make it easy to run and view multiple simulations at the same time.


# Nonlinear-Heat
below is an animation of the nonlinear heat transfer variant: $$\frac{\partial u}{\partial t} = \frac{\partial}{\partial x}[(\alpha/(\alpha + u^2)) * \frac{\partial u}{\partial x}]$$
with $$\alpha$$ ranging from 0.5 to 3. As $$\alpha$$ increases this equations becomes more like normal heat transfer (blue line below)
while for smaller values of a the initial evolution of the solution differs greatly from the normal behavior.

![Alt Text](https://github.com/danielennis521/Partial-differential-equations/blob/main/nonlinear-heat/gifs/quadratic_limit_behavior.gif)

While the examples here are fairly exagerated most real world examples of heat transfer have some degree of nonlinearity since the thermal conductivity of most materials tends to go down as temperature increases. This makes the demo above somewhat unrealistic since $$(\alpha/(\alpha + u^2))$$ will decrease at higher and lower (far below 0) temperatures. I chose this function for the 
thermal conductivity as an example though because I really like how it looks :)
