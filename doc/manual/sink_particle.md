# Sink particle

The document aims to describe how the sink particle is calculated. 
For referfence see Lee et al. (2014) Apj, 783, 50.

## Stage 1: Initialization

Create datatype `sink` with coordinate and location for the particle. 

```c
sink.x   
sink.y 
sink.z
sink.mass 
```

Either we start with an initial particle, or we can assume it to form later
(e.g. based on Truelove criterion.)

## Stage 2: Communication (host to device)

Values of `sink` need to be communicated from Host to Device. 

## Stage 3: Gravitation

Based on `sink` gravitational force effect is added to the momentun equation (via DSL).

## Stage 4: Accretion

Jeans length
\begin{equation}
\lambda_\mathrm{J} = \Big( \frac{\pi c_s^2}{G \rho} \Big)^{1/2}
\end{equation} 

Truelove-Jeans density.
\begin{equation}
\rho_\mathrm{TJ} = \frac{\pi J^2 c_s^2}{G \Delta x^2}
\end{equation} 
where $J = \Delta x / \lambda_\mathrm{J}$

## Stage 5: Data gathering (device to host)

We need to combine accretion calculated in multiple GPUs.  
