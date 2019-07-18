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

This part of the process is most complicated, and will require special
attention.  The starting point should be the Truelove criterion, but we migh
need additional subgrid model assumptions. 

Jeans length
\begin{equation}
\lambda_\mathrm{J} = \Big( \frac{\pi c_s^2}{G \rho} \Big)^{1/2}
\end{equation} 

Truelove-Jeans density.
\begin{equation}
\rho_\mathrm{TJ} = \frac{\pi J^2 c_s^2}{G \Delta x^2}
\end{equation} 
where $J = \Delta x / \lambda_\mathrm{J}$. Lee et al. (2014) set $J=1/8$.

Magnetic Truelove criterion
\begin{equation}
\rho_\mathrm{TJ, mag} = \rho_\mathrm{TJ} (1 + 0.74/\beta)
\end{equation}

Accreted mass 
\begin{equation}
 (\rho - \rho_\mathrm{TJ, mag})\Delta x^3
\end{equation}

## Stage 5: Data gathering (device to host)

After accretion we will need to gather all data to `sink.mass`. This might need gathering from multiple GPUs, i.e. like in 

Depends on the stencils size and shape used for at the accretion stage. Currently the method is not clear. 

## Plan 

### 1. Add gravitating particle

Add a particle with specific mass and location. No accretion included. 

### 2. Add simple accretion

Add an accretion property of the particle. Use just a basic form. 

### 3. Add Complicated accretion

Add accretion properties which might be useful for the sake of physical correctness and/or numerical stability. 

### 4. Add particle movement. (OPTIONAL)

Make is possible for the particle to accrete momentum in addition to mass, and therefore influence its movement. 

### 5. Multiple particles. (VERY OPTIONAL)

Create sink particles dynamically and allow the presence of multiple sinks. 


