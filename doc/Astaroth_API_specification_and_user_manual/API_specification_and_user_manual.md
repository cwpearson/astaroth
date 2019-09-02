# Astaroth specification and user manual

Copyright (C) 2014-2019, Johannes Pekkila, Miikka Vaisala.

	   Astaroth is free software: you can redistribute it and/or modify
	   it under the terms of the GNU General Public License as published by
	   the Free Software Foundation, either version 3 of the License, or
	   (at your option) any later version.
	   Astaroth is free software: you can redistribute it and/or modify
	   it under the terms of the GNU General Public License as published by
	   the Free Software Foundation, either version 3 of the License, or
	   (at your option) any later version.

	   Astaroth is distributed in the hope that it will be useful,
	   but WITHOUT ANY WARRANTY; without even the implied warranty of
	   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	   GNU General Public License for more details.

	   You should have received a copy of the GNU General Public License
	   along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.


# Introduction and background

Astaroth is a collection of tools for utilizing multiple graphics processing units (GPUs) efficiently in three-dimensional stencil computations. This document specifies the Astaroth application-programming interface (API) and domain-specific language (DSL).

Astaroth has been designed for the demands in computational sciences, where large stencils are often used to attain sufficient accuracy. The majority of previous work focuses on stencil computations with low-order stencils for which several efficient algorithms have been proposed, whereas work on high-order stencils is more limited. In addition, in computational physics multiple fields interact with each other, such as the velocity and magnetic fields of electrically conducting fluids. Such computations are especially challenging to solve efficiently because of the problem's relatively low operational intensity and the small caches provided by GPUs. Efficient methods for computations with several coupled fields and large stencils have not been addressed sufficiently in prior work.

With Astaroth, we have taken inspiration of image processing and graphics pipelines which rely on holding intermediate data in caches for the duration of computations, and extended the idea to work efficiently also with large three-dimensional stencils and an arbitrary number of coupled fields. As programming GPUs efficiently is relatively verbose and requires deep knowledge of the underlying hardware and execution model, we have created a high-level domain-specific language for expressing a wide range of tasks in computational sciences and provide a source-to-source compiler for translating stencil problems expressed in our language into efficient CUDA kernels.

The kernels generated from the Astaroth DSL are embedded in the Astaroth Core library, which is usable via the Astaroth API. While the Astaroth library is written in C++/CUDA, the API conforms to the C99 standard.


# Publications

The foundational work was done in (Väisälä, Pekkilä, 2017) and the library, API and DSL described in this document were introduced in (Pekkilä, 2019). We kindly wish the users of Astaroth to cite to these publications in their work.

> J. Pekkilä, Astaroth: A Library for Stencil Computations on Graphics Processing Units. Master's thesis, Aalto University School of Science, Espoo, Finland, 2019.

> M. S. Väisälä, Magnetic Phenomena of the Interstellar Medium in Theory and Observation. PhD thesis, University of Helsinki, Finland, 2017.

> J. Pekkilä, M. S. Väisälä, M. Käpylä, P. J. Käpylä, and O. Anjum, “Methods for compressible fluid simulation on GPUs using high-order finite differences, ”Computer Physics Communications, vol. 217, pp. 11–22, Aug. 2017.



# Astaroth API

## Devices and nodes

## Meshes

## Streams and synchronization

## Interface


# Astaroth DSL

## Uniforms

## Vertex buffers

### Input and output buffers

## Built-in variables and functions


