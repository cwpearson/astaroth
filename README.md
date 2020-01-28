# Astaroth - A Multi-GPU Library for Generic Stencil Computations {#mainpage}

[Specification](doc/Astaroth_API_specification_and_user_manual/API_specification_and_user_manual.md) | [Contributing](CONTRIBUTING.md) | [Licence](LICENCE.md) | [Repository](https://bitbucket.org/jpekkila/astaroth) | [Issue Tracker](https://bitbucket.org/jpekkila/astaroth/issues?status=new&status=open) | [Wiki](https://bitbucket.org/jpekkila/astaroth/wiki/Home)

Astaroth is a multi-GPU library for three-dimensional stencil computations. It is designed especially for performing high-order stencil
computations in structured grids, where several coupled fields are updated each time step. Astaroth consists of a multi-GPU and single-GPU
APIs and provides a domain-specific language for translating high-level descriptions of stencil computations into efficient GPU code. This
makes Astaroth especially suitable for multiphysics simulations.

Astaroth is licenced under the terms of the GNU General Public Licence, version 3, or later
(see [LICENCE.txt](LICENCE.md)). For contributing guidelines,
see [Contributing](CONTRIBUTING.md).


## System Requirements
* An NVIDIA GPU with support for compute capability 3.0 or higher (Kepler architecture or newer)

## Dependencies
Relative recent versions of

`gcc cmake cuda flex bison`.

## Building

In the base directory, run

1. `mkdir build`
2. `cd build`
3. `cmake ..`
4. `make -j`

> **Optional:** Documentation can be generated by running `doxygen` in the base directory. Generated documentation can be found in `doc/doxygen`.

> **Tip:**  The library is configured by passing [options](#markdown-header-cmake-options) to CMake with `-D[option]=[ON|OFF]`. For example, double precision can be enabled by calling `cmake -DBUILD_DOUBLE_PRECISION=ON ..`. See [CMakeLists.txt](https://bitbucket.org/jpekkila/astaroth/src/master/CMakeLists.txt) for an up-to-date list of options.

> **Note:** CMake will inform you if there are missing dependencies.

## CMake Options

| Option | Description | Default |
|--------|-------------|---------|
| CMAKE_BUILD_TYPE | Selects the build type. Possible values: Debug, Release, RelWithDebInfo, MinSizeRel. See (CMake documentation)[https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html] for more details. | Release |
| DOUBLE_PRECISION | Generates double precision code. | OFF |
| BUILD_SAMPLES | Builds projects in samples subdirectory. | OFF |
| BUILD_STANDALONE | Builds a standalone library for testing, benchmarking and simulation. | ON |
| MPI_ENABLED | Enables multi-GPU on a single node. Uses peer-to-peer communication instead of MPI. Affects Legacy & Node layers only. | OFF |
| MULTIGPU_ENABLED | Enables Astaroth to use multiple GPUs on a single node. | ON |
| DSL_MODULE_DIR | Defines the directory to be scanned when looking for DSL files. | `astaroth/acc/mhd_solver` |


## Standalone Module


```Bash
Usage: ./ac_run [options]
	     --help | -h: Prints this help.
	     --test | -t: Runs autotests.
	--benchmark | -b: Runs benchmarks.
	 --simulate | -s: Runs the simulation.
	   --render | -r: Runs the real-time renderer.
	   --config | -c: Uses the config file given after this flag instead of the default.
```

See `analysis/python/` directory of existing data visualization and analysis scripts.

## Interface

* `astaroth/include/astaroth.h`: Astaroth main header. Contains the interface for accessing single- and multi-GPU layers.
* `astaroth/include/astaroth_utils.h`: Utility library header. Provides functions for performing common tasks on host, such as allocating and verifying meshes.

## FAQ

Can I use the code even if I don't make my changes public?

> [GPL](LICENCE.md) requires only that if you release a binary based on Astaroth to the public, then you should also release the source code for it. In private you can do whatever you want (secret forks, secret collaborations, etc). **Astaroth Code source files (.ac, .h) do not belong to the library and therefore are not licenced under GPL.** The user who created the files holds copyright over them and can choose to distribute them under any licence.

How do I compile with MPI support?

> MPI implementation for Astaroth is still work in progress, these commands are for testing only. Invoke CMake with `cmake -DMPI_ENABLED=ON -DBUILD_SAMPLES=ON ..`. Otherwise the build steps are the same. Run with `mpirun -np 4 ./mpitest`.

How do I contribute?

> See [Contributing](CONTRIBUTING.md).
