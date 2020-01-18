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

> **Optional:** Documentation can be generated by running `doxygen` in the base directory. The
generated documentation can be found in `doc/doxygen`.

> **Tip:**  The library is configured by passing [options](#markdown-header-cmake-options) to CMake with `-D[option]=[ON|OFF]`.
For example, double precision can be enabled by calling `cmake -DBUILD_DOUBLE_PRECISION=ON ..`.
See [CMakeLists.txt](https://bitbucket.org/jpekkila/astaroth/src/master/CMakeLists.txt) for an up-to-date list of options.

> **Note:** CMake will inform you if there are missing dependencies.

## CMake Options

| Option | Description | Default |
|--------|-------------|---------|
| BUILD_DEBUG | Builds Astaroth with extensive error checking | OFF |
| BUILD_STANDALONE | Builds a standalone library for testing, benchmarking and simulation | ON |
| BUILD_UTILS | Builds a generic utility library (WIP replacement for BUILD_STANDALONE) | ON |
| BUILD_RT_VISUALIZATION | Builds the real-time visualization module | OFF |
| BUILD_SAMPLES | Builds projects in samples subdirectory | OFF |
| DOUBLE_PRECISION | Generates double precision code | OFF |
| MULTIGPU_ENABLED | Enables Astaroth to use multiple GPUs on a single node | ON |
| MPI_ENABLED | Enables additional functions for MPI communciation | OFF |
| DSL_MODULE_DIR | Defines the directory to be scanned when looking for DSL files | `astaroth/acc/mhd_solver` |


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

* `astaroth/include/astaroth.h`: Legacy interface for backwards compatibility and quick testing.
* `astaroth/include/astaroth_node.h`: Multi-GPU interface (single node).
* `astaroth/include/astaroth_device.h`: Single-GPU interface.
* `astaroth/src/utils/`: Utility library for host-side memory allocations, verification and other tasks.

## FAQ

Can I use the code even if I don't make my changes public?

> [GPL](LICENCE.md) requires only that if you release a binary based on Astaroth to the public, then you should also release the source code for it. In private you can do whatever you want (secret forks, secret collaborations, etc). **Astaroth Code source files (.ac, .h) do not belong to the library and therefore are not licenced under GPL.** The user who created the files holds copyright over them and can choose to distribute them under any licence.

How do I compile with MPI support?

> MPI implementation for Astaroth is still work in progress, these commands are for testing only. Invoke CMake with `cmake -DMPI_ENABLED=ON -DBUILD_MPI_TEST=ON -DCMAKE_CXX_COMPILER=$(which mpicxx) ..`. Otherwise the build steps are the same. Run with `mpirun -np 4 ./mpitest`.

How do I contribute?

> See [Contributing](CONTRIBUTING.md).
