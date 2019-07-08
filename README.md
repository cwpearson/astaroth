![astaroth_logo](./doc/astaroth_logo.svg "Astaroth Sigil")

# Astaroth - A Multi-GPU library for generic stencil computations

Astaroth is a single-node multi-GPU library for multiphysics and other problems, which involve stencil computations in a discrete mesh. It's licenced under the terms of the GNU General Public Licence, version 3, or later (see [LICENCE.txt](https://bitbucket.org/miikkavaisala/astaroth-code/src/master/astaroth_2.0/LICENCE.txt)). Astaroth ships with a domain-specific language that can be used to translate high-level representations of various stencil operations into efficient CUDA kernels.

## System requirements

NVIDIA GPU with >= 3.0 compute capability. See https://en.wikipedia.org/wiki/CUDA#GPUs_supported.

## Building (3rd party libraries for real-time visualization)

1. `cd 3rdparty`
1. `./setup_dependencies.sh` Note: this may take some time.

## Building 

There are two ways to build the code as instructed below. 

If you encounter issues, recheck that the 3rd party libraries were successfully built during the previous step.

### Method I: In the code directory

1. `cd build/`
1. `cmake -DDOUBLE_PRECISION=OFF -DBUILD_DEBUG=OFF ..` (Use `cmake -D CMAKE_C_COMPILER=icc -D CMAKE_CXX_COMPILER=icpc -DDOUBLE_PRECISION=OFF -DBUILD_DEBUG=OFF ..` if compiling on TIARA)
1. `../scripts/compile_acc.sh && make -j`
1. `./ac_run <options>`

Edit `config/astaroth.conf` to change the numerical setup. 

### Method II: With a script in a custom build directory (RECOMMENDED) 

1. `source sourceme.sh` to add relevant directories to the `PATH`
1. `ac_mkbuilddir.sh -b my_build_dir/` to set up a custom build directory. There are also other options available. See `ac_mkbuilddir.sh -h` for more. 
1. `compile_acc.sh` to generate kernels from the Domain Specific Language 
1. `cd my_build_dir/` 
1. `make -j`
1. `./ac_run <options>`

Edit `my_build_dir/astaroth.conf` to change the numerical setup. 

### Available options

- `-s` simulation
- `-b` benchmark
- `-t` automated test 

By default, the program does a real-time visualization of the simulation domain. The camera and the initial conditions can be controller by `arrow keys`, `pgup`, `pgdown` and `spacebar`.

## Visualization 

See `analysis/python/` directory of existing data visualization and analysis scripts.  

## Generating documentation

Run `doxygen doxyfile` in astaroth_2.0 directory. The generated files can be found in `doc/doxygen`. The main page of the documentation will be at `dox/doxygen/astaroth_doc_html/index.html`.

## Formatting

If you have clang-format, you may run `scripts/fix_style.sh`. This script will recursively fix style of all the source files down from the current working directory. The script will ask for a confirmation before making any changes. 

## Directory structure
TODO

## Contributing

0. **Do not break existing functionality.** Do not modify the interface functions declared in astaroth.h and device.cuh in any way. Bug fixes are exceptions. If you need new functionality, create a new function.

0. **Do not rename or redefine variables or constants declared in astaroth.h** without consulting everyone involved with the project.

0. **Ensure that the code compiles and the automated tests pass** by running `./ac_run -t` before pushing changes to master. If you want to implement a feature that consists of multiple commits, see Managing feature branches below.

### Managing feature branches 

0. Ensure that you're on the latest version of master. `git checkout master && git pull`

0. Create a feature branch with `git checkout -b <feature_name_year-month-date>`, f.ex. `git checkout -b forcingtests_2019-01-01`

0. Do your commits in that branch until your new feature works

0. Merge master with your feature branch `git merge master`

0. Resolve the conflicts and test that the code compiles and still works by running `./ac_run -t`

0. If everything is OK, commit your final changes to the feature branch and merge it to master `git commit && git checkout master && git merge <your feature branch> && git push`

0. Unless you really have to keep your feature branch around for historical/other reasons, remove it from remote by calling `git push origin --delete <your feature branch>`

A flowchart is available at [doc/commitflowchart.png](https://bitbucket.org/jpekkila/astaroth/src/2d91df19dcb3/doc/commitflowchart.png?at=master).

### About branches in general

* Unused branches should not kept around after merging them into master in order to avoid cluttering the repository. 

* `git branch -a --merged` shows a list of branches that have been merged to master and are likely not needed any more.

* `git push origin --delete <feature branch>` deletes a remote branch while `git branch -d <feature branch>` deletes a local branch

* If you think that you have messed up and lost work, run `git reflog` which lists the latests commits. All work that has been committed should be accessible with the hashes listed by this command with `git checkout <reflog hash>`.

## Coding style.

### In a nutshell
- Use [K&R indentation style](https://en.wikipedia.org/wiki/Indentation_style#K&R_style) and 4 space tabs. 
- Line width is 100 characters
- Start function names after a linebreak in source files. 
- [Be generous with `const` type qualifiers](https://isocpp.org/wiki/faq/const-correctness). 
- When in doubt, see [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).

### Header example:
```cpp
// Licence notice and doxygen description here
#pragma once
#include "avoid_including_headers_here.h"

/** Doxygen comments */
void globalFunction(void);
```


### Source example:
```cpp
#include "parent_header.h"

#include <standard_library_headers.h>

#include "other_headers.h"
#include "more_headers.h"

typedef struct {
	int data;
} SomeStruct;

static inline int small_function(const SomeStruct& stuff) { return stuff.data; }

// Pass constant structs always by reference (&) and use const type qualifier.
// Modified structs are always passed as pointers (*), never as references.
// Constant parameters should be on the left-hand side, while non-consts go to the right.
static void
local_function(const SomeStruct& constant_struct, SomeStruct* modified_struct)
{
	modified_struct->data = constant_struct.data;
}

void
globalFunction(void)
{
	return;
}
```
## TIARA cluster compilation notes

Modules used when compiling the code on TIARA cluster. 

  * cmake/3.9.5
  * gcc/8.3.0
  * cuda/10.1


