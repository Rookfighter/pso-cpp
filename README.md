# pso-cpp

![Cpp11](https://img.shields.io/badge/C%2B%2B-11-blue.svg)
![License](https://img.shields.io/packagist/l/doctrine/orm.svg)
![Travis Status](https://travis-ci.org/Rookfighter/pso-cpp.svg?branch=master)
![Appveyer Status](https://ci.appveyor.com/api/projects/status/cl5iljq9bq6lcusu?svg=true)

pso-cpp is a header-only C++ library for particle swarm optimization using
the Eigen3 linear algebra library.

## Install

Simply copy the header file into your project or install it using
the CMake build system by typing

```bash
cd path/to/repo
mkdir build
cd build
cmake ..
make install
```

The library requires Eigen3 to be installed on your system.
In Debian based systems you can simply type

```bash
apt-get install libeigen3-dev
```

Make sure Eigen3 can be found by your build system.

## Usage

There are three steps to use gradient-descent-cpp:

* Implement your objective function as functor
* Instantiate the particle swarm optimizer
* Choose your parameters

```cpp
#include <psocpp.h>

// Implement an objective functor.
struct Paraboloid
{
    template<typename Derived>
    double operator()(const Eigen::MatrixBase<Derived> &xval) const
    {
        return xval(0) * xval(0) + xval(1) * xval(1);
    }
};

int main()
{
    // Create optimizer object with Paraboloid functor as objective.
    // You can additionally specify an InertiaWeightStrategy functor as template
    // parameter.
    // You can additionally specify a Callback functor as template parameter.
    pso::Optimizer<double, Paraboloid> optimizer;

    // Set number of iterations as stop criterion.
    // Set it to 0 or negative for infinite iterations (default is 0).
    optimizer.setMaxIterations(100);

    // Set the minimum change of the x-values (particles) (default is 1e-6).
    // If the change is lower then the optimizer stops minimizing.
    optimizer.setEpsilonX(1e-3);

    // Set the minimum change of the function values (default is 1e-6).
    // If the change is lower then the optimizer stops minimizing.
    optimizer.setEpsilonF(1e-3);

    // Set the number of threads used for evaluation (OpenMP only).
    // Set it to 0 or negative for auto detection (default is 1).
    optimizer.setThreads(2);

    // Turn verbosity on, so the optimizer prints status updates after each
    // iteration.
    optimizer.setVerbose(true);

    // Set the bounds in which the optimizer should search.
    // Each column vector defines the (min, max) for each dimension  of the
    // particles.
    Eigen::MatrixXd bounds(2, 2);
    bounds << -4, -4,
               4,  4;

    // start the optimization with a particle count
    auto result = optimizer.minimize(bounds, 100);

    std::cout << "Done! Converged: " << (result.converged ? "true" : "false")
        << " Iterations: " << result.iterations << std::endl;

    // do something with final function value
    std::cout << "Final fval: " << result.fval << std::endl;

    // do something with final x-value
    std::cout << "Final xval: " << result.xval.transpose() << std::endl;

    return 0;
}
```
