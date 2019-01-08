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
