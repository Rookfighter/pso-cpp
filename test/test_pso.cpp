/*
 * test_pso.cpp
 *
 *  Created on: 08 Jan 2019
 *      Author: Fabian Meyer
 */

#ifdef _MSC_VER
#include "stdafx.h"
#endif

#include "eigen_assert.h"
#include <psocpp.h>

template<typename Scalar>
struct Paraboloid
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef typename Matrix::Index Index;

    Scalar operator()(const Matrix &particles, const Index col) const
    {
        return particles.col(col).squaredNorm();
    }
};

TEST_CASE("Particle Swarm Optimization")
{
    pso::Optimizer<double, Paraboloid<double> > opt;
    opt.setMaxIterations(100);

    SECTION("with paraboloid")
    {
        Eigen::MatrixXd bounds(2, 3);
        bounds << -5, -5, -5, 5, 5, 5;

        opt.setEpsilonX(1e-8);
        opt.setEpsilonF(1e-10);

        auto result = opt.minimize(bounds, 200);
        REQUIRE(result.converged);
        REQUIRE(Approx(0.0).margin(1e-3) == result.fval);
        REQUIRE_MAT(Eigen::VectorXd::Zero(3), result.xval, 1e-3);
    }

    SECTION("with wrong bounds")
    {
        Eigen::MatrixXd bounds(3, 3);
        bounds << -5, -5, -5,
            5, 5, 5,
            6, 6, 6;

        REQUIRE_THROWS(opt.minimize(bounds, 200));

        bounds.resize(2, 3);
        bounds << -5, -5, 6,
            5, 5, 5;

        REQUIRE_THROWS(opt.minimize(bounds, 200));
    }
}
