/*
 * test_pso.cpp
 *
 *  Created on: 08 Jan 2019
 *      Author: Fabian Meyer
 */

#include "eigen_assert.h"
#include <psocpp.h>

TEST_CASE("Particle Swarm Optimization")
{
    auto paraboloid = [](const Eigen::VectorXd &state){return state.squaredNorm();};

    pso::Optimizer<double> opt;
    opt.setMaxIterations(100);

    SECTION("with paraboloid")
    {
        Eigen::MatrixXd bounds(2, 3);
        bounds << -5, -5, -5, 5, 5, 5;

        opt.setEpsilonX(1e-8);
        opt.setEpsilonF(1e-10);

        auto result = opt.minimize(paraboloid, bounds, 200);
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

        REQUIRE_THROWS(opt.minimize(paraboloid, bounds, 200));

        bounds.resize(2, 3);
        bounds << -5, -5, 6,
            5, 5, 5;

        REQUIRE_THROWS(opt.minimize(paraboloid, bounds, 200));
    }
}
