/*
 * test_pso.cpp
 *
 *  Created on: 08 Jan 2019
 *      Author: Fabian Meyer
 */

#include "eigen_assert.h"
#include <psocpp.h>

template<typename Scalar>
struct Paraboloid
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

    Vector offset;

    Paraboloid()
        : offset()
    { }

    Paraboloid(const Vector &offset)
        : offset(offset)
    { }

    template<typename Derived>
    Scalar operator()(const Eigen::MatrixBase<Derived> &state) const
    {
        assert(offset.size() == state.size());
        return (state + offset).squaredNorm();
    }
};

typedef double Scalar;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

TEST_CASE("Particle Swarm Optimization")
{
    pso::Optimizer<Scalar, Paraboloid<Scalar> > opt;
    Paraboloid<Scalar> parab;
    parab.offset.setZero(3);
    opt.setObjective(parab);

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

    SECTION("in negative bounds")
    {
        parab.offset << 2, 2, 2;
        opt.setObjective(parab);

        Eigen::MatrixXd bounds(2, 3);
        bounds <<
            -1, -1, -1,
             1,  1,  1;

        auto result = opt.minimize(bounds, 200);

        Vector stateExp(3);
        stateExp << -1, -1, -1;

        REQUIRE_MAT(result.xval, stateExp, 1e-6);
    }

    SECTION("in positive bounds")
    {
        parab.offset << -2, -2, -2;
        opt.setObjective(parab);

        Eigen::MatrixXd bounds(2, 3);
        bounds <<
            -1, -1, -1,
             1,  1,  1;

        auto result = opt.minimize(bounds, 200);

        Vector stateExp(3);
        stateExp << 1, 1, 1;

        REQUIRE_MAT(result.xval, stateExp, 1e-6);
    }

    SECTION("in mixed bounds")
    {
        parab.offset << 2, -2, 2;
        opt.setObjective(parab);

        Eigen::MatrixXd bounds(2, 3);
        bounds <<
            -1, -1, -1,
             1,  1,  1;

        auto result = opt.minimize(bounds, 200);

        Vector stateExp(3);
        stateExp << -1, 1, -1;

        REQUIRE_MAT(result.xval, stateExp, 1e-6);
    }
}
