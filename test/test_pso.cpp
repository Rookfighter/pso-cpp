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
    SECTION("optimize paraboloid")
    {
        Paraboloid<Scalar> parab;
        parab.offset.setZero(3);

        SECTION("with constant weight")
        {
            Eigen::MatrixXd bounds(2, 3);
            bounds << -5, -5, -5, 5, 5, 5;

            pso::Optimizer<Scalar, Paraboloid<Scalar>, pso::ConstantWeight<Scalar>> opt;
            opt.setObjective(parab);
            opt.setInertiaWeightStrategy(pso::ConstantWeight<Scalar>(0.45));
            opt.setPhiParticles(0.9);
            opt.setPhiGlobal(0.9);
            opt.setMaxIterations(100);

            auto result = opt.minimize(bounds, 200);
            REQUIRE(result.converged);
            REQUIRE(Approx(0.0).margin(1e-3) == result.fval);
            REQUIRE_MAT(Eigen::VectorXd::Zero(3), result.xval, 1e-3);
        }

        SECTION("with natural exponent weight 1")
        {
            Eigen::MatrixXd bounds(2, 3);
            bounds << -5, -5, -5, 5, 5, 5;

            pso::Optimizer<Scalar, Paraboloid<Scalar>, pso::ExponentialDecrease1<Scalar>> opt;
            opt.setObjective(parab);
            opt.setInertiaWeightStrategy(pso::ExponentialDecrease1<Scalar>(0.4, 0.9));
            opt.setPhiParticles(2.0);
            opt.setPhiGlobal(2.0);
            opt.setMaxIterations(100);

            auto result = opt.minimize(bounds, 200);
            REQUIRE(result.converged);
            REQUIRE(Approx(0.0).margin(1e-3) == result.fval);
            REQUIRE_MAT(Eigen::VectorXd::Zero(3), result.xval, 1e-3);
        }

        SECTION("with natural exponent weight 2")
        {
            Eigen::MatrixXd bounds(2, 3);
            bounds << -5, -5, -5, 5, 5, 5;

            pso::Optimizer<Scalar, Paraboloid<Scalar>, pso::ExponentialDecrease2<Scalar>> opt;
            opt.setObjective(parab);
            opt.setInertiaWeightStrategy(pso::ExponentialDecrease2<Scalar>(0.4, 0.9));
            opt.setPhiParticles(2.0);
            opt.setPhiGlobal(2.0);
            opt.setMaxIterations(100);

            auto result = opt.minimize(bounds, 200);
            REQUIRE(result.converged);
            REQUIRE(Approx(0.0).margin(1e-3) == result.fval);
            REQUIRE_MAT(Eigen::VectorXd::Zero(3), result.xval, 1e-3);
        }

        SECTION("with wrong bounds")
        {
            pso::Optimizer<Scalar, Paraboloid<Scalar>> opt;
            opt.setObjective(parab);
            opt.setInertiaWeightStrategy(pso::ConstantWeight<Scalar>(0.45));
            opt.setPhiParticles(0.9);
            opt.setPhiGlobal(0.9);
            opt.setMaxIterations(100);

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
            pso::Optimizer<Scalar, Paraboloid<Scalar>> opt;
            opt.setObjective(parab);
            opt.setInertiaWeightStrategy(pso::ConstantWeight<Scalar>(0.45));
            opt.setPhiParticles(0.9);
            opt.setPhiGlobal(0.9);
            opt.setMaxIterations(100);

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
            pso::Optimizer<Scalar, Paraboloid<Scalar>> opt;
            opt.setObjective(parab);
            opt.setInertiaWeightStrategy(pso::ConstantWeight<Scalar>(0.45));
            opt.setPhiParticles(0.9);
            opt.setPhiGlobal(0.9);
            opt.setMaxIterations(100);

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
            pso::Optimizer<Scalar, Paraboloid<Scalar>> opt;
            opt.setObjective(parab);
            opt.setInertiaWeightStrategy(pso::ConstantWeight<Scalar>(0.45));
            opt.setPhiParticles(0.9);
            opt.setPhiGlobal(0.9);
            opt.setMaxIterations(100);

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
}
