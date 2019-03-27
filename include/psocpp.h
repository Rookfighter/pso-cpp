/** psocpp.h
  *
  *     Author: Fabian Meyer
  * Created on: 08 Jan 2019
  *    License: MIT
  */

#ifndef PSOCPP_PSOCPP_H_
#define PSOCPP_PSOCPP_H_

#include <Eigen/Geometry>
#include <stdexcept>
#include <limits>
#include <functional>
#include <iostream>
#include <ctime>
#include <iomanip>
#include <random>

namespace pso
{
    template<typename Scalar>
    struct NoCallback
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef typename Matrix::Index Index;

        void operator()(const size_t, const Matrix&, const Vector &, const Index) const
        {

        }
    };

    template<typename Scalar>
    struct ConstantWeight
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef typename Matrix::Index Index;

        Scalar weight;

        ConstantWeight()
            : ConstantWeight(1.0)
        { }

        ConstantWeight(const Scalar weight)
            : weight(weight)
        { }

        Scalar operator()(const size_t,
            const size_t) const
        {
            return weight;
        }
    };

    template<typename Scalar>
    struct NaturalExponentWeight1
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef typename Matrix::Index Index;

        Scalar weightMin;
        Scalar weightMax;

        NaturalExponentWeight1()
            : NaturalExponentWeight1(0.4, 0.9)
        { }

        NaturalExponentWeight1(const Scalar weightMin, const Scalar weightMax)
            : weightMin(weightMin), weightMax(weightMax)
        { }

        Scalar operator()(const size_t iteration,
            const size_t maxIt) const
        {
            Scalar exponent = static_cast<Scalar>(iteration) / (static_cast<Scalar>(maxIt) / 10.0);
            return weightMin + (weightMax - weightMin) * std::exp(-exponent);
        }
    };

    template<typename Scalar>
    struct NaturalExponentWeight2
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef typename Matrix::Index Index;

        Scalar weightMin;
        Scalar weightMax;

        NaturalExponentWeight2()
            : NaturalExponentWeight2(0.4, 0.9)
        { }

        NaturalExponentWeight2(const Scalar weightMin, const Scalar weightMax)
            : weightMin(weightMin), weightMax(weightMax)
        { }

        Scalar operator()(const size_t iteration,
            const size_t maxIt) const
        {
            Scalar exponent = static_cast<Scalar>(iteration) / (static_cast<Scalar>(maxIt) / 4.0);
            exponent *= exponent;
            return weightMin + (weightMax - weightMin) * std::exp(-exponent);
        }
    };

    template<typename Scalar,
        typename Objective,
        typename InertiaWeightStrategy = ConstantWeight<Scalar>,
        typename Callback = NoCallback<Scalar> >
    class Optimizer
    {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef typename Matrix::Index Index;

        struct Result
        {
            size_t iterations;
            bool converged;
            Scalar fval;
            Vector xval;
        };

    private:
        Objective objective_;
        Callback callback_;
        InertiaWeightStrategy weightStrategy_;

        size_t threads_;

        size_t maxit_;
        Scalar xeps_;
        Scalar feps_;

        Scalar phip_;
        Scalar phig_;
        Scalar maxVel_;

        bool verbose_;

        std::function<Scalar()> dice_;

        void randomizeParticles(const Matrix &bounds, Matrix &particles)
        {
            for(Index i = 0; i < particles.cols(); ++i)
            {
                for(Index j = 0; j < particles.rows(); ++j)
                {
                    Scalar minval = bounds(0, j);
                    Scalar maxval = bounds(1, j);
                    Scalar diff = maxval - minval;
                    particles(j, i) = minval + (dice_() * diff);
                }
            }
        }

        void randomizeVelocities(const Matrix &bounds, Matrix &velocities)
        {
            for(Index i = 0; i < velocities.cols(); ++i)
            {
                for(Index j = 0; j < velocities.rows(); ++j)
                {
                    Scalar minval = bounds(0, j);
                    Scalar maxval = bounds(1, j);
                    Scalar diff = maxval - minval;
                    Scalar vel = -diff + (dice_() * 2 * diff);
                    velocities(j, i) = std::min(maxVel_, std::max(-maxVel_, vel));
                }
            }
        }

        void evaluateObjective(const Matrix &particles,
            Vector &fvals)
        {
            #pragma omp parallel for num_threads(threads_)
            for(Index i = 0; i < particles.cols(); ++i)
                fvals(i) = objective_(particles.col(i));
        }

        void maintainBounds(const Matrix &bounds, Matrix &particles) const
        {
            for(Index i = 0; i < particles.cols(); ++i)
            {
                for(Index j = 0; j < particles.rows(); ++j)
                {
                    Scalar minval = bounds(0, j);
                    Scalar maxval = bounds(1, j);
                    Scalar val = particles(j, i);
                    particles(j, i) = std::min(maxval, std::max(minval, val));
                }
            }
        }

        void calculateVelocities(const Matrix &particles,
            const Matrix &bestParticles,
            const Index gbest,
            const size_t iteration,
            Matrix &velocities)
        {
            assert(velocities.rows() == particles.rows());
            assert(velocities.cols() == particles.cols());
            assert(velocities.rows() == bestParticles.rows());
            assert(velocities.cols() == bestParticles.cols());
            assert(gbest < bestParticles.cols());

            Scalar weight = weightStrategy_(iteration, maxit_);

            for(Index i = 0; i < velocities.cols(); ++i)
            {
                for(Index j = 0; j < velocities.rows(); ++j)
                {
                    Scalar velp = dice_() * (bestParticles(j, i) - particles(j, i));
                    Scalar velg = dice_() * (bestParticles(j, gbest) - particles(j, i));
                    Scalar vel = weight * velocities(j, i) + phip_ * velp + phig_ * velg;

                    if(maxVel_ > 0)
                        vel = std::min(maxVel_, std::max(-maxVel_, vel));

                    velocities(j, i) = vel;
                }
            }
        }

        Result _minimize(const Matrix &bounds,
            Matrix &particles)
        {
            Matrix velocities(particles.rows(), particles.cols());

            Vector fvals(particles.cols());

            Matrix bestParticles = particles;
            Vector bestFvals(particles.cols());

            Matrix prevParticles(particles.rows(), particles.cols());
            Vector prevFvals(particles.cols());

            Vector diff(particles.rows());

            Index gbest = 0;
            Index prevGbest = 0;

            // initialize velocities randomly
            randomizeVelocities(bounds, velocities);

            // evaluate objective function for the initial particles
            evaluateObjective(particles, fvals);
            bestFvals = fvals;
            bestFvals.minCoeff(&gbest);

            // init stop conditions
            size_t iterations = 0;
            Scalar fdiff = std::numeric_limits<Scalar>::infinity();
            Scalar xdiff = std::numeric_limits<Scalar>::infinity();

            while(fdiff > feps_ && xdiff > xeps_ && (maxit_ == 0 || iterations < maxit_))
            {
                // calculate new velocities
                calculateVelocities(particles, bestParticles, gbest, iterations, velocities);

                // move particles by velocity and stay within bounds
                particles += velocities;
                maintainBounds(bounds, particles);

                // evaluate objective for moved particles
                evaluateObjective(particles, fvals);

                prevParticles = bestParticles;
                prevFvals = bestFvals;
                prevGbest = gbest;

                for(Index i = 0; i < fvals.size(); ++i)
                {
                    // check if there was an improvement and update best vals
                    if(fvals(i) < bestFvals(i))
                    {
                        bestFvals(i) = fvals(i);
                        bestParticles.col(i) = particles.col(i);
                    }
                }
                bestFvals.minCoeff(&gbest);

                // calculate new diffs
                xdiff = (bestParticles - prevParticles).colwise().norm().sum();
                fdiff = (bestFvals - prevFvals).array().abs().sum();

                xdiff /= bestParticles.cols();
                fdiff /= bestFvals.size();

                if(verbose_)
                {
                    std::cout << "it=" << iterations
                        << std::fixed << std::showpoint << std::setprecision(6)
                        << "\tfdiff=" <<  fdiff
                        << "\txdiff=" << xdiff
                        << "\tf=" << bestFvals(gbest)
                        << "\tx=" << bestParticles.col(gbest).transpose()
                        << std::endl;
                }

                callback_(iterations, bestParticles, bestFvals, gbest);

                ++iterations;
            }

            Result result;
            result.iterations = iterations;
            result.converged = fdiff <= feps_ || xdiff <= xeps_;
            result.fval = bestFvals(gbest);
            result.xval = bestParticles.col(gbest);

            return result;
        }

    public:

        Optimizer()
            : objective_(), callback_(), weightStrategy_(), threads_(1),
            maxit_(0), xeps_(1e-6), feps_(1e-6), phip_(2.0), phig_(2.0),
            maxVel_(0.0), verbose_(false), dice_()
        {
            std::default_random_engine gen(std::time(0));
            std::uniform_real_distribution<Scalar> distrib(0.0, 1.0);
            dice_ = std::bind(distrib, gen);
        }

        void setThreads(const size_t threads)
        {
            threads_ = threads;
        }

        void setMaxIterations(const size_t maxit)
        {
            maxit_ = maxit;
        }

        void setEpsilonX(const Scalar eps)
        {
            xeps_ = eps;
        }

        void setEpsilonF(const Scalar eps)
        {
            feps_ = eps;
        }

        void setPhiParticles(const Scalar phip)
        {
            phip_ = phip;
        }

        void setPhiGlobal(const Scalar phig)
        {
            phig_ = phig;
        }

        void setMaxVelocity(const Scalar maxVel)
        {
            maxVel_ = maxVel;
        }

        void setVerbose(const bool verbose)
        {
            verbose_ = verbose;
        }

        void setObjective(const Objective &objective)
        {
            objective_ = objective;
        }

        void setCallback(const Callback &callback)
        {
            callback_ = callback;
        }

        void setInertiaWeightStrategy(const InertiaWeightStrategy &weightStrategy)
        {
            weightStrategy_ = weightStrategy;
        }

        Result minimize(const Matrix &bounds,
            const size_t particleCnt)
        {
            if(particleCnt == 0)
                throw std::runtime_error("particle count cannot be 0");
            if(bounds.rows() != 2)
                throw std::runtime_error("bounds has not exactly 2 rows (min, max)");
            for(Index i = 0; i < bounds.cols(); ++i)
            {
                if(bounds(0, i) >= bounds(1, i))
                    throw std::runtime_error("bounds min is greater than max");
            }

            Matrix particles(bounds.cols(), particleCnt);
            randomizeParticles(bounds, particles);

            return _minimize(bounds, particles);
        }

        Result minimize(const Matrix &bounds,
            const size_t particleCnt,
            const Vector &initGuess)
        {
            if(particleCnt == 0)
                throw std::runtime_error("particle count cannot be 0");
            if(bounds.rows() != 2)
                throw std::runtime_error("bounds has not exactly 2 rows (min, max)");
            for(Index i = 0; i < bounds.cols(); ++i)
            {
                if(bounds(0, i) >= bounds(1, i))
                throw std::runtime_error("bounds min is greater than max");
            }
            if(bounds.cols() != initGuess.size())
                throw std::runtime_error("init guess and bounds have different dimensions");

            Matrix particles(bounds.cols(), particleCnt);
            randomizeParticles(bounds, particles);
            particles.col(0) = initGuess;
            maintainBounds(bounds, particles);

            return _minimize(bounds, particles);
        }

        Result minimize(const Matrix &bounds,
            Matrix &particles)
        {
            if(bounds.rows() != 2)
                throw std::runtime_error("bounds has not exactly 2 rows (min, max)");
            if(bounds.cols() != particles.rows())
                throw std::runtime_error("columns of bounds and rows of "
                    "particles do not match");
            for(Index i = 0; i < bounds.cols(); ++i)
            {
                if(bounds(0, i) >= bounds(1, i))
                    throw std::runtime_error("bounds min is greater than max");
            }

            maintainBounds(bounds, particles);

            return _minimize(bounds, particles);
        }

        void getRandomParticles(const Matrix &bounds,
            const size_t cnt,
            Matrix &particles)
        {
            particles.resize(bounds.cols(), cnt);
            randomizeParticles(bounds, particles);
        }

    };
}

#endif
