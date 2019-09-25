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
    /** Integer type for indexing arrays, vectors and matrices. */
    typedef long int Index;

    /** @brief Dummy callback functor, which always and only returns true. */
    template<typename Scalar>
    class NoCallback
    {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

        bool operator()(const Index, const Matrix&, const Vector &, const Index) const
        {
            return true;
        }
    };

    /** @brief Inertia weight functor, which returns a constant weight. */
    template<typename Scalar>
    class ConstantWeight
    {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    private:
        Scalar weight_;
    public:
        ConstantWeight()
            : ConstantWeight(1.0)
        { }

        /** Constructor, which accepts the weight that is returned by the functor.
          * @param weight constant which will be returned as inertia weight */
        ConstantWeight(const Scalar weight)
            : weight_(weight)
        { }

        Scalar operator()(const Index,
            const Index) const
        {
            return weight_;
        }
    };

    /** @brief Inertia weight functor, which decreases linearly over time.
      *
      * The inertia weight is calculated by the following formula:
      *
      * w = wMin + (wMax - wMin) * (t / tMax) */
    template<typename Scalar>
    class LinearDecrease
    {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    private:
        Scalar weightMin_;
        Scalar weightMax_;
    public:
        LinearDecrease()
            : LinearDecrease(0.4, 0.9)
        { }

        /** @brief Constructor, which accepts the minimum and maximum weight of
          * the linear decrease.
          *
          * The returned inertia weight always lies in the interval [minval, maxval].
          * @param minval lower bound of the inertia weight
          * @param maxval upper bound of the inertia weight */
        LinearDecrease(const Scalar minval,
            const Scalar maxval)
            : weightMin_(minval), weightMax_(maxval)
        { }

        Scalar operator()(const Index iteration,
            const Index maxIt) const
        {
            Scalar factor = static_cast<Scalar>(iteration) / static_cast<Scalar>(maxIt);
            return weightMin_ + (weightMax_ - weightMin_) * factor;
        }
    };

    /** @brief Inertia weight functor, which decreases exponentially over time.
      *
      * The inertia weight is calculated by the following formula:
      *
      * w = wMin + (wMax - wMin) * exp(-t / (tMax / 10)) */
    template<typename Scalar>
    class ExponentialDecrease1
    {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    private:
        Scalar weightMin_;
        Scalar weightMax_;
    public:

        ExponentialDecrease1()
            : ExponentialDecrease1(0.4, 0.9)
        { }

        /** Constructor, which accepts the minimum and maximum weight of the
          * exponential decrease.
          * The returned inertia weight always lies in the interval [minval, maxval].
          * @param minval lower bound of the inertia weight
          * @param maxval upper bound of the inertia weight */
        ExponentialDecrease1(const Scalar minval, const Scalar maxval)
            : weightMin_(minval), weightMax_(maxval)
        { }

        Scalar operator()(const Index iteration,
            const Index maxIt) const
        {
            Scalar exponent = static_cast<Scalar>(iteration) / (static_cast<Scalar>(maxIt) / 10.0);
            return weightMin_ + (weightMax_ - weightMin_) * std::exp(-exponent);
        }
    };

    /** @brief Inertia weight functor, which decreases exponentially over time.
      *
      * The inertia weight is calculated by the following formula:
      *
      * w = wMin + (wMax - wMin) * exp(-(t / (tMax / 4))^2) */
    template<typename Scalar>
    class ExponentialDecrease2
    {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    private:
        Scalar weightMin_;
        Scalar weightMax_;
    public:
        ExponentialDecrease2()
            : ExponentialDecrease2(0.4, 0.9)
        { }

        /** Constructor, which accepts the minimum and maximum weight of the
          * exponential decrease.
          * The returned inertia weight always lies in the interval [minval, maxval].
          * @param minval lower bound of the inertia weight
          * @param maxval upper bound of the inertia weight */
        ExponentialDecrease2(const Scalar minval, const Scalar maxval)
            : weightMin_(minval), weightMax_(maxval)
        { }

        Scalar operator()(const Index iteration,
            const Index maxIt) const
        {
            Scalar exponent = static_cast<Scalar>(iteration) / (static_cast<Scalar>(maxIt) / 4.0);
            exponent *= exponent;
            return weightMin_ + (weightMax_ - weightMin_) * std::exp(-exponent);
        }
    };

    /** @brief Inertia weight functor, which decreases exponentially over time.
      *
      * The inertia weight is calculated by the following formula:
      *
      * w = (wMax - wMin - d1) * exp(1 / (1 + d2 t / tMax)) */
    template<typename Scalar>
    class ExponentialDecrease3
    {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    private:
        Scalar weightMin_;
        Scalar weightMax_;
        /** Control factors */
        Scalar d1_;
        Scalar d2_;
    public:

        ExponentialDecrease3()
            : ExponentialDecrease3(0.4, 0.95, 0.2, 7.0)
        { }

        /** Constructor, which accepts the minimum and maximum weight and two
          * control factors of the exponential decrease.
          * The returned inertia weight always lies in the interval [minval, maxval].
          * @param minval lower bound of the inertia weight
          * @param maxval upper bound of the inertia weight
          * @param d1 first control factor
          * @param d2 second control factor */
        ExponentialDecrease3(const Scalar minval,
            const Scalar maxval,
            const Scalar d1,
            const Scalar d2)
            : weightMin_(minval), weightMax_(maxval), d1_(d1), d2_(d2)
        { }

        Scalar operator()(const Index iteration,
            const Index maxIt) const
        {
            Scalar itFac = static_cast<Scalar>(iteration) / static_cast<Scalar>(maxIt);
            Scalar exponent = 1.0 / (1.0 + d2_ * itFac);
            return (weightMax_ - weightMin_ - d1_) * std::exp(exponent);
        }
    };

    /** @brief Implements the paricle swarm optimization agorithm.
      *
      * The optimization process can be configured by providing an inertia
      * weight strategy functor and a callback.
      *
      * The inertia weight functor determines the amount of velocity, which is
      * is maintained from the previous iterations. It has a huge effect on
      * convergence speed and stability of the optimization.
      *
      * The callback functor is called after each iteration and returns a boolean.
      * If it returns false the optimization process is stopped. As such, the
      * callback allows to implement additional stop criteria. */
    template<typename Scalar,
        typename Objective,
        typename InertiaWeightStrategy = ConstantWeight<Scalar>,
        typename Callback = NoCallback<Scalar> >
    class ParticleSwarmOptimization
    {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

        struct Result
        {
            Index iterations;
            bool converged;
            Scalar fval;
            Vector xval;
        };

    private:
        Objective objective_;
        Callback callback_;
        InertiaWeightStrategy weightStrategy_;

        Index threads_;

        Index maxIt_;
        Scalar xeps_;
        Scalar feps_;

        Scalar phip_;
        Scalar phig_;
        Scalar maxVel_;

        Index verbosity_;

        std::function<Scalar()> dice_;

        template<typename Derived>
        std::string vector2str(const Eigen::MatrixBase<Derived> &vec) const
        {
            std::stringstream ss1;
            ss1 << std::fixed << std::showpoint << std::setprecision(6);
            std::stringstream ss2;
            ss2 << '[';
            for(Index i = 0; i < vec.size(); ++i)
            {
                ss1 << vec(i);
                ss2 << std::setfill(' ') << std::setw(10) << ss1.str();
                if(i != vec.size() - 1)
                    ss2 << ' ';
                ss1.str("");
            }
            ss2 << ']';

            return ss2.str();
        }

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
            const Index iteration,
            Matrix &velocities)
        {
            assert(velocities.rows() == particles.rows());
            assert(velocities.cols() == particles.cols());
            assert(velocities.rows() == bestParticles.rows());
            assert(velocities.cols() == bestParticles.cols());
            assert(gbest < bestParticles.cols());

            Scalar weight = weightStrategy_(iteration, maxIt_);

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

            // initialize velocities randomly
            randomizeVelocities(bounds, velocities);

            // evaluate objective function for the initial particles
            evaluateObjective(particles, fvals);
            bestFvals = fvals;
            bestFvals.minCoeff(&gbest);

            // init stop conditions
            Index iterations = 0;
            Scalar fchange = feps_ + 1;
            Scalar xchange = xeps_ + 1;

            while((maxIt_ == 0 || iterations < maxIt_) &&
                fchange > feps_ && xchange > xeps_)
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
                xchange = (bestParticles - prevParticles).colwise().norm().sum();
                fchange = (bestFvals - prevFvals).array().abs().sum();

                xchange /= bestParticles.cols();
                fchange /= bestFvals.size();

                // evaluate callback and save its result
                bool callbackResult = callback_(iterations, bestParticles,
                    bestFvals, gbest);

                if(verbosity_ > 0)
                {
                    std::stringstream ss;
                    ss << "it=" << std::setfill('0')
                        << std::setw(4) << iterations
                        << std::fixed << std::showpoint << std::setprecision(6)
                        << "    fchange=" << fchange
                        << "    xchange=" << xchange;

                    if(verbosity_ > 2)
                        ss << "    callback=" << (callbackResult ? "true" : "false");

                    ss << "    fval=" << bestFvals(gbest);

                    if(verbosity_ > 1)
                        ss << "    xval=" << vector2str(bestParticles.col(gbest));

                    std::cout << ss.str() << std::endl;;
                }

                ++iterations;
            }

            Result result;
            result.iterations = iterations;
            result.converged = fchange <= feps_ || xchange <= xeps_;
            result.fval = bestFvals(gbest);
            result.xval = bestParticles.col(gbest);

            return result;
        }

    public:

        ParticleSwarmOptimization()
            : objective_(), callback_(), weightStrategy_(), threads_(1),
            maxIt_(0), xeps_(static_cast<Scalar>(1e-6)),
            feps_(static_cast<Scalar>(1e-6)), phip_(static_cast<Scalar>(2.0)),
            phig_(static_cast<Scalar>(2.0)), maxVel_(static_cast<Scalar>(0.0)),
            verbosity_(0), dice_()
        {
            std::default_random_engine gen(std::time(0));
            std::uniform_real_distribution<Scalar> distrib(0.0, 1.0);
            dice_ = std::bind(distrib, gen);
        }

        /** Set the amount of threads, which are used for evaluating the
          * individual particles (OMP only).
          * Set to 0 or negative to allow auto detection.
          * @param threads maximum number of threads for evaluation */
        void setThreads(const Index threads)
        {
            threads_ = threads;
        }

        /** Set the maximum number of iterations.
          * Set to 0 or negative for infinite iterations.
          * @param iterations maximum number of iterations */
        void setMaxIterations(const Index iterations)
        {
            maxIt_ = iterations;
        }

        /** Set the minimum average change of particles per iteration.
          * If the average change of particles (input parameters) falls below
          * this value, the optimization terminates.
          * @param change minimum change of input paramaters */
        void setMinParticleChange(const Scalar change)
        {
            xeps_ = change;
        }

        /** Set the minimum average change of function values per iteration.
          * If the average change of functions values falls below
          * this value, the optimization terminates.
          * @param change minimum change of function values */
        void setMinFunctionChange(const Scalar change)
        {
            feps_ = change;
        }

        /** Set the tendency of particles to move towards their local optimum
          * found so far.
          * Each particle individually maintains a memory of where it has
          * visited the lowest function value so far.
          * Increasing this value increases the particles' tendency to move
          * towards that point.
          * @param phip tendency to move towards individual optimum */
        void setPhiParticles(const Scalar phip)
        {
            phip_ = phip;
        }

        /** Set the tendency of particles to move towards the global optimum
          * found so far.
          * The swarm maintains a collective memory of where it has visited the
          * lowest function value so far.
          * Increasing this value increases the particles' tendency to move
          * towards that point.
          * @param phig tendency to move towards collective optimum */
        void setPhiGlobal(const Scalar phig)
        {
            phig_ = phig;
        }

        /** Set an upper bound for the velocity of particles.
          * A particle cannot move faster than this value, which may prevent
          * divergence.
          * @param maxvel maximum velocity of a particle */
        void setMaxVelocity(const Scalar maxvel)
        {
            maxVel_ = maxvel;
        }

        /** Set the level of verbosity during optimization.
          * Verbosity increases with increasing value.
          * 0 means no output and it can be raised up to level 3.
          * @param verbosity level of verbosity */
        void setVerbosity(const Index verbosity)
        {
            verbosity_ = verbosity;
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

        /** Perform minimization with the given bounds and number of particels.
          *
          * The swarm of particles will be drawn uniform randomly within the
          * given bounds.
          *
          * The bounds matrix has to have 2 rows and one column per dimension
          * of particle. The first row holds the minimum value of the respective
          * dimension and the second row holds the maximum value.
          *
          * @param bounds 2xM matrix for bounds of M-dimensional particles
          * @param cnt number of particles used for optimization */
        Result minimize(const Matrix &bounds,
            const Index cnt)
        {
            if(cnt == 0)
                throw std::runtime_error("particle count cannot be 0");
            if(bounds.rows() != 2)
                throw std::runtime_error("bounds has not exactly 2 rows (min, max)");
            for(Index i = 0; i < bounds.cols(); ++i)
            {
                if(bounds(0, i) >= bounds(1, i))
                    throw std::runtime_error("bounds min is greater than max");
            }

            Matrix particles(bounds.cols(), cnt);
            randomizeParticles(bounds, particles);

            return _minimize(bounds, particles);
        }

        /** Perform minimization with the given bounds, number of particels and
          * initial guess.
          *
          * The swarm of particles will be drawn uniform randomly within the
          * given bounds.
          *
          * The bounds matrix has to have 2 rows and one column per dimension
          * of particle. The first row holds the minimum value of the respective
          * dimension and the second row holds the maximum value.
          *
          * The initial guess vector has to have the same length as the number
          * of columns of the bounds. It will be included as one particle of
          * the swarm.
          *
          * @param bounds 2xM matrix for bounds of M-dimensional particles
          * @param cnt number of particles used for optimization
          * @param initGuess initial guess for a particle */
        Result minimize(const Matrix &bounds,
            const Index cnt,
            const Vector &initGuess)
        {
            if(cnt == 0)
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

            Matrix particles(bounds.cols(), cnt);
            randomizeParticles(bounds, particles);
            particles.col(0) = initGuess;
            maintainBounds(bounds, particles);

            return _minimize(bounds, particles);
        }

        /** Perform minimization with the given bounds and a pre-computed
          * swarm of particles.
          *
          * The bounds matrix has to have 2 rows and one column per dimension
          * of particle. The first row holds the minimum value of the respective
          * dimension and the second row holds the maximum value.
          *
          * @param bounds 2xM matrix for bounds of M-dimensional particles
          * @param particles initial swarm used for optimization */
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
            const Index cnt,
            Matrix &particles)
        {
            particles.resize(bounds.cols(), cnt);
            randomizeParticles(bounds, particles);
        }

    };
}

#endif
