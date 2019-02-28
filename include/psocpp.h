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

    template<typename Scalar,
        typename Objective,
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

        size_t threads_;

        size_t maxit_;
        Scalar xeps_;
        Scalar feps_;

        Scalar omega_;
        Scalar phip_;
        Scalar phig_;

        bool verbose_;

        std::function<Scalar()> getDice() const
        {
            std::mt19937 gen(std::time(0));
            std::uniform_real_distribution<Scalar> distrib(0.0, 1.0);
            return std::bind(distrib, gen);
        }

        void randomizeParticles(const Matrix &bounds, Matrix &particles) const
        {
            auto dice = getDice();
            for(Index i = 0; i < particles.cols(); ++i)
            {
                for(Index j = 0; j < particles.rows(); ++j)
                {
                    Scalar min = bounds(0, j);
                    Scalar max = bounds(1, j);
                    Scalar diff = max - min;
                    particles(j, i) = min + (dice() * diff);
                }
            }
        }

        void randomizeVelocities(const Matrix &bounds, Matrix &velocities) const
        {
            auto dice = getDice();
            for(Index i = 0; i < velocities.cols(); ++i)
            {
                for(Index j = 0; j < velocities.rows(); ++j)
                {
                    Scalar min = bounds(0, j);
                    Scalar max = bounds(1, j);
                    Scalar diff = max - min;
                    velocities(j, i) = -diff + (dice() * 2 * diff);
                }
            }
        }

        void evaluateObjective(const Matrix &particles,
            Vector &fvals) const
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
                    Scalar min = bounds(0, j);
                    Scalar max = bounds(1, j);
                    Scalar val = particles(j, i);
                    particles(j, i) = val < min ? min : val;
                    particles(j, i) = val > max ? max : val;
                }
            }
        }

        void calculateVelocities(const Matrix &currParticles,
            const Matrix &prevParticles,
            const Index currBest,
            Matrix &velocities) const
        {
            assert(velocities.rows() == currParticles.rows());
            assert(velocities.cols() == currParticles.cols());
            assert(velocities.rows() == prevParticles.rows());
            assert(velocities.cols() == prevParticles.cols());
            assert(currBest < currParticles.cols());

            auto dice = getDice();
            for(Index i = 0; i < velocities.cols(); ++i)
            {
                for(Index j = 0; j < velocities.rows(); ++j)
                {
                    Scalar velp = dice() * (prevParticles(j, i) - currParticles(j, i));
                    Scalar velg = dice() * (currParticles(j, currBest) - currParticles(j, i));
                    velocities(j, i) = omega_ * velocities(j, i) + phip_ * velp + phig_ * velg;
                }
            }
        }

        Result _minimize(const Matrix &bounds,
            Matrix &currParticles) const
        {
            Matrix prevParticles(currParticles.rows(), currParticles.cols());
            Matrix velocities(currParticles.rows(), currParticles.cols());
            Vector currFvals(currParticles.cols());
            Vector prevFvals(currParticles.cols());
            Index currBest = 0;
            Index prevBest = 0;

            // initialize velocities randomly
            randomizeVelocities(bounds, velocities);

            // evaluate objective function for the initial particles
            evaluateObjective(currParticles, currFvals);
            currFvals.minCoeff(&currBest);

            // init stop conditions
            size_t iterations = 0;
            Scalar fdiff = std::numeric_limits<Scalar>::infinity();
            Scalar xdiff = std::numeric_limits<Scalar>::infinity();

            while(fdiff > feps_ && xdiff > xeps_ && (maxit_ == 0 || iterations < maxit_))
            {
                prevParticles = currParticles;
                prevFvals = currFvals;
                prevBest = currBest;

                // move particles by velocity and stay within bounds
                currParticles += velocities;
                maintainBounds(bounds, currParticles);

                // evaluate objective for moved particles
                evaluateObjective(currParticles, currFvals);
                for(Index i = 0; i < currFvals.size(); ++i)
                {
                    // if there was no improvement revert to old value
                    if(currFvals(i) >= prevFvals(i))
                    {
                        currFvals(i) = prevFvals(i);
                        currParticles.col(i) = prevParticles.col(i);
                    }
                }
                currFvals.minCoeff(&currBest);

                // calculate new velocities
                calculateVelocities(currParticles, prevParticles, currBest, velocities);

                // check if there was any improvement
                if(currFvals(currBest) < prevFvals(prevBest))
                {
                    xdiff = (currParticles.col(currBest) -
                        prevParticles.col(prevBest)).norm();
                    fdiff = std::abs(currFvals(currBest) - prevFvals(prevBest));
                }

                if(verbose_)
                {
                    std::cout << "it=" << iterations
                        << std::fixed << std::showpoint << std::setprecision(6)
                        << "\tfdiff=" <<  fdiff
                        << "\txdiff=" << xdiff
                        << "\tf=" << currFvals(currBest)
                        << "\tx=" << currParticles.col(currBest).transpose()
                        << std::endl;
                }

                callback_(iterations, currParticles, currFvals, currBest);

                ++iterations;
            }

            Result result;
            result.iterations = iterations;
            result.converged = fdiff <= feps_ || xdiff <= xeps_;
            result.fval = currFvals(currBest);
            result.xval = currParticles.col(currBest);

            return result;
        }

    public:

        Optimizer()
            : threads_(1), maxit_(0), xeps_(1e-6), feps_(1e-6), omega_(0.5),
            phip_(0.5), phig_(0.5), verbose_(false)
        {
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

        void setOmega(const Scalar omega)
        {
            omega_ = omega;
        }

        void setPhiParticles(const Scalar phip)
        {
            phip_ = phip;
        }

        void setPhiGlobal(const Scalar phig)
        {
            phig_ = phig;
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

        Result minimize(const Matrix &bounds,
            const size_t particleCnt) const
        {
            if(bounds.rows() != 2)
                throw std::runtime_error("bounds has not exactly 2 rows (min, max)");
            for(Index i = 0; i < bounds.cols(); ++i)
            {
                if(bounds(0, i) >= bounds(1, i))
                    throw std::runtime_error("bounds min is greater than max");
            }

            Matrix currParticles(bounds.cols(), particleCnt);
            randomizeParticles(bounds, currParticles);

            return _minimize(bounds, currParticles);
        }

        Result minimize(const Matrix &bounds,
            const Matrix &particles) const
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

            Matrix currParticles = particles;
            maintainBounds(bounds, currParticles);

            return _minimize(bounds, currParticles);
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
