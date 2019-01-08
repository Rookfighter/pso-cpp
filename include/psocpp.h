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

namespace pso
{
    template<typename Scalar>
    class PSO
    {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef std::function<Scalar(const Vector &)> Objective;

    private:

        Objective objective_;
        Matrix::Index best_;

        bool particlesInitialized_;
        Matrix particles_;
        Matrix particlesPrev_;

        Vector funcvalues_;

        Matrix velocities_;

        Matrix bounds_;

        size_t threads_;

        size_t maxit_;
        Scalar xeps_;
        Scalar feps_;

        Scalar omega_;
        Scalar phip_;
        Scalar phig_;

        bool verbose_;

        std::function<Scalar()> dice_;

        void initParticles()
        {
            for(Matrix::Index i = 0; i < particles_.cols(); ++i)
            {
                for(Matrix::Index j = 0; j < particles_.rows(); ++j)
                {
                    Scalar min = bounds(0, j);
                    Scalar max = bounds(0, j);
                    Scalar diff = max - min;
                    particles_(j, i) = min + (dice_() * diff);
                }
            }
        }

        void initVelocities()
        {
            for(Matrix::Index i = 0; i < velocities_.cols(); ++i)
            {
                for(Matrix::Index j = 0; j < velocities_.rows(); ++j)
                {
                    Scalar min = bounds(0, j);
                    Scalar max = bounds(0, j);
                    Scalar diff = max - min;
                    velocities_(j, i) = -diff + (dice_() * 2 * diff);
                }
            }
        }

        void evaluateObjective()
        {
            #pragma omp parallel num_threads(threads == 0 ? omp_get_max_threads() : threads_)
            for(Matrix::Index i = 0; i < particles_.cols(); ++i)
            {
                Scalar tmp = objective_(particles_.col(i));
                funcvalues_(i) = tmp < funcvalues_(i) ? tmp : funcvalues_(i);
            }
            funcvalues_.minCoeff(&best_);
        }

        void evaluateVelocities()
        {
            for(Matrix::Index i = 0; i < velocities_.cols(); ++i)
            {
                for(Matrix::Index j = 0; j < velocities_.rows(); ++j)
                {
                    Scalar velp = dice_() * (particlesPrev_(j, i) - particles_(j, i));
                    Scalar velg = dice_() * (particles_(j, best_) - particles_(j, i));
                    velocities_(j, i) = omega_ * velocities_(j, i) + phip_ * velp + phig_ * velg;
                }
            }
        }

        void maintainBounds()
        {
            for(Matrix::Index i = 0; i < particles_.cols(); ++i)
            {
                for(Matrix::Index j = 0; j < particles_.rows(); ++j)
                {
                    Scalar min = bounds(0, j);
                    Scalar max = bounds(1, j);
                    Scalar val = particles_(j, i);
                    particles_(j, i) = val < min ? min : val;
                    particles_(j, i) = val > max ? max : val;
                }
            }
        }

    public:

        PSO()
            : objective_(), best_(0), particlesInitialized_(false), particles_(),
            particlesPrev_(),
            funcvalues_(), velocities_(), bounds_(), threads_(1), maxit_(0),
            xeps_(1e-6), feps_(1e-6), omega_(0.5), phip_(0.5), phig_(0.5),
            verbose_(false), dice_()
        {
            std::default_random_engine gen(std::time(0));
            std::uniform_real_distribution<Scalar> distrib(0.0, 1.0);
            dice_ = std::bind(distrib, gen);
        }

        void setObjective(const Objective &objective)
        {
            objective_ = objective;
        }

        void setBounds(const Matrix &bounds)
        {
            if(bounds.rows() != 2)
                throw std::runtime_error("bounds has not exactly 2 rows (min, max)");

            for(Matrix::Index i = 0; i < bounds.cols(); ++i)
            {
                if(bounds(0, i) >= bounds(1, i))
                    throw std::runtime_error("bounds min is greater than max");
            }

            bounds_ = bounds;
        }

        void setParticleCount(const size_t cnt, const size_t dim)
        {
            particles_.resize(dim, cnt);
            particlesPrev_.resize(dim, cnt);
            velocities_.resize(dim, cnt);
            funcvalues_.resize(cnt);
            particlesInitialized_ = false;
        }

        void setParticles(const Matrix &particles)
        {
            particles_ = particles;
            particlesPrev_.resize(particles_.rows(), particles_.cols());
            velocities_.resize(particles_.rows(), particles_.cols());
            funcvalues_.resize(particles.cols());
            particlesInitialized_ = true;
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

        void minimize()
        {
            assert(particles_.rows() == velocities_.rows());
            assert(particles_.cols() == velocities_.cols());

            if(bounds.cols() != particles_.rows())
            {
                throw std::runtime_error("columns of bounds and rows of "
                    "particles do not match");
            }

            // initialize particles if not initialized yet
            if(!particlesInitialized_)
                initParticles();
            // initialize velocities randomly
            initVelocities();

            // evaluate objective function for the initial particles
            funcvalues_ = std::numeric_limits<Scalar>::max();
            evaluateObjective();

            // init stop conditions
            size_t it = 0;
            Scalar fdiff = std::numeric_limits<Scalar>::max();
            Scalar xdiff = std::numeric_limits<Scalar>::max();

            while(fdiff > feps_ && xdiff > xeps_ && (maxit_ == 0 || it < maxit_))
            {
                // evaoluate velocities
                evaluateVelocities();
                // move particles by velocity and stay within bounds
                particlesPrev_ = particles_;
                particles_ += velocities_;
                maintainBounds();
                // evaluate objective for moved particles
                Matrix::Index bestPrev = best_;
                Scalar bestvalPrev = funcvalues_(best_);
                evaluateObjective();

                // check if there was any improvement
                if(funcvalues_(best_) < bestvalPrev)
                {
                    xdiff = (particles_.col(best) - particlesPrev_.col(bestPrev)).norm();
                    fdiff = std::abs(funcvalues_(best_) - bestvalPrev);
                }

                if(verbose_)
                {
                    std::cout << "it=" << it
                        << "\tfdiff=" << fdiff
                        << "\txdiff" << xdiff
                        << "\tf=" << funcvalues_(best_)
                        << "\tx=" << particles_.col(best_).transpose()
                        << std::endl;
                }

                ++it;
            }
        }

    };
}

#endif
