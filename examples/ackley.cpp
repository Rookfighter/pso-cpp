#include <psocpp.h>

// Implement an objective functor.
struct Ackley
{
    static double pi()
    { return 3.141592653589; }

    Ackley()
    { }

    template<typename Derived>
    double operator()(const Eigen::MatrixBase<Derived> &xval) const
    {
        assert(xval.size() == 2);
        double x = xval(0);
        double y = xval(1);
        return -20.0 * std::exp(-0.2 * std::sqrt(0.5 * (x * x + y * y))) -
            std::exp(0.5 * (std::cos(2 * pi() * x) + std::cos(2 * pi() * y))) +
            std::exp(1) + 20.0;
    }
};

int main()
{
    // Create optimizer object with Ackley functor as objective.
    //
    // You can specify an InertiaWeightStrategy functor as
    // template parameter. There are ConstantWeight, LinearDecrease,
    // ExponentialDecrease1, ExponentialDecrease2, ExponentialDecrease3
    // available. (Default is ConstantWeight)
    //
    // You can additionally specify a Callback functor as template parameter.
    pso::Optimizer<double, Ackley,
        pso::ExponentialDecrease2<double>> optimizer;

    // Set number of iterations as stop criterion.
    // Set it to 0 or negative for infinite iterations (default is 0).
    optimizer.setMaxIterations(100);

    // Set the minimum change of the x-values (particles) (default is 1e-6).
    // If the change in the current iteration is lower than this value, then
    // the optimizer stops minimizing.
    optimizer.setMinParticleChange(1e-6);

    // Set the minimum change of the function values (default is 1e-6).
    // If the change in the current iteration is lower than this value, then
    // the optimizer stops minimizing.
    optimizer.setMinFunctionChange(1e-6);

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
    bounds << -5, -5,
               5,  5;

    // start the optimization with a particle count
    auto result = optimizer.minimize(bounds, 40);

    std::cout << "Done! Converged: " << (result.converged ? "true" : "false")
        << " Iterations: " << result.iterations << std::endl;

    // do something with final function value
    std::cout << "Final fval: " << result.fval << std::endl;

    // do something with final x-value
    std::cout << "Final xval: " << result.xval.transpose() << std::endl;

    return 0;
}
