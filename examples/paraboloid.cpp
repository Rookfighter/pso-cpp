#include <psocpp.h>

// Implement an objective functor.
struct Paraboloid
{
    template<typename Derived>
    float operator()(const Eigen::MatrixBase<Derived> &xval) const
    {
        return xval(0) * xval(0) + xval(1) * xval(1);
    }
};

int main()
{
    // Create optimizer object with Paraboloid functor as objective.
    // You can additionally specify an InertiaWeightStrategy functor as template
    // parameter.
    // You can additionally specify a Callback functor as template parameter.
    pso::Optimizer<float, Paraboloid> optimizer;

    // Set number of iterations as stop criterion.
    // Set it to 0 or negative for infinite iterations (default is 0).
    optimizer.setMaxIterations(100);

    // Set the minimum change of the x-values (particles) (default is 1e-6).
    // If the change is lower then the optimizer stops minimizing.
    optimizer.setEpsilonX(1e-3);

    // Set the minimum change of the function values (default is 1e-6).
    // If the change is lower then the optimizer stops minimizing.
    optimizer.setEpsilonF(1e-3);

    // Set the number of threads used for evaluation (OpenMP only).
    // Set it to 0 or negative for auto detection (default is 1).
    optimizer.setThreads(2);

    // Turn verbosity on, so the optimizer prints status updates after each
    // iteration.
    optimizer.setVerbose(true);

    // Set the bounds in which the optimizer should search.
    // Each column vector defines the (min, max) for each dimension  of the
    // particles.
    Eigen::MatrixXf bounds(2, 2);
    bounds << -4, -4,
               4,  4;

    // start the optimization with a particle count
    auto result = optimizer.minimize(bounds, 100);

    std::cout << "Done! Converged: " << (result.converged ? "true" : "false")
        << " Iterations: " << result.iterations << std::endl;

    // do something with final function value
    std::cout << "Final fval: " << result.fval << std::endl;

    // do something with final x-value
    std::cout << "Final xval: " << result.xval.transpose() << std::endl;

    return 0;
}
