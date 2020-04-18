//Base functions
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
//Linear algebra classes
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
//Setup grid
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_refinement.h>
//Manage degree of freedom
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
//Finite element section
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
//Numeric component and solver
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h> // it works to transfer solution from the old to the new refinement
//Output
#include <fstream>
#include <iostream>
//Parallel
#include <deal.II/base/function_parser.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/timer.h>
#include <deal.II/meshworker/copy_data.h>
#include <deal.II/meshworker/scratch_data.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/distributed/solution_transfer.h>


namespace Step15
{
    using namespace dealii;

    namespace LA {
          //using namespace dealii::LinearAlgebraPETSc;
          using namespace dealii::LinearAlgebraTrilinos;
      }

    template <int dim>
    class MinimalSurfaceProblem
    {
    public:
        MinimalSurfaceProblem();
        ~MinimalSurfaceProblem();
        void run();
    private:
        void   setup_system(const bool initial_step);
        void   assemble_system();
        void   solve();
        void   refine_mesh();
        void   set_boundary_values();
        double compute_residual(const double alpha) const;
        double determine_step_length() const;
        void output_results(const unsigned refinement) const;

        /******************PARALLEL*************
        ******************PARALLEL**************/
        MPI_Comm communicator;
        ConditionalOStream pout;
        mutable TimerOutput timer;
        parallel::distributed::Triangulation<dim> triangulation;
        //Triangulation<dim> triangulation;
        DoFHandler<dim> dof_handler;
        FE_Q<dim>       fe;

        /******************PARALLEL*************
        ******************PARALLEL*************/

        AffineConstraints<double> hanging_node_constraints;
        IndexSet locally_relevant_dofs;
        IndexSet locally_owned_dofs;
        LA::MPI::SparseMatrix system_matrix;
        LA::MPI::Vector present_solution;
        LA::MPI::Vector newton_update;
        LA::MPI::Vector system_rhs;
        LA::MPI::Vector locally_relevant_solution;


        //SparsityPattern      sparsity_pattern;
        //SparseMatrix<double> system_matrix;
        //Vector<double> present_solution;
        //Vector<double> newton_update;
        //Vector<double> system_rhs;
    };
    template <int dim>
    class BoundaryValues : public Function<dim>
    {
    public:
        BoundaryValues()
                : Function<dim>()
        {}
        virtual double value(const Point<dim> & p,
                             const unsigned int component = 0) const override;
    };
    template <int dim>
    double BoundaryValues<dim>::value(const Point<dim> &p,
                                      const unsigned int /*component*/) const
    {
        return std::sin(2 * numbers::PI * (p[0] + p[1])); //g(x,y)=sin(2π(x+y))  --> p[0]=x p[1]=y
    }
    template <int dim>
    MinimalSurfaceProblem<dim>::MinimalSurfaceProblem()
    :
    communicator(MPI_COMM_WORLD)
    ,pout(std::cout, Utilities::MPI::this_mpi_process(communicator)==0)
    ,timer(pout, TimerOutput::summary, TimerOutput::cpu_and_wall_times)
    , triangulation(communicator)
    , dof_handler(triangulation)
    , fe(2)
    {}
    template <int dim>
    MinimalSurfaceProblem<dim>::~MinimalSurfaceProblem()
    {
        dof_handler.clear();
    }
    template <int dim>
    void MinimalSurfaceProblem<dim>::setup_system(const bool initial_step)
    {

        if (initial_step)
        {
            TimerOutput::Scope timer_section(timer, "Setup system");
            dof_handler.distribute_dofs(fe);

            locally_owned_dofs = dof_handler.locally_owned_dofs();
            DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
            present_solution.reinit(locally_owned_dofs,communicator);

            //present_solution.reinit(dof_handler.n_dofs());
            hanging_node_constraints.clear();
            DoFTools::make_hanging_node_constraints(dof_handler,
                                                    hanging_node_constraints);
            /******************PARALLEL*************
            VectorTools::interpolate_boundary_values(dof_handler,
                                                      0,
                                                      exact_solution,
                                                      hanging_node_constraints);
            ****************PARALLEL*************/
            hanging_node_constraints.close();
        }
        /******************PARALLEL*************
        ****************PARALLEL*************/
        newton_update.reinit(locally_owned_dofs,communicator);
        system_rhs.reinit(locally_owned_dofs,communicator);
        //newton_update.reinit(dof_handler.n_dofs());
        //system_rhs.reinit(dof_handler.n_dofs());
        DynamicSparsityPattern dsp(dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler, dsp);

        SparsityTools::distribute_sparsity_pattern(
                dsp,
                dof_handler.n_locally_owned_dofs_per_processor(),
                communicator,
                locally_relevant_dofs);


        hanging_node_constraints.condense(dsp);
        //sparsity_pattern.copy_from(dsp); // comment this line for ***PARALLEL*** part
        /******************PARALLEL*************
        ****************PARALLEL*************/
        system_matrix.reinit(locally_owned_dofs,dsp,communicator);
        //system_matrix.reinit(sparsity_pattern);

    }
    template <int dim>
    void MinimalSurfaceProblem<dim>::assemble_system()
    {
        TimerOutput::Scope timer_section(timer, "Assemble system");
        const QGauss<dim> quadrature_formula(fe.degree + 1);
        system_matrix = 0;
        system_rhs    = 0;

          MeshWorker::ScratchData<dim> scratch(fe,
        //FEValues<dim> fe_values(fe, // comment this line for ***PARALLEL*** part
                                quadrature_formula,
                                update_gradients | update_quadrature_points | // update_values for ***PARALLEL*** part??
                                update_JxW_values);
        const unsigned int dofs_per_cell = fe.dofs_per_cell;
        const unsigned int n_q_points    = quadrature_formula.size();
        /******************PARALLEL*************
        ****************PARALLEL*************/
        MeshWorker::CopyData<1,1,1> copy_data(dofs_per_cell);
        //FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell); // comment this line for ***PARALLEL*** part
        //Vector<double>     cell_rhs(dofs_per_cell);
        std::vector<Tensor<1, dim>> old_solution_gradients(n_q_points);
        //std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);// comment this line for ***PARALLEL*** part


          auto worker = [&](
            const decltype(dof_handler.begin_active()) &cell,
            MeshWorker::ScratchData<dim> &scratch,
            MeshWorker::CopyData<1, 1, 1> &copy_data) {

              //for (const auto &cell : dof_handler.active_cell_iterators()){ // comment this line for ***PARALLEL*** part

              /******************PARALLEL*************
              ****************PARALLEL*************/
              //cell_matrix = 0; // comment this line for ***PARALLEL*** part
              //cell_rhs    = 0; // comment\ this line for ***PARALLEL*** part
              //fe_values.reinit(cell); // comment this line for ***PARALLEL*** part
              copy_data.matrices[0] = 0;
              copy_data.vectors[0] = 0;
              auto &fe_values = scratch.reinit(cell);
              fe_values.get_function_gradients(present_solution,
                                               old_solution_gradients);
              for (unsigned int q = 0; q < n_q_points; ++q) {
                  const double coeff =
                          1.0 / std::sqrt(1 + old_solution_gradients[q] *
                                              old_solution_gradients[q]);
                  for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                          /******************PARALLEL*************
                          ****************PARALLEL*************/
                          //cell_matrix(i, j) += // comment this line for ***PARALLEL*** part
                          copy_data.matrices[0](i, j) +=
                                  (((fe_values.shape_grad(i, q)      // ((\nabla \phi_i
                                     * coeff                         //   * a_n
                                     * fe_values.shape_grad(j, q))   //   * \nabla \phi_j)
                                    -                                //  -
                                    (fe_values.shape_grad(i, q)      //  (\nabla \phi_i
                                     * coeff * coeff * coeff         //   * a_n^3
                                     * (fe_values.shape_grad(j, q)   //   * (\nabla \phi_j
                                        * old_solution_gradients[q]) //      * \nabla u_n)
                                     * old_solution_gradients[q]))   //   * \nabla u_n)))
                                   * fe_values.JxW(q));              // * dx
                      /******************PARALLEL*************
                     ****************PARALLEL*************/
                      //cell_rhs(i) -= (fe_values.shape_grad(i, q)  // \nabla \phi_i // comment this line for ***PARALLEL*** part
                      copy_data.vectors[0](i) -= (fe_values.shape_grad(i, q) //is it a vector or a matrix?
                                                     * coeff                     // * a_n
                                                     * old_solution_gradients[q] // * u_n
                                                     * fe_values.JxW(q));        // * dx
                  }
              }
              /******************PARALLEL*************
              ****************PARALLEL*************/
              cell->get_dof_indices(copy_data.local_dof_indices[0]);
              //cell->get_dof_indices(local_dof_indices);// comment this line for ***PARALLEL*** part
              /******************PARALLEL*************
              ****************PARALLEL*************/

              /* for (unsigned int i = 0; i < dofs_per_cell; ++i)// comment this for loop for ***PARALLEL***
               {
                   for (unsigned int j = 0; j < dofs_per_cell; ++j)
                       system_matrix.add(local_dof_indices[i],
                                         local_dof_indices[j],
                                         cell_matrix(i, j));
                   system_rhs(local_dof_indices[i]) += cell_rhs(i);
               }*/
          };


        auto copier = [&](const MeshWorker::CopyData<1, 1, 1> &copy_data) {
            hanging_node_constraints.distribute_local_to_global(copy_data.matrices[0],
                                                                copy_data.vectors[0],
                                                                copy_data.local_dof_indices[0],
                                                                system_matrix,
                                                                system_rhs);
        };

        hanging_node_constraints.condense(system_matrix);
        hanging_node_constraints.condense(system_rhs);


        using CellFilter = FilteredIterator<typename DoFHandler<dim>::active_cell_iterator >;
        WorkStream::run(
                CellFilter(IteratorFilters::LocallyOwnedCell(),
                           dof_handler.begin_active()),
                CellFilter(IteratorFilters::LocallyOwnedCell(),
                           dof_handler.end()),
                worker,
                copier,
                scratch,
                copy_data);

        std::map<types::global_dof_index, double> boundary_values;
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 0,
                                                 Functions::ZeroFunction<dim>(),
                                                 boundary_values);
        MatrixTools::apply_boundary_values(boundary_values,
                                           system_matrix,
                                           newton_update,
                                           system_rhs);
        /****************PARALLEL*************
        ****************PARALLEL*************/
        system_matrix.compress(VectorOperation::add);
        system_rhs.compress(VectorOperation::add);
    }
    template <int dim>
    void MinimalSurfaceProblem<dim>::solve()
    {

        TimerOutput::Scope timer_section(timer, "Solve system");
        SolverControl solver_control(system_rhs.size(),
                                     system_rhs.l2_norm() * 1e-6);
        /****************PARALLEL*************
        ****************PARALLEL*************/
        //SolverCG<>    solver(solver_control); // comment this line for ***PARALLEL*** part
        //PreconditionSSOR<> preconditioner; // comment this line for ***PARALLEL*** part
        //preconditioner.initialize(system_matrix, 1.2); // comment this line for ***PARALLEL*** part
        //solver.solve(system_matrix, newton_update, system_rhs, preconditioner); // comment this line for ***PARALLEL*** part
        LA::SolverCG solver(solver_control);
        LA::MPI::PreconditionSSOR::AdditionalData data(1.2);
        LA::MPI::PreconditionSSOR preconditioner;
        preconditioner.initialize(system_matrix);
        //LA::MPI::PreconditionAMG::AdditionalData data;
        //LA::MPI::PreconditionAMG amg;
        //amg.initialize(system_matrix);
        //solver.solve(system_matrix, newton_update, system_rhs, amg);
        solver.solve(system_matrix, newton_update, system_rhs, preconditioner);
        hanging_node_constraints.distribute(newton_update);
        const double alpha = determine_step_length();
        present_solution.add(alpha, newton_update);
        locally_relevant_solution = present_solution;
    }
    template <int dim>
    void MinimalSurfaceProblem<dim>::refine_mesh()
    {
        TimerOutput::Scope timer_section(timer, "Refine mesh");
        //Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
        Vector<float> estimated_error_per_cell(triangulation.n_locally_owned_active_cells());
        KellyErrorEstimator<dim>::estimate(
                dof_handler,
                QGauss<dim - 1>(fe.degree + 1),
                std::map<types::boundary_id, const Function<dim> *>(),
                locally_relevant_solution,
                //present_solution, --> cambiato in locally_relevant_solution
                estimated_error_per_cell);
        /****************PARALLEL*************
        ****************PARALLEL*************/
        parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                                               estimated_error_per_cell,
                                                                               0.3,
                                                                               0.03);
        /*GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                        estimated_error_per_cell,
                                                        0.3,
                                                        0.03); // comment this line for ***PARALLEL*** part **/
        triangulation.prepare_coarsening_and_refinement();
        parallel::distributed::SolutionTransfer<dim,LA::MPI::Vector> solution_transfer(dof_handler);
        //parallel::distributed::SolutionTransfer<dim> solution_transfer(dof_handler); //applicare parallel solution transfer(locally_relevant_solution)
        solution_transfer.prepare_for_coarsening_and_refinement(locally_relevant_solution);
        triangulation.execute_coarsening_and_refinement();
        dof_handler.distribute_dofs(fe);
        //Vector<double> (dof_handler.n_dofs());
        auto tmp(locally_relevant_solution);
        //LA::MPI::Vector tmp(dof_handler.n_dofs()); //-->auto tmp(locally_relevant_solution)
        //solution_transfer.interpolate(present_solution, tmp);
        solution_transfer.interpolate(present_solution);
        //solution_transfer.interpolate(locally_relevant_solution, tmp);
        present_solution = tmp;
        set_boundary_values();
        hanging_node_constraints.clear();
        DoFTools::make_hanging_node_constraints(dof_handler,
                                                hanging_node_constraints);
        hanging_node_constraints.close();
        hanging_node_constraints.distribute(present_solution);
        setup_system(false);
    }
    template <int dim>
    void MinimalSurfaceProblem<dim>::set_boundary_values()
    {
        std::map<types::global_dof_index, double> boundary_values;
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 0,
                                                 BoundaryValues<dim>(),
                                                 boundary_values);
        for (auto &boundary_value : boundary_values)
            present_solution(boundary_value.first) = boundary_value.second;
    }
    template <int dim>
    double MinimalSurfaceProblem<dim>::compute_residual(const double alpha) const
    {
        TimerOutput::Scope timer_section(timer, "Compute residual");
        Vector<double> residual(dof_handler.n_dofs());
        //Vector<double> evaluation_point(dof_handler.n_dofs());
        LA::MPI::Vector locally_relevant_evaluation_point(locally_relevant_solution);
        locally_relevant_evaluation_point.add(alpha, newton_update);
        //evaluation_point = present_solution;
       // evaluation_point.add(alpha, newton_update);
        const QGauss<dim> quadrature_formula(fe.degree + 1);

        MeshWorker::ScratchData<dim> scratch(fe,
        //FEValues<dim>     fe_values(fe,
                                    quadrature_formula,
                                    update_gradients | update_quadrature_points |
                                    update_JxW_values);
        const unsigned int dofs_per_cell = fe.dofs_per_cell;
        const unsigned int n_q_points    = quadrature_formula.size();

        MeshWorker::CopyData<0,1,1> copy_data(dofs_per_cell);

        //Vector<double>              cell_residual(dofs_per_cell);
        std::vector<Tensor<1, dim>> gradients(n_q_points);
        //std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        auto worker = [&](
                const decltype(dof_handler.begin_active()) &cell,
                MeshWorker::ScratchData<dim> &scratch,
                MeshWorker::CopyData<0, 1, 1> &copy_data) {


            //for (const auto &cell : dof_handler.active_cell_iterators())


            //cell_residual = 0;
            copy_data.vectors[0]=0;
            //fe_values.reinit(cell);
            auto &fe_values = scratch.reinit(cell);
            //fe_values.get_function_gradients(evaluation_point, gradients);
            fe_values.get_function_gradients(locally_relevant_evaluation_point, gradients);
            for (unsigned int q = 0; q < n_q_points; ++q) {
                const double coeff = 1 / std::sqrt(1 + gradients[q] * gradients[q]);
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    copy_data.vectors[0](i) -= (fe_values.shape_grad(i, q)
                                                //cell_residual(i) -= (fe_values.shape_grad(i, q) // \nabla \phi_i
                                                * coeff                    // * a_n
                                                * gradients[q]             // * u_n
                                                * fe_values.JxW(q));       // * dx
            }
            //cell->get_dof_indices(local_dof_indices);
            cell->get_dof_indices(copy_data.local_dof_indices[0]);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                residual(copy_data.local_dof_indices[0][i]) += copy_data.vectors[0](i);
                //residual(local_dof_indices[i]) += cell_residual(i);

        };
        hanging_node_constraints.condense(residual);

        auto copier = [&](const MeshWorker::CopyData<0, 1, 1> &copy_data) {
            hanging_node_constraints.distribute_local_to_global(copy_data.vectors[0],
                                                                copy_data.local_dof_indices[0],
                                                                residual);
        };

        using CellFilter= FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;


        WorkStream::run(
                CellFilter(IteratorFilters::LocallyOwnedCell(),
                           dof_handler.begin_active()),
                CellFilter(IteratorFilters::LocallyOwnedCell(),
                           dof_handler.end()),
                worker,
                copier,
                scratch,
                copy_data);
        residual.compress(VectorOperation::add);

        std::vector<bool> boundary_dofs(dof_handler.n_dofs());
        DoFTools::extract_boundary_dofs(dof_handler,
                                        ComponentMask(),
                                        boundary_dofs);
        for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
            if (boundary_dofs[i] == true)
                residual(i) = 0;
        return residual.l2_norm();

    }


    template <int dim>
    double MinimalSurfaceProblem<dim>::determine_step_length() const
    {
        return 0.1;
    }

    template <int dim>
    void MinimalSurfaceProblem<dim>::output_results(const unsigned refinement) const
    {
        TimerOutput::Scope timer_section(timer, "Output results");
        DataOut<dim>       data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(locally_relevant_solution, "solution");
        data_out.add_data_vector(newton_update, "update");
        data_out.build_patches();
        data_out.write_vtu_in_parallel("solution_" + std::to_string(refinement) + ".vtu",communicator);
    }

    template <int dim>
    void MinimalSurfaceProblem<dim>::run()
    {
        unsigned int refinement = 0;
        bool         first_step = true;
        GridGenerator::hyper_ball(triangulation);
        triangulation.refine_global(2);
        double previous_res = 0;
        while (first_step || (previous_res > 1e-3))
        {
            if (first_step == true)
            {
                std::cout << "******** Initial mesh "
                          << " ********" << std::endl;
                setup_system(true);
                set_boundary_values();
                first_step = false;
            }
            else
            {
                ++refinement;
                std::cout << "******** Refined mesh " << refinement << " ********"
                          << std::endl;
                refine_mesh();
            }
            std::cout << "  Initial residual: " << compute_residual(0) << std::endl;
            for (unsigned int inner_iteration = 0; inner_iteration < 5;
                 ++inner_iteration)
            {
                assemble_system();
                previous_res = system_rhs.l2_norm();
                solve();
                std::cout << "  Residual: " << compute_residual(0) << std::endl;
            }

           /* DataOut<dim> data_out;
            data_out.attach_dof_handler(dof_handler);
            data_out.add_data_vector(present_solution, "solution");
            data_out.add_data_vector(newton_update, "update");
            data_out.build_patches();
            const std::string filename =
                    "solution-" + Utilities::int_to_string(refinement, 2) + ".vtu";
            std::ofstream         output(filename);
            DataOutBase::VtkFlags vtk_flags;
            vtk_flags.compression_level =
                    DataOutBase::VtkFlags::ZlibCompressionLevel::best_speed;
            data_out.set_flags(vtk_flags);
            data_out.write_vtu_in_parallel(output,communicator);
            //data_out.write_vtu_in_parallel(filename,communicator);
            //data_out.write_vtu(output);// comment this line for ***PARALLEL****/
            output_results(refinement);


        }
    }
} // namespace Step15


int main(int argc, char** argv)
{
    try
    {
        dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,argv,-1);

        using namespace dealii;
        using namespace Step15;
        MinimalSurfaceProblem<2> laplace_problem_2d;
        laplace_problem_2d.run();
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    return 0;
}

