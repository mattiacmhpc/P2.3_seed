/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2016 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Wolfgang Bangerth, 1999,
 *          Guido Kanschat, 2011
 */

#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/meshworker/copy_data.h>
#include <deal.II/meshworker/scratch_data.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/conditional_ostream.h>

#include <fstream>
#include <iostream>

using namespace dealii;
namespace LA {
    //using namespace dealii::LinearAlgebraPETSc;
    using namespace dealii::LinearAlgebraTrilinos;
}


template <int dim>
class Step3
{
public:
  Step3();

  void
  run(const unsigned int n_cycles           = 1,
      const unsigned int initial_refinement = 3);


private:
  void
  make_grid(const unsigned int ref_level);
  void
  estimate_error();
  void
  mark_cells_for_refinement();
  void
  refine_grid();
  void
  setup_system();
  void
  assemble_system();
  void
  solve();
  void
  compute_error();
  void
  output_results(const unsigned int cycle) const;

  MPI_Comm communicator;

  ConditionalOStream pout;

  mutable TimerOutput timer;

  parallel::distributed::Triangulation<dim> triangulation;
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;

  IndexSet locally_relevent_dofs;
  IndexSet locally_owned_dofs;

  AffineConstraints<double> constraints;

  //LA::MPI::SparsityPattern      sparsity_pattern;
  LA::MPI::SparseMatrix system_matrix;

  LA::MPI::Vector solution;
  LA::MPI::Vector system_rhs;

  LA::MPI::Vector locally_relevant_solution;

  Vector<float> error_estimator;

  Vector<double> L2_error_per_cell;
  Vector<double> H1_error_per_cell;

  /** Exact solution (used to manufacture a rhs). */
  FunctionParser<dim> exact_solution;

  /** Manufactured right hand side. */
  FunctionParser<dim> rhs_function;

  /** Utility to compute error tables. */
  ParsedConvergenceTable error_table;
};

template <int dim>
Step3<dim>::Step3()
  : communicator(MPI_COMM_WORLD)
  ,pout(std::cout, Utilities::MPI::this_mpi_process(communicator)==0)//only the first proc can print output
  ,timer(pout, TimerOutput::summary, TimerOutput::cpu_and_wall_times)
  , triangulation(communicator)
  , fe(1)
  , dof_handler(triangulation)
  , exact_solution("exp(x)*exp(y)")
  , rhs_function("-2*exp(x)*exp(y)")
  , error_table({"u"}, {{VectorTools::H1_norm, VectorTools::L2_norm}})
{}


template <int dim>
void
Step3<dim>::make_grid(const unsigned int ref_level)
{
  TimerOutput::Scope timer_section(timer, "Make grid");
  triangulation.clear();
  GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global(ref_level);

  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl;
}

template <int dim>
void
Step3<dim>::estimate_error()
{
  // We fill error_estimator with an indicator on when the grid needs refinement
  error_estimator.reinit(triangulation.n_active_cells());
  QGauss<dim - 1> face_quadrature(fe.degree + 1);

  KellyErrorEstimator<dim>::estimate(
    dof_handler, face_quadrature, {}, locally_relevant_solution, error_estimator);
}

template <int dim>
void
Step3<dim>::mark_cells_for_refinement()
{
  parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
                                                    error_estimator,
                                                    0.33,
                                                    0.0);
}



template <int dim>
void
Step3<dim>::refine_grid()
{
  TimerOutput::Scope timer_section(timer, "Refine grid");
  triangulation.execute_coarsening_and_refinement();
  // triangulation.refine_global(1);
}


template <int dim>
void
Step3<dim>::setup_system()
{
  TimerOutput::Scope timer_section(timer, "Setup dofs");
  dof_handler.distribute_dofs(fe);
  locally_owned_dofs = dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevent_dofs);
  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);

  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           exact_solution,constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);

  //sparsity_pattern.copy_from(dsp);

              system_matrix.reinit(locally_owned_dofs,dsp,communicator);

  solution.reinit(locally_owned_dofs, communicator);
  system_rhs.reinit(locally_owned_dofs, communicator);

  locally_relevant_solution.reinit(locally_owned_dofs,
          locally_relevent_dofs,communicator);

}


template <int dim>
void
Step3<dim>::assemble_system() {
    TimerOutput::Scope timer_section(timer, "Assemble system");
    QGauss<dim> quadrature_formula(2);
    MeshWorker::ScratchData<dim> scratch(fe,
            //FEValues<dim>      fe_values(fe,
                                         quadrature_formula,
                                         update_quadrature_points | update_values |
                                         update_gradients | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();

    MeshWorker::CopyData<1, 1, 1> copy_data(dofs_per_cell);
    //FullMatrix<double> cell_matrix(dofs_per_cell)
    //std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    auto worker = [&](
            const decltype(dof_handler.begin_active()) &cell,
            MeshWorker::ScratchData<dim> &scratch,
            MeshWorker::CopyData<1, 1, 1> &copy_data) {

        auto &fe_values = scratch.reinit(cell);

        copy_data.matrices[0] = 0;
        copy_data.vectors[0] = 0;

        const auto &q_points = fe_values.get_quadrature_points();


        for (unsigned int q_index = 0; q_index < n_q_points; ++q_index) {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    copy_data.matrices[0](i, j) +=
                            (fe_values.shape_grad(i, q_index) *
                             fe_values.shape_grad(j, q_index) * fe_values.JxW(q_index));

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                copy_data.vectors[0](i) += (fe_values.shape_value(i, q_index) *
                                            rhs_function.value(q_points[q_index]) * fe_values.JxW(q_index));
        }
        cell->get_dof_indices(copy_data.local_dof_indices[0]);
    };
        auto copier = [&](const MeshWorker::CopyData<1, 1, 1> &copy_data) {
            constraints.distribute_local_to_global(copy_data.matrices[0],
                                                   copy_data.vectors[0],
                                                   copy_data.local_dof_indices[0],
                                                   system_matrix,
                                                   system_rhs);
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

        //
        system_matrix.compress(VectorOperation::add);
        system_rhs.compress(VectorOperation::add);
    }


template <int dim>
void
Step3<dim>::solve()
{
  TimerOutput::Scope timer_section(timer, "Solve system");
  SolverControl      solver_control(1000, 1e-12, false, false);
  LA::SolverCG         solver(solver_control);
  //Needed by PETSc preconditioner
  LA::MPI::PreconditionAMG::AdditionalData data;

  LA::MPI::PreconditionAMG amg;
  amg.initialize(system_matrix);
  solver.solve(system_matrix, solution, system_rhs, amg);
  constraints.distribute(solution);
  locally_relevant_solution= solution;
}



template <int dim>
void
Step3<dim>::compute_error()
{
  TimerOutput::Scope timer_section(timer, "Compute error");
  L2_error_per_cell.reinit(triangulation.n_active_cells());
  H1_error_per_cell.reinit(triangulation.n_active_cells());
  QGauss<dim> error_quadrature(2 * fe.degree + 1);

  VectorTools::integrate_difference(dof_handler,
                                    locally_relevant_solution,
                                    exact_solution,
                                    L2_error_per_cell,
                                    error_quadrature,
                                    VectorTools::L2_norm);


  VectorTools::integrate_difference(dof_handler,
                                    locally_relevant_solution,
                                    exact_solution,
                                    H1_error_per_cell,
                                    error_quadrature,
                                    VectorTools::H1_norm);

  std::cout << "L2 norm of error: " << L2_error_per_cell.l2_norm() << std::endl;
  std::cout << "H1 norm of error: " << H1_error_per_cell.l2_norm() << std::endl;
}


template <int dim>
void
Step3<dim>::output_results(const unsigned int cycle) const
{
  TimerOutput::Scope timer_section(timer, "Output results");
  DataOut<dim>       data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(locally_relevant_solution, "solution");
  data_out.add_data_vector(L2_error_per_cell, "L2_error");
  data_out.add_data_vector(H1_error_per_cell, "H1_error");
  data_out.add_data_vector(error_estimator, "Error_estimator");
  data_out.build_patches();

  //std::ofstream output("solution_" + std::to_string(cycle) + ".vtu");
  data_out.write_vtu_in_parallel("solution_" + std::to_string(cycle) + ".vtu",communicator);
}


template <int dim>
void
Step3<dim>::run(const unsigned int n_cycles,
                const unsigned int initial_refinement)
{
  make_grid(initial_refinement);
  for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
    {
      std::cout << "Cycle " << cycle << std::endl;
      setup_system();
      assemble_system();
      solve();

      // Compute the actual error from the exact solution
      compute_error();
      // Compute an estimate of the error using Kelly error estimator
      estimate_error();
      error_table.error_from_exact(dof_handler, locally_relevant_solution, exact_solution);
      output_results(cycle);

      if (cycle != n_cycles - 1)
        {
          // Mark and refine
          mark_cells_for_refinement();
          refine_grid();
        }
    }

  if(Utilities::MPI::this_mpi_process(communicator)==0)
  error_table.output_table(std::cout);
}



int
main(int argc, char** argv)
{
   Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,argv,-1);//-1 is the total n of threads
  deallog.depth_console(2);

  Step3<2> laplace_problem;
  laplace_problem.run(15);

  return 0;
}
