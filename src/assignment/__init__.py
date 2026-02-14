"""
Module 2: Task Assignment via Constraint Programming.

Provides a CP-SAT-based optimizer that replaces the greedy nearest-AGV
dispatcher from Module 1. Falls back to greedy if OR-Tools is unavailable.

Quick start:
    from src.assignment.solver import CPSATAssignmentSolver
    solver = CPSATAssignmentSolver(pick_station_resources)
    assignments = solver.solve(pending_tasks, idle_agvs, warehouse)
"""

from src.assignment.solver import (
    CPSATAssignmentSolver,
    SolverConfig,
    SolverStatus,
    AssignmentResult,
)
from src.assignment.cost_matrix import CostMatrix, compute_cost_matrix

__all__ = [
    "CPSATAssignmentSolver",
    "SolverConfig",
    "SolverStatus",
    "AssignmentResult",
    "CostMatrix",
    "compute_cost_matrix",
]
