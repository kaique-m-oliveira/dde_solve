"""
DDE_solve - A python package for solving Delay Differential Equations (DDEs) 
and Neutral Delay Differential Equations (NDDEs) using Continuous Runge-Kutta methods.
"""

from .integrator import Problem, Solution, solve_dde, solve_ndde
from .methods import CERK3, CERK4, CERK5

__version__ = "0.1.2"
__all__ = [
        'solve_dde', 
        'solve_ndde', 
        'Problem',
        'Solution', 
        'CERK3',
        'CERK4',
        'CERK5'
        ]
