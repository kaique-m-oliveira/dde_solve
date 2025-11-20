from bisect import bisect_left

import numpy as np

from .methods import *
from .tools import *

METHODS = {'CERK3': CERK3,
           'CERK4': CERK4,
           'CERK5': CERK5}


class Problem:
    def __init__(self, f, alpha, phi, t_span, Atol, Rtol, beta=False, phi_t=False, neutral=False):
        ndim, n_state_delays, n_neutral_delays, f, alpha, phi, t_span, beta, phi_t = validade_arguments(
            f, alpha, phi, t_span,  beta=beta, phi_t=phi_t)
        self.t_span = np.array(t_span)
        self.ndim, self.n_state_delays, self.n_neutral_delays = ndim, n_state_delays, n_neutral_delays
        self.f, self.alpha, self.phi, self.t_span = f, alpha, phi, t_span
        self.beta, self.phi_t = beta, phi_t
        self.y_type = np.zeros(self.ndim, dtype=float).dtype
        self.neutral = neutral
        self.Atol = np.full(ndim, Atol)
        self.Rtol = np.full(ndim, Rtol)


class Solution:
    def __init__(self, problem: Problem, discs=[], neutral=None):
        self.problem = problem
        self.t = [problem.t_span[0]]
        self.y = [np.atleast_1d(problem.phi(problem.t_span[0]))]
        self.etas = [problem.phi]
        self.etas_t = [problem.phi_t]

        self.breaking_discs = {}
        self.phi_t_breaks = {}
        if discs:
            self.validade_discs(discs)
        else:
            self.discs = [problem.t_span[0]]

        self.status = "Running"
        self.eta_calls = 0
        self.eta_t_calls = 0
        self.t_next = None
        self.neutral = neutral
        self.steps = 0
        self.fails = 0
        self.feval = 0

    @property
    def eta(self, ov=False, limit_direction=None):
        def eval(t, ov=ov, limit_direction=limit_direction):
            self.eta_calls += 1
            t = np.atleast_1d(t)  # accept scalar or array
            eps = np.finfo(float).eps
            results = np.empty((len(t), self.problem.ndim), dtype=float)
            for i in range(len(t)):
                idx = bisect_left(self.t, t[i])
                if t[i] <= self.t[0] + 2*eps:
                    if limit_direction is not None:
                        if limit_direction[i] != 0:
                            if t[i] in self.breaking_discs:
                                disc = self.breaking_discs[t[i]]
                                results[i] = disc[limit_direction[i]]
                                continue
                    results[i] = self.etas[0](t[i])
                elif t[i] <= self.t[-1] + 2*eps:
                    results[i] = self.etas[idx](t[i])
                else:
                    if ov:
                        results[i] = self.etas[-1](t[i])
                    else:
                        raise ValueError(
                            f"eta isn't defined in {t[i]}, only on {self.t[0], self.t[-1]}")
            return np.squeeze(results)
        return eval

    @property
    def eta_t(self, ov=False, limit_direction=None):
        def eval(t, ov=ov, limit_direction=limit_direction):
            self.eta_calls += 1
            t = np.atleast_1d(t)  # accept scalar or array
            eps = np.finfo(float).eps
            results = np.empty((len(t), self.problem.ndim), dtype=float)
            for i in range(len(t)):
                idx = bisect_left(self.t, t[i])
                if t[i] <= self.t[0] + 2*eps:
                    if limit_direction is not None:
                        if limit_direction[i] != 0:
                            if t[i] in self.phi_t_breaks:
                                disc = self.phi_t_breaks[t[i]]
                                results[i] = disc[limit_direction[i]]
                                continue
                            #Special consideration for the first step
                            elif t[i] == self.t[0]:
                                if limit_direction[i] == -1:
                                    results[i] = self.etas_t[0](self.t[0])
                                    continue
                                if limit_direction[i] == 1:
                                    results[i] = self.etas_t[1](self.t[0])
                                    continue
                    results[i] = self.etas_t[0](t[i])
                elif t[i] <= self.t[-1] + 2*eps:
                    results[i] = self.etas_t[idx](t[i])
                else:
                    if ov:
                        results[i] = self.etas_t[-1](t[i])
                    else:
                        raise ValueError(
                            f"eta isn't defined in {t[i]}, only on {self.t[0], self.t[-1]}")
            return np.squeeze(results)
        return eval

    def validade_discs(self, discs):
        if not isinstance(discs, (list, tuple, np.ndarray)):
            raise TypeError("discs should be a list of tuples")

        for disc in discs:
            if not isinstance(disc, (list, tuple, np.ndarray)):
                raise TypeError("discs should be a list of tuples")

            if len(disc) != 3:
                raise TypeError(
                    f" Problem with one of the discontinuities, lenght of {disc} is not 3")

            # Validating discontinuity
            if disc[0] > self.problem.t_span[0]:
                raise ValueError(
                    f"Discontinuities beyond t_span[0] are not allowed")

            # Validading left limit
            if isinstance(disc[1], numbers.Real) or np.isscalar(disc[1]):
                if 1 != self.problem.ndim:
                    raise TypeError(
                        f"Dimension of one the left limits doesn't math dimension of phi")
            elif isinstance(disc[1], (list, np.ndarray)):
                if len(disc[1]) != self.problem.ndim:
                    raise TypeError(
                        f"Dimension of one the left limits doesn't math dimension of phi")
            else:
                raise TypeError(
                    f"Unsupported type left limit:{type(disc[1])}")

            # Validading right limit
            if isinstance(disc[2], numbers.Real) or np.isscalar(disc[1]):
                if 1 != self.problem.ndim:
                    raise TypeError(
                        f"Dimension of one the right limits doesn't math dimension of phi")
            elif isinstance(disc[2], (list, np.ndarray)):
                if len(disc[2]) != self.problem.ndim:
                    raise TypeError(
                        f"Dimension of one the right limits doesn't math dimension of phi")
            else:
                raise TypeError(
                    f"Unsupported type right limit:{type(disc[1])}")

        discs.sort(key=lambda x: x[0])

        for disc in discs:
            self.breaking_discs[disc[0]] = {-1: disc[1], 1: disc[2]}

        self.discs = [x[0] for x in discs]
        if self.problem.t_span[0] not in self.discs:
            self.discs.append(self.problem.t_span[0])

    def update(self, onestep):
        success, step = onestep

        if success:
            self.t.append(step.t[0] + step.h)
            self.y.append(step.y[1])
            self.etas.append(step.new_eta[1])
            self.etas_t.append(step.new_eta_t[1])

            if step.disc:
                self.discs.append(step.disc)
                if step.breaking_step:
                    self.breaking_discs[step.disc] = {-1 : -1, 1 : 1}
                    progress = step.investigate_branches()
                    if progress == "terminated":
                        self.status = "terminated"
                        return "terminated"
                    elif progress == "one branch":
                        return "one branch"
                    elif progress == "branches":
                        self.limit_directions = step.limit_directions
                        return "branches"
        else:
            self.status = "failed"
            return "failed"

        return "success"


# Only used if there is solution branching
class SolutionList:
    def __init__(self):
        self.solutions = []

    def append_solution(self, sol):
        self.solutions.append(sol)


def recursive_integration(method, solution, solutionList):
    for limit_direction in solution.limit_directions:
        solution_copy = deepcopy(solution)
        status, solution_copy = integrate_branch(method, solution_copy, limit_direction)
        if status == "branches":
            recursive_integration(method, solution_copy, solutionList)
        else:
            solutionList.append_solution(solution_copy)
            print('solutionList', solutionList.solutions)


def integrate_branch(method, solution, limit_direction):
    t, tf = solution.t[-1], solution.problem.t_span[-1]
    problem = solution.problem
    neutral = solution.neutral
    h = (1e-7 ** (1 / 4)) * 0.1  # Initial stepsize

    onestep = method(problem, solution, h, neutral)
    onestep.eta = lambda t: solution.eta(
        t, limit_direction=limit_direction)

    status = solution.update(onestep.one_step_CRK())

    calls = 0
    while t < tf:
        h = min(h, tf - t)
        if status == "success":
            onestep = method(problem, solution, h, neutral)

        elif status == "one branch":
            limit_direction = onestep.limit_direction
            onestep = method(problem, solution, h, neutral)
            onestep.eta = lambda t: solution.eta(
                t, limit_direction=limit_direction)

        elif status == "branches":
            return "branches", solution

        elif status == "terminated" or status == "failed":
            raise ValueError(f"solution failed duo to {status} at t = {solution.t[-1]}")

        status = solution.update(onestep.one_step_CRK())
        calls += onestep.number_of_calls
        h = onestep.h_next
        t = solution.t[-1]
    return "Success", solution


def solve_dde(t_span, f, alpha, phi, method='CERK5', Atol = 1e-7, Rtol = 1e-7, discs=[]):
    problem = Problem(f, alpha, phi, t_span, Atol, Rtol)
    solution = Solution(problem, discs=discs)
    t, tf = problem.t_span


    if method in METHODS:
        method = METHODS[method]
    else:
        raise ValueError(f'there is no method {method}')

    order = method.order["discrete_method"]
    h = get_initial_step(problem, solution, Atol, Rtol, order)
    onestep = method(problem, solution, h)
    onestep.first_step = True

    branch_status = onestep.first_step_investigate_branch()

    if branch_status == "one branch":
        limit_direction = onestep.limit_direction
        onestep.eta = lambda t: solution.eta(
            t, limit_direction=limit_direction)
    elif branch_status == "branches":
        solutionList = SolutionList()
        solution.limit_directions = onestep.limit_directions
        recursive_integration(method, solution, solutionList)
        return solutionList
    elif branch_status == "terminated" or branch_status == "failed":
        print('steps', solution.steps)
        print('fails', solution.fails)
        print('feval', solution.feval)
        print(f"solution failed duo to {branch_status} at t = {solution.t[-1]}")
        return solution
        # raise ValueError(f"solution failed duo to {branch_status}")

    status = solution.update(onestep.one_step_CRK())

    h = onestep.h_next
    t = solution.t[-1]

    times = []
    calls = 0
    while t < tf:
        if h is not None:
            h = min(h, tf - t)
        if status == "success":
            onestep = method(problem, solution, h)
        elif status == "one branch":
            limit_direction = onestep.limit_direction
            onestep = method(problem, solution, h)
            onestep.eta = lambda t: solution.eta(
                t, limit_direction=limit_direction)
        elif status == "branches":
            solutionList = SolutionList()
            recursive_integration(solution, solutionList)
            return solutionList
        elif status == "terminated" or status == "failed":
            print('steps', solution.steps)
            print('fails', solution.fails)
            print('feval', solution.feval)
            print(f"solution failed duo to {status} at t = {solution.t[-1]}")
            return solution

        status = solution.update(onestep.one_step_CRK())
        calls += onestep.number_of_calls
        h = onestep.h_next
        t = solution.t[-1]

    solution.status = "success"
    return solution


def solve_ndde(t_span, f, alpha, beta, phi, phi_t, method='CERK5', discs=[], Atol=1e-7, Rtol=1e-7):
    problem = Problem(f, alpha, phi, t_span, Atol = Atol, Rtol = Rtol, beta = beta, phi_t = phi_t, neutral=True)
    solution = Solution(problem, discs=discs, neutral=True)
    t, tf = problem.t_span
    default_investigation = discs

    if method in METHODS:
        method = METHODS[method]
    else:
        raise ValueError(f'there is no method {method}')
    

    order = method.order["discrete_method"]
    h = get_initial_step(problem, solution, Atol, Rtol, order, neutral = True)

    onestep = method(problem, solution, h, neutral=True)
    onestep.first_step = True

    #Not doing default termination investigation 
    if default_investigation:
        branch_status = onestep.first_step_investigate_branch()

        if branch_status == "one branch":
            limit_direction = onestep.limit_direction
            onestep.eta = lambda t: solution.eta(
                t, limit_direction=limit_direction)
        elif branch_status == "branches":
            solutionList = SolutionList()
            solution.limit_directions = onestep.limit_directions
            recursive_integration(method, solution, solutionList)
            return solutionList
        elif branch_status == "terminated" or branch_status == "failed":
            raise ValueError(f"solution failed duo to {branch_status}")

    status = solution.update(onestep.one_step_CRK())

    eta_t_left = solution.eta_t(t_span[0], limit_direction=[-1])
    eta_t_right = solution.eta_t(t_span[0], limit_direction=[1])
    solution.breaking_discs[t_span[0]] = {-1: eta_t_left, 1: eta_t_right}

    h = onestep.h_next
    t = solution.t[-1]

    times = []
    calls = 0
    while t < tf:

        if h is not None:
            h = min(h, tf - t)

        if status == "success":
            onestep = method(problem, solution, h, neutral=True)
        elif status == "one branch":
            limit_direction = onestep.limit_direction
            onestep = method(problem, solution, h, neutral=True)
            onestep.eta = lambda t: solution.eta(
                t, limit_direction=limit_direction)
        elif status == "branches":
            solutionList = SolutionList()
            input('here')
            recursive_integration(method, solution, solutionList)
            return solutionList
        elif status == "terminated" or status == "failed":
            print(f'terminated at {t} for {status}')
            print('steps', solution.steps)
            print('fails', solution.fails)
            print('feval', solution.feval)
            return solution

        status = solution.update(onestep.one_step_CRK())
        calls += onestep.number_of_calls
        h = onestep.h_next
        t = solution.t[-1]

    return solution
