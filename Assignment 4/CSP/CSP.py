import random
import copy

from typing import Set, Dict, List, TypeVar, Optional
from abc import ABC, abstractmethod

from util import monitor

Value = TypeVar('Value')


class Variable(ABC):
    @property
    @abstractmethod
    def startDomain(self) -> Set[Value]:
        """ Returns the set of initial values of this variable (not taking constraints into account). """
        pass

    def __copy__(self):
        return self

    def __deepcopy__(self, memodict={}):
        return self


class CSP(ABC):
    def __init__(self, MRV=True, LCV=True):
        self.MRV = MRV
        self.LCV = LCV

    @property
    @abstractmethod
    def variables(self) -> Set[Variable]:
        """ Return the set of variables in this CSP.
            Abstract method to be implemented for specific instances of CSP problems.
        """
        pass

    def remainingVariables(self, assignment: Dict[Variable, Value]) -> Set[Variable]:
        """ Returns the variables not yet assigned. """
        return self.variables.difference(assignment.keys())

    @abstractmethod
    def neighbors(self, var: Variable) -> Set[Variable]:
        """ Return all variables related to var by some constraint.
            Abstract method to be implemented for specific instances of CSP problems.
        """
        pass

    def assignmentToStr(self, assignment: Dict[Variable, Value]) -> str:
        """ Formats the assignment of variables for this CSP into a string. """
        s = ""
        for var, val in assignment.items():
            s += f"{var} = {val}\n"
        return s

    def isComplete(self, assignment: Dict[Variable, Value]) -> bool:
        """ Return whether the assignment covers all variables.
            :param assignment: dict (Variable -> value)
        """
        return self.remainingVariables(assignment) == set()

    @abstractmethod
    def isValidPairwise(self, var1: Variable, val1: Value, var2: Variable, val2: Value) -> bool:
        """ Return whether this pairwise assignment is valid with the constraints of the csp.
            Abstract method to be implemented for specific instances of CSP problems.
        """
        pass

    def isValid(self, assignment: Dict[Variable, Value]) -> bool:
        """ Return whether the assignment is valid (i.e. is not in conflict with any constraints).
            You only need to take binary constraints into account.
            Hint: use `CSP::neighbors` and `CSP::isValidPairwise` to check that all binary constraints are satisfied.
            Note that constraints are symmetrical, so you don't need to check them in both directions.
        """
        checked_paires = []
        for i in assignment:
            for j in self.neighbors(i):
                if (i, j) in checked_paires or (j, i) in checked_paires:
                    continue
                if j in assignment:
                    if not self.isValidPairwise(i, assignment[i], j, assignment[j]):
                        return False
        return True

    def solveBruteForce(self, initialAssignment: Dict[Variable, Value] = dict()) -> Optional[Dict[Variable, Value]]:
        """ Called to solve this CSP with brute force technique.
            Initializes the domains and calls `CSP::_solveBruteForce`. """
        domains = domainsFromAssignment(initialAssignment, self.variables)
        return self._solveBruteForce(initialAssignment, domains)

    @monitor
    def _solveBruteForce(self, assignment: Dict[Variable, Value], domains: Dict[Variable, Set[Value]]) -> Optional[
        Dict[Variable, Value]]:
        """ Implement the actual backtracking algorithm to brute force this CSP.
            Use `CSP::isComplete`, `CSP::isValid`, `CSP::selectVariable` and `CSP::orderDomain`.
            :return: a complete and valid assignment if one exists, None otherwise.
        """
        if self.isComplete(assignment):
            return assignment
        var = self.selectVariable(assignment, domains)
        for val in self.orderDomain(assignment, domains, var):
            assignment[var] = val
            if self.isValid(assignment):
                result = self._solveBruteForce(assignment, domains)
                if result is not None:
                    return result
            assignment.pop(var)
        return None

    def solveForwardChecking(self, initialAssignment: Dict[Variable, Value] = dict()) -> Optional[
        Dict[Variable, Value]]:
        """ Called to solve this CSP with forward checking.
            Initializes the domains and calls `CSP::_solveForwardChecking`. """
        domains = domainsFromAssignment(initialAssignment, self.variables)
        for var in set(initialAssignment.keys()):
            domains = self.forwardChecking(initialAssignment, domains, var)
        return self._solveForwardChecking(initialAssignment, domains)

    @monitor
    def _solveForwardChecking(self, assignment: Dict[Variable, Value], domains: Dict[Variable, Set[Value]]) -> Optional[
        Dict[Variable, Value]]:
        """ Implement the actual backtracking algorithm with forward checking.
            Use `CSP::forwardChecking` and you should no longer need to check if an assignment is valid.
            :return: a complete and valid assignment if one exists, None otherwise.
        """
        for i in domains.values():
            if len(i) == 0:
                return None

        if self.isComplete(assignment):
            return assignment

        var = self.selectVariable(assignment, domains)

        for val in self.orderDomain(assignment, domains, var):
            assignment[var] = val
            temp_domains = copy.deepcopy(domains)
            domains = self.forwardChecking(assignment, domains, var)
            result = self._solveForwardChecking(assignment, domains)
            if result is not None:
                return result
            domains = temp_domains
            assignment.pop(var)
        return None

    def forwardChecking(self, assignment: Dict[Variable, Value], domains: Dict[Variable, Set[Value]],
                        variable: Variable) -> Dict[Variable, Set[Value]]:
        """ Implement the forward checking algorithm from the theory lectures.

        :param domains: current domains.
        :param assignment: current assignment.
        :param variable: The variable that was just assigned (only need to check changes).
        :return: the new domains after enforcing all constraints.
        """
        value = assignment[variable]
        for i in self.neighbors(variable):
            for j in copy.deepcopy(domains[i]):
                if not self.isValidPairwise(variable, value, i, j):
                    domains[i].remove(j)
                    if len(domains[i]) == 0:
                        return domains
        return domains

    def selectVariable(self, assignment: Dict[Variable, Value], domains: Dict[Variable, Set[Value]]) -> Variable:
        """ Implement a strategy to select the next variable to assign. """
        if not self.MRV:
            return random.choice(list(self.remainingVariables(assignment)))

        # Selection of variable with minimum remaining values,
        # if multiple variables have the same amount of remaining values, select the one with the most constraints

        min_remaining_values = float('inf')
        min_remaining_values_var = None
        for i in self.remainingVariables(assignment):
            if len(domains[i]) < min_remaining_values:
                min_remaining_values = len(domains[i])
                min_remaining_values_var = i
            elif len(domains[i]) == min_remaining_values:
                if len(self.neighbors(i)) > len(self.neighbors(min_remaining_values_var)):
                    min_remaining_values_var = i
        return min_remaining_values_var

    def orderDomain(self, assignment: Dict[Variable, Value], domains: Dict[Variable, Set[Value]], var: Variable) -> \
            List[Value]:
        """ Implement a smart ordering of the domain values. """
        if not self.LCV:
            return list(domains[var])

        # Order domain values by least constraining value heuristic
        # (i.e. the value that rules out the fewest values in the remaining variables)

        least_constraining_values = []

        for i in domains[var]:
            count = 0
            for j in self.neighbors(var):
                if i in domains[j]:
                    count += 1
            least_constraining_values.append((i, count))
        least_constraining_values.sort(key=lambda x: x[1])
        return [i[0] for i in least_constraining_values]

    def solveAC3(self, initialAssignment: Dict[Variable, Value] = dict()) -> Optional[Dict[Variable, Value]]:
        """ Called to solve this CSP with AC3.
            Initializes domains and calls `CSP::_solveAC3`. """
        domains = domainsFromAssignment(initialAssignment, self.variables)
        for var in set(initialAssignment.keys()):
            domains = self.ac3(initialAssignment, domains, var)
        return self._solveAC3(initialAssignment, domains)

    @monitor
    def _solveAC3(self, assignment: Dict[Variable, Value], domains: Dict[Variable, Set[Value]]) -> Optional[
        Dict[Variable, Value]]:
        """
            Implement the actual backtracking algorithm with AC3.
            Use `CSP::ac3`.
            :return: a complete and valid assignment if one exists, None otherwise.
        """
        for i in domains.values():
            if len(i) == 0:
                return None

        if self.isComplete(assignment):
            return assignment

        var = self.selectVariable(assignment, domains)

        for val in self.orderDomain(assignment, domains, var):
            assignment[var] = val
            temp_domains = copy.deepcopy(domains)
            domains = self.ac3(assignment, domains, var)
            result = self._solveAC3(assignment, domains)
            if result is not None:
                return result
            domains = temp_domains
            assignment.pop(var)
        return None

    def ac3(self, assignment: Dict[Variable, Value], domains: Dict[Variable, Set[Value]], variable: Variable) -> Dict[
        Variable, Set[Value]]:
        """ Implement the AC3 algorithm from the theory lectures.

        :param domains: current domains.
        :param assignment: current assignment.
        :param variable: The variable that was just assigned (only need to check changes).
        :return: the new domains ensuring arc consistency.
        """
        queue = []
        for i in assignment.keys():
            domains[i] = {assignment[i]}

        for i in self.neighbors(variable):
            queue.append((variable, i))
            queue.append((i, variable))

        while len(queue) != 0:
            current = queue.pop(0)
            temp_domains = self.removeInconsistentValues(current[0], current[1], domains)
            if temp_domains is not None:
                for i in self.neighbors(current[0]):
                    if (i, current[0]) not in queue or (current[0], i) not in queue:
                        queue.insert(0, (i, current[0]))
                domains = temp_domains
        return domains

    def removeInconsistentValues(self, X, Y, domains):
        temp = copy.deepcopy(domains[X])
        removed = False
        for x in temp:
            if not any([self.isValidPairwise(X, x, Y, y) for y in domains[Y]]):
                domains[X].remove(x)
                removed = True
        if removed:
            return domains
        else:
            return None


def domainsFromAssignment(assignment: Dict[Variable, Value], variables: Set[Variable]) -> Dict[Variable, Set[Value]]:
    """ Fills in the initial domains for each variable.
        Already assigned variables only contain the given value in their domain.
    """
    domains = {v: v.startDomain for v in variables}
    for var, val in assignment.items():
        domains[var] = {val}
    return domains
