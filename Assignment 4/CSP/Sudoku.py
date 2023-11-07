from typing import Set, Dict

from CSP import CSP, Variable, Value


class Sudoku(CSP):
    def __init__(self, MRV=True, LCV=True):
        super().__init__(MRV=MRV, LCV=LCV)
        # TODO: Implement Sudoku::__init__ (problem 4)
        self._variables = set(Cell((x, y)) for x in range(9) for y in range(9))

    @property
    def variables(self) -> Set['Cell']:
        """ Return the set of variables in this CSP. """
        # TODO: Implement Sudoku::variables (problem 4)
        return self._variables

    def getCell(self, x: int, y: int) -> 'Cell':
        """ Get the variable corresponding to the cell on (x, y) """
        # TODO: Implement Sudoku::getCell (problem 4)
        for i in self.variables:
            if i.coords == (x, y):
                return i

    def neighbors(self, var: 'Cell') -> Set['Cell']:
        """ Return all variables related to var by some constraint. """
        # TODO: Implement Sudoku::neighbors (problem 4)
        current_x, current_y = var.coords
        neighbors = set()

        # Same column and row
        for i in range(9):
            if i != current_y:
                neighbors.add(self.getCell(current_x, i))
            if i != current_x:
                neighbors.add(self.getCell(i, current_y))

        # Same box
        box_x = current_x // 3
        box_y = current_y // 3
        for i in range(3):
            for j in range(3):
                if box_x * 3 + i != current_x and box_y * 3 + j != current_y:
                    neighbors.add(self.getCell(box_x * 3 + i, box_y * 3 + j))

        return neighbors

    def isValidPairwise(self, var1: 'Cell', val1: Value, var2: 'Cell', val2: Value) -> bool:
        """ Return whether this pairwise assignment is valid with the constraints of the csp. """
        # TODO: Implement Sudoku::isValidPairwise (problem 4)
        if val1 == val2:
            # Check if they are neighbors of each other
            if self.neighbors(var1).__contains__(var2):
                return False
            else:
                return True
        else:
            return True

    def assignmentToStr(self, assignment: Dict['Cell', Value]) -> str:
        """ Formats the assignment of variables for this CSP into a string. """
        s = ""
        for y in range(9):
            if y != 0 and y % 3 == 0:
                s += "---+---+---\n"
            for x in range(9):
                if x != 0 and x % 3 == 0:
                    s += '|'

                cell = self.getCell(x, y)
                s += str(assignment.get(cell, ' '))
            s += "\n"
        return s

    def parseAssignment(self, path: str) -> Dict['Cell', Value]:
        """ Gives an initial assignment for a Sudoku board from file. """
        initialAssignment = dict()

        with open(path, "r") as file:
            for y, line in enumerate(file.readlines()):
                if line.isspace():
                    continue
                assert y < 9, "Too many rows in sudoku"

                for x, char in enumerate(line):
                    if char.isspace():
                        continue

                    assert x < 9, "Too many columns in sudoku"

                    var = self.getCell(x, y)
                    val = int(char)

                    if val == 0:
                        continue

                    assert val > 0 and val < 10, f"Impossible value in grid"
                    initialAssignment[var] = val
        return initialAssignment


class Cell(Variable):
    def __init__(self, coords):
        super().__init__()
        # TODO: Implement Cell::__init__ (problem 4)
        # You can add parameters as well.
        self.coords = coords

    @property
    def startDomain(self) -> Set[Value]:
        """ Returns the set of initial values of this variable (not taking constraints into account). """
        # TODO: Implement Cell::startDomain (problem 4)
        return set(range(1, 10))
