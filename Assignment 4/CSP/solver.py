""" Command line interface to call the CSP solver. """
from enum import Enum

from typer import Typer
from tqdm import tqdm

from Sudoku import Sudoku
from NQueens import NQueens


# IMPORTANT: Do not edit this file!


class Method(str, Enum):
    bf = "bf"
    fc = "fc"
    ac3 = "ac3"


app = Typer()


def solve(csp, method: Method, initialAssignment=dict()):
    if method == Method.bf:
        print("Solving with brute force")
        assignment = csp.solveBruteForce(initialAssignment)
    elif method == Method.fc:
        print("Solving with forward checking")
        assignment = csp.solveForwardChecking(initialAssignment)
    elif method == Method.ac3:
        print("Solving with ac3")
        assignment = csp.solveAC3(initialAssignment)
    else:
        raise RuntimeError(f"Method '{method}' not found.")

    if assignment:
        s = csp.assignmentToStr(assignment)
        tqdm.write("\nSolution:")
        tqdm.write(s)
    else:
        tqdm.write("No solution found")


@app.command()
def sudoku(path: str, method: Method = Method.bf, MRV: bool = True, LCV: bool = True):
    """ Solve Sudoku as a CSP. """
    if method == Method.bf:
        MRV = False
        LCV = False
    csp = Sudoku(MRV=MRV, LCV=LCV)
    initialAssignment = csp.parseAssignment(path)
    solve(csp, method, initialAssignment)


@app.command()
def queens(n: int = 5, method: Method = Method.bf, MRV: bool = True, LCV: bool = True):
    """ Solve the N Queens problem as a CSP. """
    if method == Method.bf:
        MRV = False
        LCV = False
    csp = NQueens(n=n, MRV=MRV, LCV=LCV)
    solve(csp, method)


@app.command()
def multi_queens(n: int = 10, it: int = 10, mean: bool = False, stdev: bool = False, method: Method = Method.bf,
                 MRV: bool = True, LCV: bool = True):
    """ Solve the N Queens problem as a CSP. Multiple iterations."""
    import os
    import json
    import numpy as np

    with open("log.json", "w") as file:
        json.dump({}, file)
        file.close()

    for i in range(it):
        os.system('python solver.py queens --n ' + str(
            n) + f" {'--no-mrv' if not MRV else ''}" + f" {'--no-lcv' if not LCV else ''}" + f" {'' if method == Method.bf else '--method ' + method.value}")

    values = json.load(open("log.json", "r")).values()
    calc = np.mean(list(values)), np.std(list(values))

    print(f"Mean: {calc[0]}, Standard deviation: {calc[1]}")


if __name__ == "__main__":
    app()
