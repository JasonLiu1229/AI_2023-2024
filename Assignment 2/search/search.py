# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem, succesor=None):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    from util import Stack
    visited = Stack()
    not_visited = Stack()
    path_Stack = Stack()
    not_visited.push(problem.getStartState())
    path_Stack.push([])

    while not not_visited.isEmpty():
        current = not_visited.pop()
        path = path_Stack.pop()
        if current in visited.list:
            continue
        visited.push(current)

        if problem.isGoalState(current):
            return path

        succesors = []
        succesors = problem.getSuccessors(current)
        for i in succesors:
            if i[0] not in visited.list:
                new_path = path + [i[1]]
                not_visited.push(i[0])
                path_Stack.push(new_path)
    return []


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue
    visited = Queue()
    not_visited = Queue()
    path_queue = Queue()
    not_visited.push(problem.getStartState())
    path_queue.push([])

    while not not_visited.isEmpty():
        current = not_visited.pop()
        path = path_queue.pop()
        if current in visited.list:
            continue
        visited.push(current)

        if problem.isGoalState(current):
            return path

        succesors = problem.getSuccessors(current)
        for i in succesors:
            if i[0] not in visited.list:
                new_path = path + [i[1]]
                not_visited.push(i[0])
                path_queue.push(new_path)
    return []


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue, Queue
    visited = Queue()
    not_visited = PriorityQueue()
    path_priorityQueue = PriorityQueue()
    g_valueQueue = PriorityQueue()
    not_visited.push(problem.getStartState(), 0)
    path_priorityQueue.push([], 0)
    g_valueQueue.push(0, 0)

    while not not_visited.isEmpty():
        current = not_visited.pop()
        path = path_priorityQueue.pop()
        cost = g_valueQueue.pop()
        if current in visited.list:
            continue
        visited.push(current)

        if problem.isGoalState(current):
            return path

        succesors = []
        succesors = problem.getSuccessors(current)
        for i in succesors:
            if i[0] not in visited.list:
                new_path = path + [i[1]]
                new_cost = cost + i[2]
                not_visited.push(i[0], new_cost)
                path_priorityQueue.push(new_path, new_cost)
                g_valueQueue.push(new_cost, new_cost)
    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue, Queue
    visited = Queue()
    not_visited = PriorityQueue()
    path_priorityQueue = PriorityQueue()
    g_valueQueue = PriorityQueue()
    not_visited.push(problem.getStartState(), 0)
    path_priorityQueue.push([], 0)
    g_valueQueue.push(0, 0)

    while not not_visited.isEmpty():
        current = not_visited.pop()
        path = path_priorityQueue.pop()
        cost = g_valueQueue.pop()
        if current in visited.list:
            continue
        visited.push(current)

        if problem.isGoalState(current):
            return path

        succesors = []
        succesors = problem.getSuccessors(current)
        for i in succesors:
            if i[0] not in visited.list:
                new_path = path + [i[1]]
                g_value = cost + i[2]
                h_value = heuristic(i[0], problem)
                f_value = h_value + g_value
                not_visited.push(i[0], f_value)
                path_priorityQueue.push(new_path, f_value)
                g_valueQueue.push(g_value, f_value)
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
