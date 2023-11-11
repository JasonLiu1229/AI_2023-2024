# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState


class node:
    def __init__(self, state, parent=None, action=None, agent=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visited = False
        self.score = 0
        self.agent = agent

    def add_child(self, child):
        self.children.append(child)

    def is_leaf(self):
        return len(self.children) == 0

    def is_visited(self):
        return self.visited

    def visit(self):
        self.visited = True

    def get_state(self):
        return self.state

    def get_parent(self):
        return self.parent

    def get_action(self):
        return self.action

    def get_children(self):
        return self.children

    def get_score(self):
        return self.score

    def set_state(self, state):
        self.state = state

    def set_parent(self, parent):
        self.parent = parent

    def set_action(self, action):
        self.action = action

    def set_children(self, children):
        self.children = children

    def set_score(self, score):
        self.score = score

    def get_agent(self):
        return self.agent

    def set_agent(self, agent):
        self.agent = agent

    def __str__(self):
        return f"Node: {self.state}"


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = 0

        # Check if current state is winning state
        if successorGameState.isWin():
            return float("inf")

        # Check if current state is losing state
        if successorGameState.isLose():
            return float("-inf")

        # Calculate Manhattan distance to all foods from perspective of pacman at the new position
        food_distances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        if len(food_distances) == 0:
            food_distances = [0]

        # Calculate Manhattan distance to all foods from perspective of pacman at the current position
        food_distances_current = [manhattanDistance(currentGameState.getPacmanPosition(), food) for food in
                                  newFood.asList()]
        if len(food_distances_current) == 0:
            food_distances_current = [0]

        # Calculate Manhattan distance to all ghosts from perspective of pacman at the new position
        ghost_distances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]

        # Calculate Manhattan distance to all ghost from perspective of pacman at the current position
        ghost_distances_current = [manhattanDistance(currentGameState.getPacmanPosition(), ghost.getPosition()) for
                                   ghost in newGhostStates]

        # Score calculation

        # If pacman is in the same position as a ghost, return -inf
        if min(ghost_distances_current) == 0:
            return float("-inf")

        # Amount of food left in the successor state
        foodLeftSuccessor = len(newFood.asList())
        # Amount of food left in the current state
        foodLeftCurrent = len(currentGameState.getFood().asList())
        # Amount of pellets left in the successor state
        pelletsLeftSuccessor = len(successorGameState.getCapsules())
        # Amount of pellets left in the current state
        pelletsLeftCurrent = len(currentGameState.getCapsules())
        # Amount of scared ghosts in the successor state
        scaredGhostsSuccessor = sum(newScaredTimes)

        score += successorGameState.getScore() - currentGameState.getScore()

        # Check if the next action is stop, if so give a penalty
        if action == Directions.STOP:
            score -= 100

        # Reward if pacman eats food or pellets
        if newPos in currentGameState.getCapsules():
            score += 100 * (pelletsLeftSuccessor - pelletsLeftCurrent)
        if foodLeftSuccessor < foodLeftCurrent:
            score += 200

        # Give penalty based on how much food are left
        score -= 10 * foodLeftSuccessor

        # reward if new position is closer to food
        if min(food_distances) < min(food_distances_current):
            score += 100
        else:
            score -= 100

        # If ghost are scared, give reward based on how close the ghosts are
        if scaredGhostsSuccessor > 0:
            if min(ghost_distances) < min(ghost_distances_current):
                score += 100
            else:
                score -= 100

        # If ghost are not scared, give penalty based on how close the ghosts are
        else:
            if min(ghost_distances) < min(ghost_distances_current):
                score -= 100
            else:
                score += 100

        return score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether the game state is a winning state

        gameState.isLose():
        Returns whether the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def build_tree(input_node, cur_depth: int, agent, numAgents):
            if cur_depth == self.depth or input_node.get_state().isWin() or input_node.get_state().isLose():
                return
            for action in input_node.get_state().getLegalActions(agent):
                child = node(input_node.get_state().generateSuccessor(agent, action), input_node, action,
                             (agent + 1) % numAgents)
                input_node.add_child(child)
                build_tree(child, cur_depth + 1 if (agent + 1) % numAgents == 0 else cur_depth, (agent + 1) % numAgents,
                           numAgents)

        def max_value(input_node: node, cur_depth: int, agent, numAgent):
            if input_node.is_leaf():
                input_node.set_score(self.evaluationFunction(input_node.get_state()))
                return input_node.get_score()
            v = float("-inf")
            for child in input_node.get_children():
                v = max(v, min_value(child, cur_depth, child.get_agent(), numAgent))
            input_node.set_score(v)
            return v

        def min_value(input_node: node, cur_depth: int, agent, numAgent):
            if input_node.is_leaf():
                input_node.set_score(self.evaluationFunction(input_node.get_state()))
                return input_node.get_score()
            v = float("inf")
            for child in input_node.get_children():
                next_agent = child.get_agent()
                if next_agent == 0:
                    v = min(v, max_value(child, cur_depth + 1, next_agent, numAgent))
                else:
                    v = min(v, min_value(child, cur_depth, next_agent, numAgent))
            input_node.set_score(v)
            return v

        def minimax(input_node: node, depth, agent, numAgent):
            if depth == 0 or input_node.get_state().isWin() or input_node.get_state().isLose():
                input_node.set_score(self.evaluationFunction(input_node.get_state()))
                return input_node.get_score()
            if agent == 0:
                return max_value(input_node, depth, agent, numAgent)
            else:
                return min_value(input_node, depth, agent, numAgent)

        number_agent = gameState.getNumAgents()

        root = node(gameState)

        build_tree(root, 0, 0, number_agent)

        best_score = float("-inf")
        best_action = None

        minimax(root, self.depth, 0, number_agent)

        for i in root.get_children():
            if i.get_score() > best_score:
                best_score = i.get_score()
                best_action = i.get_action()

        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def max_value(input_node: node, cur_depth: int, agent, numAgent, alpha, beta):
            if input_node.is_leaf():
                input_node.set_score(self.evaluationFunction(input_node.get_state()))
                return input_node.get_score()
            v = float("-inf")
            for child in input_node.get_children():
                v = max(v, min_value(child, cur_depth, child.get_agent(), numAgent, alpha, beta))
                if v > beta:
                    input_node.set_score(v)
                    return v
                alpha = max(alpha, v)
            input_node.set_score(v)
            return v

        def min_value(input_node: node, cur_depth: int, agent, numAgent, alpha, beta):
            if input_node.is_leaf():
                input_node.set_score(self.evaluationFunction(input_node.get_state()))
                return input_node.get_score()
            v = float("inf")
            for child in input_node.get_children():
                next_agent = child.get_agent()
                if next_agent == 0:
                    v = min(v, max_value(child, cur_depth + 1, next_agent, numAgent, alpha, beta))
                else:
                    v = min(v, min_value(child, cur_depth, next_agent, numAgent, alpha, beta))
                if v < alpha:
                    input_node.set_score(v)
                    return v
                beta = min(beta, v)
            input_node.set_score(v)
            return v

        def alphaBetaTree(input_node, cur_depth: int, agent, numAgents, alpha, beta):
            if cur_depth == self.depth or input_node.get_state().isWin() or input_node.get_state().isLose():
                return
            for action in input_node.get_state().getLegalActions(agent):
                child = node(input_node.get_state().generateSuccessor(agent, action), input_node, action,
                             (agent + 1) % numAgents)
                input_node.add_child(child)
                alphaBetaTree(child, cur_depth + 1 if (agent + 1) % numAgents == 0 else cur_depth,
                              (agent + 1) % numAgents, numAgents, alpha, beta)

                if agent == 0:
                    v = max_value(child.parent, cur_depth, child.parent.get_agent(), numAgents, alpha, beta)
                    if v > beta:
                        return
                    alpha = max(alpha, v)
                else:
                    v = min_value(child.parent, cur_depth, child.parent.get_agent(), numAgents, alpha, beta)
                    if v < alpha:
                        return
                    beta = min(beta, v)

        number_agent = gameState.getNumAgents()

        root = node(gameState)

        alphaBetaTree(root, 0, 0, number_agent, float("-inf"), float("inf"))

        best_score = float("-inf")
        best_action = None

        for i in root.get_children():
            if i.get_score() > best_score:
                best_score = i.get_score()
                best_action = i.get_action()

        return best_action





class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
