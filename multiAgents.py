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


import random

import util
from game import Agent
from pacman import GameState
from util import manhattanDistance


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
        # Collect legal moves and child states
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

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        nowPos = currentGameState.getPacmanPosition()
        nowFood = currentGameState.getFood()

        if nowPos == newPos:
            return -1

        for ghostState in newGhostStates:
            distance = manhattanDistance(newPos, ghostState.getPosition())
            scared = ghostState.scaredTimer > 0
            if distance < 2 and not scared:
                return -100

        distance = float("inf")
        for food in nowFood.asList():
            distance = min(distance, manhattanDistance(newPos, food))

        return 1 / (distance + 1)


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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
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

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.max_state(gameState, 0)[1]

    def max_state(self, gameState: GameState, depth: int):
        actions = gameState.getLegalActions(0)
        all_actions = []

        if len(actions) == 0:
            return self.evaluationFunction(gameState), None

        for action in actions:
            next_state = gameState.getNextState(0, action)
            if next_state.isWin() or next_state.isLose():
                all_actions.append((self.evaluationFunction(next_state), action))
            else:
                all_actions.append((self.min_state(next_state, depth, 1), action))

        return max(all_actions, key=lambda x: x[0])

    def min_state(self, gameState: GameState, depth: int, agentIndex: int):
        actions = gameState.getLegalActions(agentIndex)
        min_score = float("inf")
        if len(actions) == 0:
            return self.evaluationFunction(gameState)
        for action in actions:
            next_state = gameState.getNextState(agentIndex, action)
            if agentIndex == gameState.getNumAgents() - 1:
                if depth == self.depth - 1 or len(next_state.getLegalActions(0)) == 0:
                    min_score = min(min_score, self.evaluationFunction(next_state))
                else:
                    min_score = min(min_score, self.max_state(next_state, depth + 1)[0])
            else:
                min_score = min(min_score, self.min_state(next_state, depth, agentIndex + 1))
        return min_score


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.max_state(gameState, 0, float("-inf"), float("inf"))[1]

    def max_state(self, gameState: GameState, depth: int, alpha: float, beta: float):
        actions = gameState.getLegalActions(0)

        v = float("-inf")
        selected = None

        for action in actions:
            next_state = gameState.getNextState(0, action)
            if next_state.isWin() or next_state.isLose():
                score = self.evaluationFunction(next_state)
            else:
                score = self.min_state(next_state, depth, 1, alpha, beta)

            if score >= beta:
                return score, action

            if score > alpha:
                alpha = score
                selected = action

        return alpha, selected

    def min_state(self, gameState: GameState, depth: int, agentIndex: int, alpha: float, beta: float):
        actions = gameState.getLegalActions(agentIndex)
        if len(actions) == 0:
            return self.evaluationFunction(gameState)

        for action in actions:
            next_state = gameState.getNextState(agentIndex, action)
            if agentIndex == gameState.getNumAgents() - 1:
                if depth == self.depth - 1 or len(next_state.getLegalActions(0)) == 0:
                    score = self.evaluationFunction(next_state)
                else:
                    score, _ = self.max_state(next_state, depth + 1, alpha, beta)
            else:
                score = self.min_state(next_state, depth, agentIndex + 1, alpha, beta)

            if score <= alpha:
                return score

            beta = min(beta, score)

        return beta


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
        return self.max_state(gameState, 0)[1]

    def max_state(self, gameState: GameState, depth: int):
        actions = gameState.getLegalActions(0)
        all_actions = []

        if len(actions) == 0:
            return self.evaluationFunction(gameState), None

        for action in actions:
            next_state = gameState.getNextState(0, action)
            if next_state.isWin() or next_state.isLose():
                all_actions.append((self.evaluationFunction(next_state), action))
            else:
                all_actions.append((self.min_state(next_state, depth, 1), action))

        return max(all_actions, key=lambda x: x[0])

    def min_state(self, gameState: GameState, depth: int, agentIndex: int):
        actions = gameState.getLegalActions(agentIndex)
        scores = []
        if len(actions) == 0:
            return self.evaluationFunction(gameState)
        for action in actions:
            next_state = gameState.getNextState(agentIndex, action)
            if agentIndex == gameState.getNumAgents() - 1:
                if depth == self.depth - 1 or len(next_state.getLegalActions(0)) == 0:
                    scores.append(self.evaluationFunction(next_state))
                else:
                    scores.append(self.max_state(next_state, depth + 1)[0])
            else:
                scores.append(self.min_state(next_state, depth, agentIndex + 1))
        return sum(scores) / len(scores)


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
