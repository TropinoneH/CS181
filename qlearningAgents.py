# qlearningAgents.py
# ------------------
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


import math
import random

import util
from featureExtractors import *
from game import *
from learningAgents import ReinforcementAgent


class QLearningAgent(ReinforcementAgent):
    """
    Q-Learning Agent

    Functions you should fill in:
      - computeValueFromQValues
      - computeActionFromQValues
      - getQValue
      - getAction
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.q_values = util.Counter()

    def getQValue(self, state, action):
        """
        Returns Q(state,action)
        Should return 0.0 if we have never seen a state
        or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.q_values[(state, action)]

    def computeValueFromQValues(self, state):
        """
        Returns max_action Q(state,action)
        where the max is over legal actions.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        if len(actions) == 0:
            return 0.0
        return max([self.getQValue(state, act) for act in actions])

    def computeActionFromQValues(self, state):
        """
        Compute the best action to take in a state.  Note that if there
        are no legal actions, which is the case at the terminal state,
        you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        if len(actions) == 0:
            return None
        values = [(self.getQValue(state, act), act) for act in actions]
        return max(values, key=lambda x: x[0])[1]

    def getAction(self, state):
        """
        Compute the action to take in the current state.  With
        probability self.epsilon, we should take a random action and
        take the best policy action otherwise.  Note that if there are
        no legal actions, which is the case at the terminal state, you
        should choose None as the action.

        HINT: You might want to use util.flipCoin(prob)
        HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward: float):
        """
        The parent class calls this to observe a
        state = action => nextState and reward transition.
        You should do your Q-Value update here

        NOTE: You should never call this function,
        it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        self.q_values[(state, action)] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * (
            reward + self.discount * self.computeValueFromQValues(nextState)
        )

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args["epsilon"] = epsilon
        args["gamma"] = gamma
        args["alpha"] = alpha
        args["numTraining"] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
    ApproximateQLearningAgent

    You should only have to overwrite getQValue
    and update.  All other QLearningAgent functions
    should work as is.
    """

    def __init__(self, extractor="IdentityExtractor", **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
        Should return Q(state,action) = w * featureVector
        where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        feats = self.featExtractor.getFeatures(state, action)
        q_val = 0
        for feat in feats:
            q_val += feats[feat] * self.weights[feat]
        return q_val

    def update(self, state, action, nextState, reward: float):
        """
        Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        sample = reward + self.discount * self.computeValueFromQValues(nextState) - self.getQValue(state, action)
        feats = self.featExtractor.getFeatures(state, action)
        for feat in feats:
            self.weights[feat] += self.alpha * sample * feats[feat]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass


class BetterExtractor(FeatureExtractor):
    "Your extractor entry goes here.  Add features for capsuleClassic."

    def closestCap(self, pos, caps, walls):
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            if (pos_x, pos_y) in caps:
                return dist
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist + 1))
        return None

    def getFeatures(self, state, action):
        # Add more features here
        "*** YOUR CODE HERE ***"
        features = SimpleExtractor().getFeatures(state, action)

        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        newPos = (next_x, next_y)

        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        features["#-of-ghosts-1-step-away"] = 0
        for c, g in enumerate(ghosts):
            if newPos in Actions.getLegalNeighbors(g, walls):
                if state.getGhostStates()[c].scaredTimer <= 1:
                    features["#-of-ghosts-1-step-away"] += 1
        features["#-of-ghosts-1-step-away"] /= 5
        minScaredTime = min([state.getGhostStates()[c].scaredTimer for c, _ in enumerate(ghosts)])
        scaredNum = sum([state.getGhostStates()[c].scaredTimer != 0 for c, _ in enumerate(ghosts)])
        capPoses = state.getCapsules()
        # print(x, y)
        if minScaredTime <= 10 and len(capPoses) > 0:
            features["ClosestCap2Pac"] = self.closestCap(newPos, capPoses, walls) / (walls.width + walls.height) / 8
        if scaredNum == 0:
            features["ClosestCap2Pac"] *= 1.9
        elif scaredNum < 2:
            features["ClosestCap2Pac"] *= 2.9
        elif scaredNum < 3:
            features["ClosestCap2Pac"] *= 1.85

        features["eats-food"] *= 10
        return features
