# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import Queue, manhattanDistance
from game import Directions, Agent
from enum import Enum
import random, util
import numpy as np

class Action(Enum):
    North = 0
    South = 1
    East  = 2
    West  = 3
    Stop  = 4

BOARD_DATA = None
def set_board_data(board):
    global BOARD_DATA
    BOARD_DATA=board

def moveToPos(pos,move):
    x,y = pos
    match move:
        case 'North': return (x,y+1)
        case 'South': return (x,y-1)
        case 'East': return (x+1,y)
        case 'West': return (x-1,y)
        case 'Stop': return (x,y)
        case _: raise ValueError(f"Unrecognized move: '{move}'")
def posToMove(pos,move):
    x, y = pos
    if move == (x,y): return 'Stop'
    elif move == (x, y + 1): return 'North'
    elif move == (x, y - 1): return 'South'
    elif move == (x + 1, y): return 'East'
    elif move == (x - 1, y): return 'West'
    else: raise ValueError(f"Unrecognized new position: '{move}'")

def getClosestFood(state):
    pacman = state.getPacmanPosition()
    newFood = state.getFood()
    foodList = newFood.asList()

    foodDistance = []
    for pos in foodList: foodDistance.append(manhattanDistance(pacman,pos))
    #randomMinIndex provides higher variability, but firstMinIndex performs better
    return foodList[randomMinIndex(foodDistance)]

def getPosFromIndex(index):
    assert BOARD_DATA is not None
    x = index / BOARD_DATA.height
    y = index % BOARD_DATA.height
    return x,y
def getIndexFromPos(pos):
    assert BOARD_DATA is not None
    (x,y) = pos
    return x * BOARD_DATA.height + y

def firstMinIndex(lst):
    return lst.index(min(lst))
def randomMinIndex(lst):
    minVal = min(lst)
    minIndices = [index for index, value in enumerate(lst) if value == minVal]
    return random.choice(minIndices)

def get_neighbors(pos, gameState):
    x, y = pos
    neighbors = []

    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        new_pos = (x + dx, y + dy)
        if not gameState.hasWall(new_pos[0], new_pos[1]): neighbors.append(new_pos)

    return neighbors


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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:
          In my evaluation function I have divided the final score of the state in two parts
           1. When the ghosts are scared identified scaredTimes>0.
           2. Normal ghosts.
        Common evaluation score between both parts is the sum of the score for current score the steps
          for which the ghosts are scared, the reciprocal of the sum of food distance and number of foods eaten

          In the first case, from the sum I subtract the distance of the ghosts from current state
          and the number of power pellets, as the ghosts are currently in scared state. So closer pacman is to ghost better score

          In the second case since the ghosts are not scared hence distance to ghosts and number of power pellets
          are added to the sum.
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    """ Manhattan distance to the foods from the current state """
    foodList = newFood.asList()
    from util import manhattanDistance
    foodDistance = [0]
    for pos in foodList:
        foodDistance.append(manhattanDistance(newPos,pos))

    """ Manhattan distance to each ghost from the current state"""
    ghostPos = []
    for ghost in newGhostStates:
        ghostPos.append(ghost.getPosition())
    ghostDistance = [0]
    for pos in ghostPos:
        ghostDistance.append(manhattanDistance(newPos,pos))

    numberofPowerPellets = len(currentGameState.getCapsules())

    score = 0
    numberOfNoFoods = len(newFood.asList(False))           
    sumScaredTimes = sum(newScaredTimes)
    sumGhostDistance = sum (ghostDistance)
    reciprocalfoodDistance = 0
    if sum(foodDistance) > 0:
        reciprocalfoodDistance = 1.0 / sum(foodDistance)
        
    score += currentGameState.getScore()  + reciprocalfoodDistance + numberOfNoFoods

    if sumScaredTimes > 0:    
        score +=   sumScaredTimes + (-1 * numberofPowerPellets) + (-1 * sumGhostDistance)
    else :
        score +=  sumGhostDistance + numberofPowerPellets
    return score
# Abbreviation
better = betterEvaluationFunction

class RandomAgent(Agent):
    """
      Randomly chooses an action.
    """

    def getAction(self, gameState):
        """
        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        return random.choice(legalMoves)

class NaiveAgent(Agent):
    """
    Naively chooses best action at each state
    Currently using code from ReflexAgent
    """

    def naiveEval(self, currentGameState, action):
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

        # If successor state is a win state return very high score.
        if successorGameState.isWin(): return 999999

        """ Manhattan distance to the available foods from the successor state """
        foodList = newFood.asList()
        foodDistance = [0]
        for pos in foodList: foodDistance.append( manhattanDistance(newPos,pos) )
            
        """ Manhattan distance to each ghost in the game from successor state"""
        ghostPos = []
        for ghost in newGhostStates: ghostPos.append(ghost.getPosition())
        ghostDistance = []
        for pos in ghostPos: ghostDistance.append(manhattanDistance(newPos,pos))

        """ Manhattan distance to each ghost in the game from current state"""
        ghostPosCurrent = []
        for ghost in currentGameState.getGhostStates(): ghostPosCurrent.append(ghost.getPosition())
        ghostDistanceCurrent = []
        for pos in ghostPosCurrent: ghostDistanceCurrent.append(manhattanDistance(newPos,pos))

        score = 0
        # Get Number of food available in successor state
        numberOfFoodLeft = len(foodList)
        # Get Number of food available in current state
        numberOfFoodLeftCurrent = len(currentGameState.getFood().asList())
        # Get Number of Power Pellets available in successor state
        numberofPowerPellets = len(successorGameState.getCapsules())
        # Get state of ghosts in successor state
        sumScaredTimes = sum(newScaredTimes)
            
        #Relative Score    
        score += successorGameState.getScore() - currentGameState.getScore()
        if action == Directions.STOP: score -= 10
            
        # Add Score if pacman eats power pellet in next state.
        if newPos in currentGameState.getCapsules(): score += 150 * numberofPowerPellets
        # Add score if there are lesser number of food available in successor state.
        if numberOfFoodLeft < numberOfFoodLeftCurrent: score += 200

        # For each food left subtract 10 score.     
        score -= 10 * numberOfFoodLeft

        # If ghosts are scared lesser distance to ghosts is better.
        if sumScaredTimes > 0 :
            if min(ghostDistanceCurrent) < min(ghostDistance): score += 200
            else: score -=100
        # If ghosts are not scared greater distance to ghosts is better.
        else:
            if min(ghostDistanceCurrent) < min(ghostDistance): score -= 100
            else: score += 200
        
        return score

    def getAction(self, gameState):
        legalMoves = gameState.getLegalActions()
        scores = [self.naiveEval(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return legalMoves[chosenIndex]

class AStarAgent(Agent):
    """
    Uses A* to find the next action
    """

    def get_min_score_node(self,visited,f_score):
        min_score = float('inf')
        min_score_node = None
        for pos in visited.list:
            score = f_score[pos]
            if score < min_score:
                min_score = score
                min_score_node = pos
        return min_score_node

    def reconstruct_path(self,came_from,current):
        path = [current]
        while current in came_from:
            current=came_from[current]
            path.insert(0,current)
        return path

    def AStar(self,pos,goal,state):
        "Returns the first state towards the closest food"
        assert BOARD_DATA is not None
        #nodes visited
        visited=Queue()
        visited.push(pos)
        #preceding node
        came_from = {}
        #cost from start to n
        g_score = np.full((BOARD_DATA.width,BOARD_DATA.height),np.inf)
        g_score[pos]=0
        #cost from start to goal
        f_score = np.full((BOARD_DATA.width,BOARD_DATA.height),np.inf)
        f_score[pos]=manhattanDistance(pos,goal)
        while not visited.isEmpty():
            #lowest cost node
            current = self.get_min_score_node(visited,f_score)
            if current == goal: 
                path = self.reconstruct_path(came_from,current)
                if len(path)==1: return posToMove(pos,path[1])
                return posToMove(pos,path[1])
            visited.pop(current)

            neighbors = get_neighbors(current, state)
            for neighbor in neighbors:
                tentative = g_score[current] + 1
                if tentative < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    f_score[neighbor] = tentative + manhattanDistance(neighbor, goal)
                    if not visited.contains(neighbor): visited.push(neighbor)
        raise StopIteration("A* did not reach the goal state")

    def getAction(self, state):
        pacman = state.getPacmanPosition()
        closest = getClosestFood(state)
        return self.AStar(pacman,closest,state)

class QLearningAgent(Agent):
    """
    Uses a Q-learning algorithm to select the next action

    Reference: https://www.geeksforgeeks.org/machine-learning/q-learning-in-python/
    """
    def __init__(self):
        self.alpha   = 0.8 # learning rate    - determines impact of new information
        self.gamma   = 0.9 # discount         - balances immediate and future rewards
        self.epsilon = 0.2 # exploration prob - decides whether a random action is chosen
        self.epochs  = 100

        self.actions = ['North','South','East','West','Stop']
        # North, South, East, West, Stop
        self.n_actions = 5
        self.q_table = None
        self.goal = None

    #TODO: def show_Q_table(self):

    def getBestLegalAction(self, state):
        index = getIndexFromPos(state.getPacmanPosition())
        legal = state.getLegalActions()

        legal_indices = [self.actions.index(a) for a in legal]

        return max(legal_indices,key=lambda i: self.q_table[index][i]) # type: ignore

    def update_Q_table(self,state):
        assert self.goal is not None
        assert self.q_table is not None

        current_state  = state
        current_index  = getIndexFromPos(state.getPacmanPosition())

        while True: #iteration is faster and less memory intensive
            legalActions = current_state.getLegalActions()
            if not legalActions: break
            legal_indices = [self.actions.index(a) for a in legalActions]
            if np.random.rand() < self.epsilon: action = random.choice(legal_indices)
            else: action = self.getBestLegalAction(current_state)

            new_state = current_state.generatePacmanSuccessor(self.actions[action])  
            new_pacman = new_state.getPacmanPosition()
            new_index = getIndexFromPos(new_pacman)

            reward = 1 if new_pacman == self.goal else -1 if new_pacman in new_state.getGhostPositions() else 0
            current_q = self.q_table[current_index][action]

            self.q_table[current_index][action] = current_q + self.alpha * (reward + self.gamma * self.getBestLegalAction(new_state) - current_q)

            #print(self.actions[action])
            #print(new_pacman,self.goal)
            if new_pacman == self.goal: break
            current_state = new_state
            current_index = new_index

    def getAction(self,state):
        # Total number of states on the board
        n_states = BOARD_DATA.width * BOARD_DATA.height # type: ignore
        self.q_table = np.zeros((n_states,self.n_actions))
        self.goal = getClosestFood(state)

        for _ in range(self.epochs): self.update_Q_table(state)
        actionIndex = self.getBestLegalAction(state)
        return self.actions[actionIndex]
