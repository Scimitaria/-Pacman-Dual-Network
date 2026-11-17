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


from util import Queue, manhattanDistance, aStarDistance, tPrint, lookup, posToMove, get_neighbors
from game import Directions, Agent
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
import random, json, os, math
import numpy as np # type: ignore

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

PARALLEL = None
def set_parallel():
    global PARALLEL
    PARALLEL = True

def getClosestFood(state):
    pacman = state.getPacmanPosition()
    newFood = state.getFood()
    foodList = newFood.asList()

    foodDistance = []
    for pos in foodList: foodDistance.append(aStarDistance(pacman,pos,state,BOARD_DATA))
    #randomMinIndex provides higher variability, but firstMinIndex performs better
    return foodList[randomMinIndex(foodDistance)]
def getClosestGhost(state):
    pacman = state.getPacmanPosition()
    ghosts = state.getGhostPositions()
    ghostDistance = []
    for pos in ghosts: 
        try: ghostDistance.append(aStarDistance(pacman,tuple(map(round,pos)),state,BOARD_DATA))
        except: ghostDistance.append(manhattanDistance(pacman,tuple(map(round,pos))))
    return ghosts[randomMinIndex(ghostDistance)]
def getClosestCapsule(state):
    capsules = state.getCapsules()
    if not capsules: return None
    pacman = state.getPacmanPosition()
    capsuleDistances = []
    for pos in capsules: capsuleDistances.append(aStarDistance(pacman,pos,state,BOARD_DATA))
    return capsules[randomMinIndex(capsuleDistances)]

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

def normalizePos(pos):
    return tuple(map(round,pos))

NEXT_TO_DIST = 1 # maze distance at or below this is "nextto"
NEAR_DIST    = 5 # at or below this is "near" (matches safe_distance in getActionFromPolicy), above is "far"
def getDistanceClass(pacman, pos, state):
    "Buckets the maze distance from pacman to pos into nextto / near / far"
    pos = normalizePos(pos)
    try: dist = aStarDistance(pacman, pos, state, BOARD_DATA) - 1 # aStarDistance counts squares on the path incl. the start
    except: dist = manhattanDistance(pacman, pos)
    if dist <= NEXT_TO_DIST: return "nextto"
    if dist <= NEAR_DIST: return "near"
    return "far"

def toTuple(x):
    "JSON turns tuples into lists; this turns them back at every nesting level so keys are hashable"
    return tuple(toTuple(i) for i in x) if isinstance(x, list) else x

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
        self.evaluationFunction = lookup(evalFn, globals())
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

class naiveAStarAgent(Agent):
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
        closestFood = getClosestFood(state)
        return self.AStar(pacman,closestFood,state)

class AStarAgent(Agent):
    """
    Uses A* to find the next action
    Closest food if safe, flees if in danger
    """
    #def __init__(self, index=0):
    #    super().__init__(index)
    #    self.last_action = None

    def get_min_score_node(self,visited,f_score):
        min_score = float('inf')
        min_score_node = None
        for pos in visited.list:
            score = f_score[pos]
            if score < min_score:
                min_score = score
                min_score_node = pos
        return min_score_node
    def get_max_score_node(self,visited,f_score):
        max_score = float('-inf')
        max_score_node = None
        for pos in visited.list:
            score = f_score[pos]
            if score > max_score:
                max_score = score
                max_score_node = pos
        return max_score_node

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
        f_score[pos]=aStarDistance(pos,goal,state,BOARD_DATA)
        while not visited.isEmpty():
            #lowest cost node
            current = self.get_min_score_node(visited,f_score)
            if current == goal: 
                path = self.reconstruct_path(came_from,current)
                return posToMove(pos,path[1])
            visited.pop(current)

            neighbors = get_neighbors(current, state)
            for neighbor in neighbors:
                tentative = g_score[current] + 1
                if tentative < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    f_score[neighbor] = tentative + aStarDistance(neighbor, goal, state, BOARD_DATA)
                    if not visited.contains(neighbor):
                        visited.push(neighbor)
        raise StopIteration("A* did not reach the goal state")

    def flee(self,pos,ghosts,state):
        "Runs away as efficiently as possible"
        assert BOARD_DATA is not None
        neighbors = get_neighbors(pos, state)
        maxDist = 0
        maxPos = None
        for neighbor in neighbors:
            dist = sum([aStarDistance(neighbor,ghost[0],state,BOARD_DATA) for ghost in ghosts])
            if dist > maxDist:
                maxDist = dist
                maxPos = neighbor
        return posToMove(pos,maxPos)

    def getAction(self, state):
        if False:pass#not state.isJunction() and self.last_action is not None: return self.last_action
        else:
            assert BOARD_DATA is not None
            safe_distance = 5
            pacman = state.getPacmanPosition()
            closestFood = getClosestFood(state)
            closestGhosts = []
            closestCapsule = getClosestCapsule(state)

            for ghost in state.getGhostPositions():
                ghostState = None
                pos = normalizePos(ghost)
                ghostDist = aStarDistance(pos,pacman,state,BOARD_DATA)
                if ghostDist < safe_distance and ghostDist < aStarDistance(pacman,closestFood,state,BOARD_DATA)+2:
                    for agent in state.data.agentStates:
                        agentPos = agent.getPosition()
                        if agentPos == ghost: 
                            ghostState = agent
                            closestGhosts.append((pos,agent,ghostDist))
                            continue
                    if ghostState is None: raise IndexError(f"Ghost state not found. \n\tPosition: {ghost}; \n\tghost positions: {[agent.getPosition() for agent in state.data.agentStates]}")

            closestGhost = getClosestGhost(state)
            closestState = [ghost for ghost in state.data.agentStates if ghost.getPosition() == closestGhost][0]
            assert closestState is not None
            if closestState.scaredTimer > 1:
                #Chase vulnerable ghosts
                return self.AStar(pacman,normalizePos(closestGhost),state)
            if closestGhosts:
                #Run away if ghosts are close
                return self.flee(pacman,closestGhosts,state)
            if closestCapsule is not None:
                #Eat capsules
                return self.AStar(pacman,closestCapsule,state)
            return self.AStar(pacman,closestFood,state)

class randStateAgent(Agent):
    """
    Uses A* to find the next action
    Closest food if safe, flees if in danger
    """
    def __init__(self, index=0):
        super().__init__(index)
        self.last_action = None

    def get_min_score_node(self,visited,f_score):
        min_score = float('inf')
        min_score_node = None
        for pos in visited.list:
            score = f_score[pos]
            if score < min_score:
                min_score = score
                min_score_node = pos
        return min_score_node
    def get_max_score_node(self,visited,f_score):
        max_score = float('-inf')
        max_score_node = None
        for pos in visited.list:
            score = f_score[pos]
            if score > max_score:
                max_score = score
                max_score_node = pos
        return max_score_node

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
        f_score[pos]=aStarDistance(pos,goal,state,BOARD_DATA)
        while not visited.isEmpty():
            #lowest cost node
            current = self.get_min_score_node(visited,f_score)
            if current == goal: 
                path = self.reconstruct_path(came_from,current)
                return posToMove(pos,path[1])
            visited.pop(current)

            neighbors = get_neighbors(current, state)
            for neighbor in neighbors:
                tentative = g_score[current] + 1
                if tentative < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    f_score[neighbor] = tentative + aStarDistance(neighbor, goal, state, BOARD_DATA)
                    if not visited.contains(neighbor):
                        visited.push(neighbor)
        raise StopIteration("A* did not reach the goal state")

    def flee(self,pos,ghosts,state):
        "Runs away as efficiently as possible"
        assert BOARD_DATA is not None
        neighbors = get_neighbors(pos, state)
        maxDist = 0
        maxPos = None
        for neighbor in neighbors:
            dist = sum([aStarDistance(neighbor,ghost[0],state,BOARD_DATA) for ghost in ghosts])
            if dist > maxDist:
                maxDist = dist
                maxPos = neighbor
        if maxPos is None: maxPos = random.choice(neighbors)
        return posToMove(pos,maxPos)

    def getAction(self, state):
        if not state.isJunction() and self.last_action is not None: return self.last_action
        else:
            assert BOARD_DATA is not None
            safe_distance = 5
            pacman = state.getPacmanPosition()
            closestFood = getClosestFood(state)
            closestGhosts = []
            for ghost in state.getGhostPositions():
                ghostState = None
                pos = normalizePos(ghost)
                ghostDist = aStarDistance(pos,pacman,state,BOARD_DATA)
                if ghostDist < safe_distance and ghostDist < aStarDistance(pacman,closestFood,state,BOARD_DATA)+2:
                    for agent in state.data.agentStates:
                        agentPos = agent.getPosition()
                        if agentPos == ghost: 
                            ghostState = agent
                            closestGhosts.append((pos,agent,ghostDist))
                            continue
                    if ghostState is None: raise IndexError(f"Ghost state not found. \n\tPosition: {ghost}; \n\tghost positions: {[agent.getPosition() for agent in state.data.agentStates]}")

            closestGhost = getClosestGhost(state)
            closestState = [ghost for ghost in state.data.agentStates if ghost.getPosition() == closestGhost][0]
            assert closestState is not None

            match random.randint(0,2):
                case 0:
                    #Chase vulnerable ghosts
                    return self.AStar(pacman,normalizePos(closestGhost),state)
                case 1:
                    #Run away
                    return self.flee(pacman,closestGhosts,state)
                case _: return self.AStar(pacman,closestFood,state)

class naiveQLearningAgent(Agent):
    """
    Uses a Q-learning algorithm to select the next action
    Currently broken

    Reference: https://www.geeksforgeeks.org/machine-learning/q-learning-in-python/
    """
    def __init__(self):
        self.alpha   = 0.8 # learning rate    - determines impact of new information
        self.gamma   = 0.3 # discount         - balances immediate and future rewards
        self.epsilon = 0.3 # exploration prob - decides whether a random action is chosen
        self.epochs = 0
        self.gameDepth = 100

        self.actions = ['North','South','East','West','Stop']
        with open('q.json','r') as q: data = json.load(q)
        self.q_table = { tuple(tuple(x) for x in json.loads(key)): value for key, value in data.items() }

        self.isTraining = True

    def assembleTableState(self,state):
        (px,py) = state.getPacmanPosition()
        (fx,fy) = getClosestFood(state)
        (gx,gy) = getClosestGhost(state)
        return ((px,py),(fx-px,fy-py),(gx-px,gy-py))

    def getBestLegalAction(self, state, legalActions):
        try:
            state_qs = self.q_table[state]
            legal_qs = {action: state_qs[action] for action in legalActions}
            return max(legal_qs,key=legal_qs.get) # type: ignore
        except:
            return random.choice(legalActions)

    def getBestQValue(self, state):
        try: return max(self.q_table[state].values())
        except: return 0

    def update_Q_table(self,state):
        assert self.q_table is not None

        current_state  = state
        current_qState = self.assembleTableState(state)

        for _ in range(self.gameDepth): #iteration is faster and less memory intensive
            legalActions = current_state.getLegalActions()
            if not legalActions: break
            if np.random.rand() < self.epsilon: action = random.choice(legalActions)
            else: action = self.getBestLegalAction(current_qState,legalActions)

            new_state = current_state.generatePacmanSuccessor(action)
            if new_state.getFood().isEmpty(): break
            new_qState = self.assembleTableState(new_state)

            reward = betterEvaluationFunction(new_state)
            try: current_q = self.q_table[current_qState][action]
            except: current_q = 0

            try: self.q_table[current_qState][action] = current_q + self.alpha * (reward + self.gamma * self.getBestQValue(new_qState) - current_q)
            except:
                self.q_table[current_qState] = {}
                self.q_table[current_qState][action] = current_q + self.alpha * (reward + self.gamma * self.getBestQValue(new_qState) - current_q)

            current_state = new_state
            current_qState = new_qState

    def getAction(self,state):
        if self.isTraining:
            for epoch in range(self.epochs):
                tPrint(f"Epochs: {epoch}") 
                self.update_Q_table(state)
                with open('q.json','w') as file:
                    file.seek(0)
                    file.truncate()
                    q_table_json = { json.dumps(key): value for key, value in self.q_table.items() }
                    json.dump(q_table_json, file, indent=4)
            self.isTraining = False
        return self.getBestLegalAction(self.assembleTableState(state),state.getLegalActions())

class QLearningAgent(Agent):
    """
    Uses a Q-learning algorithm to select the best policy

    Reference: https://www.geeksforgeeks.org/machine-learning/q-learning-in-python/

    Framework version: python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic
    """
    def __init__(self, q_table=None, fixedPolicy=None):
        self.alpha   = 0.2 # learning rate    - determines impact of new information
        self.gamma   = 0.8 # discount         - balances immediate and future rewards
        self.epsilon = 0.3 # exploration prob - decides whether a random action is chosen
        self.epochs  = 1000 
        self.gameDepth = 100
        self.syncEvery = 10
        self.cores = 8

        self.isTraining = self.epochs>0
        self.isLearning = False
        self.lastQState = None
        self.lastPolicy = None
        self.lastScore  = 0 # game score when the previous policy was chosen, for the reward delta

        self.policies = ["Eat pellets","Chase ghosts","Eat capsules","Flee ghosts"]

        # Baseline mode: -a fixedPolicy=random or -a "fixedPolicy=Eat pellets" (any name from self.policies)
        # Skips training and the Q-table entirely; every move uses that policy (or a random one each move)
        if fixedPolicy is not None and fixedPolicy != "random" and fixedPolicy not in self.policies:
            raise ValueError(f"fixedPolicy must be 'random' or one of {self.policies}, got {fixedPolicy!r}")
        self.fixedPolicy = fixedPolicy
        if fixedPolicy is not None:
            self.isTraining = False
            self.q_table = {}
        elif q_table is not None: self.q_table = q_table # parallel workers get the table passed in
        else:
            with open('q.json','r') as q: data = json.load(q)
            self.q_table = { toTuple(json.loads(key)): value for key, value in data.items() }
        self.visits = {} # (table state, policy) -> number of updates, used to weight the parallel merge
        self.policyCalls = 0     # times getBestPolicy was called
        self.randomFallbacks = 0 # times getBestPolicy picked randomly because the state wasn't in the table

    def get_min_score_node(self,visited,f_score):
        min_score = float('inf')
        min_score_node = None
        for pos in visited.list:
            score = f_score[pos]
            if score < min_score:
                min_score = score
                min_score_node = pos
        return min_score_node
    def get_max_score_node(self,visited,f_score):
        max_score = float('-inf')
        max_score_node = None
        for pos in visited.list:
            score = f_score[pos]
            if score > max_score:
                max_score = score
                max_score_node = pos
        return max_score_node

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
        f_score[pos]=aStarDistance(pos,goal,state,BOARD_DATA)
        while not visited.isEmpty():
            #lowest cost node
            current = self.get_min_score_node(visited,f_score)
            if current == goal: 
                path = self.reconstruct_path(came_from,current)
                return posToMove(pos,path[1])
            visited.pop(current)

            neighbors = get_neighbors(current, state)
            for neighbor in neighbors:
                tentative = g_score[current] + 1
                if tentative < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    f_score[neighbor] = tentative + aStarDistance(neighbor, goal, state, BOARD_DATA)
                    if not visited.contains(neighbor):
                        visited.push(neighbor)
        raise StopIteration("A* did not reach the goal state")

    def flee(self,pos,ghosts,state):
        "Runs away as efficiently as possible"
        if not ghosts: return random.choice(state.getLegalPacmanActions())
        assert BOARD_DATA is not None
        neighbors = get_neighbors(pos, state)
        maxDist = 0
        maxPos = None
        for neighbor in neighbors:
            try: dist = sum([aStarDistance(neighbor,ghost[0],state,BOARD_DATA) for ghost in ghosts])
            except: dist = sum([manhattanDistance(neighbor,ghost[0]) for ghost in ghosts])
            if dist > maxDist:
                maxDist = dist
                maxPos = neighbor
        return posToMove(pos,maxPos)


    def getAStarDirection(self, pacman, pos, state):
        "First move on the A* path from pacman to pos (North/South/East/West)"
        pos = normalizePos(pos)
        if pos == pacman: return "Here"
        try: return self.AStar(pacman, pos, state)
        except StopIteration: return "Unreachable"

    def assembleTableState(self,state):
        (px,py) = state.getPacmanPosition()
        (fx,fy) = getClosestFood(state)
        #t_state = ((px,py),(fx-px,fy-py))
        food     = self.getAStarDirection((px,py),(fx, fy), state)
        capsules = tuple(self.getAStarDirection((px,py), c, state) for c in state.getCapsules())
        ghosts   = tuple(
            (self.getAStarDirection((px,py), g.getPosition(), state), getDistanceClass((px,py), g.getPosition(), state), g.scaredTimer>1)
            for g in state.getGhostStates()
        )
        # e.g. ("North", ("West",), (("East","near",False), ("North","far",True)))
        return (food, capsules, ghosts)

    def getPolicyQs(self, state):
        """
        Q values to choose from for a known state. The table only stores policies that were actually tried.
        Training: untried policies count as 0 so they still get explored.
        Playing:  only tried policies are considered, so "never tried" can't beat "tried and was bad".
        """
        state_qs = self.q_table[state]
        if self.isLearning: return {policy: state_qs.get(policy, 0) for policy in self.policies}
        return {policy: q for policy, q in state_qs.items() if policy in self.policies}

    def getBestPolicy(self, state):
        self.policyCalls += 1
        policies = self.getPolicyQs(state) if state in self.q_table else {}
        if not policies:
            self.randomFallbacks += 1
            return random.choice(self.policies)
        best = max(policies.values())
        return random.choice([policy for policy, q in policies.items() if q == best]) # random tie-break

    def logFallbacks(self, label):
        pct = 100 * self.randomFallbacks / self.policyCalls if self.policyCalls else 0
        tPrint(f"[{label}] getBestPolicy random fallbacks: {self.randomFallbacks}/{self.policyCalls} ({pct:.1f}%)")
    def getActionFromPolicy(self, state, policy):
        safe_distance = 5
        pacman = state.getPacmanPosition()
        closestFood = getClosestFood(state)
        closestGhosts = []
        closestCapsule = getClosestCapsule(state)

        for ghost in state.getGhostPositions():
            ghostState = None
            pos = normalizePos(ghost)
            try: ghostDist = aStarDistance(pos,pacman,state,BOARD_DATA)
            except: ghostDist = manhattanDistance(pos,pacman)
            if ghostDist < safe_distance and ghostDist < aStarDistance(pacman,closestFood,state,BOARD_DATA)+2:
                for agent in state.data.agentStates:
                    agentPos = agent.getPosition()
                    if agentPos == ghost: 
                        ghostState = agent
                        closestGhosts.append((pos,agent,ghostDist))
                        continue
                if ghostState is None: raise IndexError(f"Ghost state not found. \n\tPosition: {ghost}; \n\tghost positions: {[agent.getPosition() for agent in state.data.agentStates]}")

        closestGhost = getClosestGhost(state)
        closestState = [ghost for ghost in state.data.agentStates if ghost.getPosition() == closestGhost][0]
        assert closestState is not None

        #Had to add cond back in to prevent length-1 A* path
        if policy == "Chase ghosts" and closestState.scaredTimer>1:
            try: return self.AStar(pacman,closestGhost,state)
            except StopIteration: pass
        if policy == "Flee ghosts" and closestGhosts:  return self.flee(pacman,closestGhosts,state)
        if policy == "Eat capsules" and closestCapsule is not None: return self.AStar(pacman,closestCapsule,state)
        return self.AStar(pacman,closestFood,state)

    def getBestQValue(self, state):
        "Value of the next state for the Q update (only called while training, so untried policies count as 0)"
        if state not in self.q_table: return 0
        return max(self.getPolicyQs(state).values())

    def save_q_table(self,table):
        q_table_json = { json.dumps(key): value for key, value in self.q_table.items() }
        with open(f"{table}.json", "w") as file: 
            file.seek(0)
            file.truncate()
            json.dump(q_table_json, file, indent=4)
    def update_Q_table(self,state):
        "Updates Q for the previous move, now that its outcome (Pacman and ghost moves) is known"
        if self.lastQState is None: return
        # Reward is the change in game score since the previous policy was chosen:
        # +10 pellet, +200 ghost, -500 death, +500 win, -1 per time step
        reward = state.getScore() - self.lastScore
        future = 0 if state.isLose() or state.isWin() else self.getBestQValue(self.assembleTableState(state))

        try: current_q = self.q_table[self.lastQState][self.lastPolicy]
        except: current_q = 0
        self.q_table.setdefault(self.lastQState, {})[self.lastPolicy] = current_q + self.alpha * (reward + self.gamma * future - current_q)
        key = (self.lastQState, self.lastPolicy)
        self.visits[key] = self.visits.get(key, 0) + 1
    def merge_Q_tables(self,results):
        "Uses federated q-learning to weight the table"
        totals = {}
        for result in results:
            for (qState, policy), (q, n) in result.items():
                weighted, count = totals.get((qState, policy), (0,0))
                totals[(qState,policy)] = (weighted+n*q, count+n)
        for (qState, policy), (weighted, count) in totals.items():
            self.q_table.setdefault(qState, {})[policy] = weighted/count

    def playTrainingGame(self, layout, numGhosts):
        "Plays one full game through game.py, learning from every move"
        from pacman import ClassicGameRules
        from ghostAgents import RandomGhost
        import textDisplay

        ghosts = [RandomGhost(i) for i in range(1, numGhosts + 1)] #Pacman is another agent in the game with index 0
        self.isLearning = True
        ClassicGameRules().newGame(layout, self, ghosts, textDisplay.NullGraphics(), quiet=True).run()
        self.isLearning = False

    def trainWorker(self,q_table, layout, numGhosts, games):
        "Runs in its own process: plays training games on a private copy of the table, returns ({(state, policy): (q, visits)}, (policyCalls, randomFallbacks))"
        agent = QLearningAgent(q_table=q_table)
        agent.isTraining = False
        for _ in range(games): agent.playTrainingGame(layout, numGhosts)
        qs = { (qState, policy): (agent.q_table[qState][policy], n) for (qState, policy), n in agent.visits.items() }
        return qs, (agent.policyCalls, agent.randomFallbacks)

    def train(self, state):
        "Each epoch is a full game played through game.py"
        layout = state.data.layout
        numGhosts = state.getNumAgents() - 1

        if not PARALLEL:
            for epoch in range(self.epochs):
                tPrint(f"Epochs: {epoch}")
                self.playTrainingGame(layout, numGhosts)
        else:
            workers = os.cpu_count() or 1
            done = 0
            lastCheckpoint = 0 # largest power of 10 already saved, so each q-{inc}.json is written once
            with ProcessPoolExecutor(max_workers=workers) as executor:
                while done < self.epochs:
                    # Split this round's games across workers, each starting from the same table
                    n = min(workers * self.syncEvery, self.epochs - done)
                    games = [n // workers + (1 if i < n % workers else 0) for i in range(workers)]
                    games = [g for g in games if g > 0]
                    results = list(executor.map(self.trainWorker, [self.q_table] * len(games), [layout] * len(games), [numGhosts] * len(games), games))
                    self.merge_Q_tables([qs for qs, _ in results])
                    # Workers don't share memory, so each returns its own counts and they're summed here
                    for _, (calls, fallbacks) in results:
                        self.policyCalls += calls
                        self.randomFallbacks += fallbacks
                    done += n
                    inc = 10 ** math.floor(math.log10(done)) # Round down to nearest power of 10
                    saved = inc > lastCheckpoint # first round past this power of 10
                    if saved:
                        self.save_q_table(f"q-{inc}")
                        lastCheckpoint = inc
                    tPrint(f"Epochs: {done}/{self.epochs} and saved: {saved} to q-{inc}")
        self.logFallbacks("training")
        self.policyCalls = 0 # reset so the real game is counted separately
        self.randomFallbacks = 0
        self.save_q_table('q')

    def final(self, state):
        "Called by game.py when a game ends"
        if not self.isLearning:
            if self.fixedPolicy is None: self.logFallbacks("game")
            self.policyCalls = 0 # reset so each game reports its own counts
            self.randomFallbacks = 0
            return
        self.update_Q_table(state)
        self.lastQState = None
        self.lastPolicy = None
        self.lastScore  = 0

    def getAction(self,state):
        assert BOARD_DATA is not None
        if self.isTraining:
            self.isTraining = False
            self.train(state)

        if self.fixedPolicy is not None:
            policy = random.choice(self.policies) if self.fixedPolicy == "random" else self.fixedPolicy
            return self.getActionFromPolicy(state, policy)

        qState = self.assembleTableState(state)
        if not self.isLearning:
            policy = self.getBestPolicy(qState)
        else:
            self.update_Q_table(state)
            if np.random.rand() < self.epsilon: policy = random.choice(self.policies)
            else: policy = self.getBestPolicy(qState)
            self.lastQState = qState
            self.lastPolicy = policy
            self.lastScore  = state.getScore()
        return self.getActionFromPolicy(state,policy)
