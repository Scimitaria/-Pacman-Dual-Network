# textDisplay.py
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


import time, sys
try: 
    import pacman
except:
    pass

DRAW_EVERY = 1
SLEEP_TIME = 0 # This can be overwritten by __init__
DISPLAY_MOVES = False
QUIET = False # Supresses output

previous_lines = 0

class NullGraphics:
    def initialize(self, state, isBlue = False):
        pass

    def update(self, state):
        pass

    def checkNullDisplay(self):
        return True

    def pause(self):
        time.sleep(SLEEP_TIME)

    def draw(self, state):
        print(state)

    def updateDistributions(self, dist):
        pass

    def finish(self):
        pass

class PacmanGraphics:
    def __init__(self, speed=None):
        if speed != None:
            global SLEEP_TIME
            SLEEP_TIME = speed

    def initialize(self, state, isBlue = False):
        self.draw(state)
        self.pause()
        self.turn = 0
        self.agentCounter = 0

    def update(self, state):
        numAgents = len(state.agentStates)
        self.agentCounter = (self.agentCounter + 1) % numAgents
        if self.agentCounter == 0:
            self.turn += 1
            if DISPLAY_MOVES:
                ghosts = [pacman.nearestPoint(state.getGhostPosition(i)) for i in range(1, numAgents)]
                print("%4d) P: %-8s" % (self.turn, str(pacman.nearestPoint(state.getPacmanPosition()))),'| Score: %-5d' % state.score,'| Ghosts:', ghosts) # type: ignore
            if self.turn % DRAW_EVERY == 0:
                self.draw(state)
                self.pause()
        if state._win or state._lose:
            self.draw(state)

    def pause(self):
        time.sleep(SLEEP_TIME)

    def draw_buffered(self,state):
        global previous_lines
        out = str(state)

        if previous_lines:
            # Move to the beginning of the first line of the old state
            sys.stdout.write(f"\033[{previous_lines}A")

            # Clear each old line
            for _ in range(previous_lines):
                sys.stdout.write("\033[2K")
                sys.stdout.write("\033[1B")

            # Return to the beginning of the first line
            sys.stdout.write(f"\033[{previous_lines}A")

        sys.stdout.write(out)
        sys.stdout.flush()

        previous_lines = out.count("\n")
    def draw(self,state):
        print(str(state))

    def finish(self):
        pass
