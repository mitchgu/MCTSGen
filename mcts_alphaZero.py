# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value network
to guide the tree search and evaluate the leaf nodes

@author: Junxiao Song
"""
import numpy as np
import copy 
from quadtree import Board
import random


# assume the policy_value_fn works

def bs_policy_value_function(board):
    return random.uniform(0, 1), np.zeros((28,28,1))

def policy_value_fn(board):
    # given board outputs actions, probabilities and scores
    width = board.width
    height = board.height
    gradient = np.random.choice([-1, 0, 1], size=(width,height))
    prob_matrix = gradient2prob(board, gradient)

    actions = []
    counter = 0
    for i in board.nodes.keys():
        for j in range(5):
            prob = sum_prob(prob_matrix, i, j, board.nodes[i][0], board.nodes[i][1])
            actions.append((counter*5+j, prob))#(node to change, action)
        counter+=1

    score = random.uniform(0,1)

    return actions, score

def sum_prob(prob, node, action, width, height):
    sum = 0
    node_x = node[0]
    node_y = node[1]
    if action==0:
        x1, x2, y1, y2 = node_x, node_x+width, node_y, node_y+height
    elif action==1:
        x1, x2, y1, y2 = node_x, node_x+width//2, node_y, node_y+height//2
    elif action==2:
        x1, x2, y1, y2 = node_x+width//2, node_x+width, node_y, node_y+height//2
    elif action==3:
        x1, x2, y1, y2 = node_x, node_x+width//2, node_y+height//2, node_y+height
    else:
        x1, x2, y1, y2 = node_x+width//2, node_x+width, node_y+height//2, node_y+height
    for i in range(x1,x2):
        for j in range(y1,y2):
            sum+=prob[i][j]
    return sum

def gradient2prob(board, gradient):
    data = board.data
    width = board.width
    height = board.height
    for i in range(width): 
        for j in range(height):
            if data[i][j]==255:
                gradient[i][j] = -gradient[i][j]
    return softmax(gradient)

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors -- output from policy function - a list of tuples of actions
            and their prior probability according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value, Q plus bonus u(P).
        Returns:
        A tuple of (action, next_node)
        """
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        Arguments:
        leaf_value -- the value of subtree evaluation from the current player's perspective.        
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node: a combination of leaf evaluations, Q, and
        this node's prior adjusted for its visit count, u
        c_puct -- a number in (0, inf) controlling the relative impact of values, Q, and
            prior probability, P, on this node's score.
        """
        self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search.
    """

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """Arguments:
        policy_value_fn -- a function that takes in a board state and outputs a list of (action, probability)
            tuples and also a score in [-1, 1] (i.e. the expected value of the end game score from 
            the current player's perspective) for the current player.
        c_puct -- a number in (0, inf) that controls how quickly exploration converges to the
            maximum-value policy, where a higher value means relying on the prior more
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._visited = set()

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at the leaf and
        propagating it back through its parents. State is modified in-place, so a copy must be
        provided.
        Arguments:
        state -- a copy of the state.
        """
        node = self._root
        while(1):            
            if node.is_leaf():
                break                
            # Greedily select next move.
            action, node = node.select(self._c_puct)            
            state.update(action)

        if state.data.tobytes() in self._visited and node._parent:
            node._parent._children.pop(action) # remove this node
            return # Don't update anything      

        # Evaluate the leaf using a network which outputs a list of (action, probability)
        # tuples p and also a score v in [-1, 1] for the current player.
        
        leaf_value, _ = self._policy(state)
        action_probs = state.get_actions()


        # Check for end of game.
        end = state.end()
        if not end:
            node.expand(action_probs)
            self._visited.add(state.data.tobytes())

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """Runs all playouts sequentially and returns the available actions and their corresponding probabilities 
        Arguments:
        state -- the current state, including both game state and the current player.
        temp -- temperature parameter in (0, 1] that controls the level of exploration
        Returns:
        the available actions and the corresponding probabilities 
        """        
        for n in range(self._n_playout):
            state_copy = state.copy()
            self._playout(state_copy)
  
        # calc the move probabilities based on the visit counts at the root node
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))       
         
        return acts, act_probs

    def update_with_move(self, last_move, state):
        """Step forward in the tree, keeping everything we already know about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
            state.update(last_move)
        else:
            self._root = TreeNode(None, 1.0)
        # self._visited = _set()

    def __str__(self):
        return "MCTS"
        
class Game(object):
    """AI player based on MCTS"""
    def __init__(self, policy_value_function, c_puct=5, n_playout=2000):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)

    def step(self, board, temp=1e-3, return_prob=0):
        acts, probs = self.mcts.get_move_probs(board, temp)
        # add Dirichlet Noise for exploration (needed for self-play training)
        move_i = np.random.choice(np.arange(len(acts)), p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs))))
        # print(acts[move_i])
        self.mcts.update_with_move(acts[move_i], board) # update the root node and reuse the search tree


# class Game(object):
#     """AI player based on MCTS"""
#     def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=0):
#         self.mcts = MCTS(policy_value_function, c_puct, n_playout)
#         self._is_selfplay = is_selfplay

#     def get_action(self, board, temp=1e-3, return_prob=0):
#         n_sensible_nodes = len(board.nodes.keys())
#         move_probs = np.zeros((n_sensible_nodes,5)) # the pi vector returned by MCTS as in the alphaGo Zero paper
#         if n_sensible_nodes < board.width*board.height:
#             acts, probs = self.mcts.get_move_probs(board, temp)
#             move_probs[list(acts)] = probs         
#             if self._is_selfplay:
#                 # add Dirichlet Noise for exploration (needed for self-play training)
#                 move = np.random.choice(acts, p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs))))    
#                 self.mcts.update_with_move(move) # update the root node and reuse the search tree
#             else:
#                 # with the default temp=1e-3, this is almost equivalent to choosing the move with the highest prob
#                 move = np.random.choice(acts, p=probs)       
#                 # reset the root node
#                 self.mcts.update_with_move(-1)             
# #                location = board.move_to_location(move)
# #                print("AI move: %d,%d\n" % (location[0], location[1]))
                
#             if return_prob:
#                 return move, move_probs
#             else:
#                 return move
#         else:            
#             print("WARNING: the board is fully split")

#     def __str__(self):
#         return "MCTS {}".format(self.player)    

# size = 128
# image = Board(size, size)

# for i in range(10):
#     image.start_play()
#     policy = policy_value_fn(image)
# print(policy[0])
# print(policy[1])

# game = Game(policy)
# move = game.get_action(image)
# print(move)
