import copy
import math
import numpy as np
import scipy.misc as smp
import random


# if end when the whole space is split, the image is pretty much just noise, maybe end after a number of steps
# generate probabilities from discriminator or network?

# if width and height are 1s remove from possible nodes and add to archived nodes

def softmax(x):
	probs = np.exp(x - np.max(x))
	probs /= np.sum(probs)
	return probs


class Board(object):

	# Board for the game

	def __init__(self, width, height):
		self.width = width
		self.height = height
		self.data = np.zeros((width,height), dtype=np.uint8)
		# self.act_probs = softmax(gradient)
		self.nodes = {(0,0): [width, height, False]} #last parameter means been inverted or not
		# self.n_unit_node = 0

	def copy(self):
		c = Board(self.width, self.height)
		c.data = self.data.copy()
		c.nodes = self.nodes.copy()
		return c
	
	def update(self, action):
		# action is element from [0,1,2,3,4] 0 meaning invert whole square, 1 invert top left, 2 top right, 3 bottom left, 4 bottom right
		loc, loc_action = action
		x, y = loc
		width, height, already_inverted = self.nodes[loc]

		if loc_action==0:
			x1, x2, y1, y2 = x, x+width, y, y+height
			# if width==1 or height==1:
			# 	self.nodes.pop((x1,y1), None)
			# else:
			self.nodes[(x1,y1)] = [width, height, True]
		else:
			if loc_action==1:
				x1, x2, y1, y2 = x, x+width//2, y, y+height//2
				self.nodes[(x2,y1)] = [width//2, height//2, False]
				self.nodes[(x1,y2)] = [width//2, height//2, False]
				self.nodes[(x2,y2)] = [width//2, height//2, False]

			elif loc_action==2:
				x1, x2, y1, y2 = x+width//2, x+width, y, y+height//2
				self.nodes[(x,y1)] = [width//2, height//2, False]
				self.nodes[(x,y2)] = [width//2, height//2, False]
				self.nodes[(x1,y2)] = [width//2, height//2, False]

			elif loc_action==3:
				x1, x2, y1, y2 = x, x+width//2, y+height//2, y+height
				self.nodes[(x1,y)] = [width//2, height//2, False]
				self.nodes[(x2,y)] = [width//2, height//2, False]
				self.nodes[(x2,y1)] = [width//2, height//2, False]

			else:
				x1, x2, y1, y2 = x+width//2, x+width, y+height//2, y+height
				self.nodes[(x,y1)] = [width//2, height//2, False]
				self.nodes[(x,y)] = [width//2, height//2, False]
				self.nodes[(x1,y)] = [width//2, height//2, False]

			self.nodes[(x1,y1)] = [width//2, height//2, True]


		for i in range(x1, x2):
			for j in range(y1, y2):
				if self.data[i,j]==0:
					self.data[i,j] = 255
				else:
					self.data[i,j] = 0

	def get_actions(self, grad=None):
		actions = []
		for loc, loc_v in self.nodes.items():
			width, height, already_inverted = loc_v
			if not already_inverted:
				actions.append((loc, 0))
			if width > 1 and height > 1:
				for i in range(1,5):
					actions.append((loc, i))
		prob = 1/len(actions) if actions else 0
		actions = [(a, prob) for a in actions]
		return actions


	def update_random(self):
		# loc = random.sample(self.nodes.keys(),1)[0]
		# width, height, already_inverted = self.nodes[loc]
		# if already_inverted:
		# 	loc_action = random.randint(1,4)
		# elif width==1 or height==1:
		# 	loc_action = 0
		# else:
		# 	loc_action = random.randint(0,4)
		actions = self.get_actions()
		if True:
			action, prob = random.sample(actions, 1)[0]
			self.update(action)


	def end(self):
		if len(self.nodes)>128:
			return True
		else:
			return False

	def visualize(self):
		data = self.data
		width = self.width
		height = self.height
		temp = np.zeros( (width,height,3), dtype=np.uint8 )
		for i in range(width):
			for j in range(height):
				for k in range(3):
					temp[i][j][k] = data[i][j]
		img = smp.toimage( temp )
		img.show()


	
# size = 28
# # gradient = np.random.choice([-1, 0, 1], size=(size,size))

# image = Board(size, size)
# for i in range(30):
# 	image.start_play()
# 	print(len(image.nodes))

# image.visualize()