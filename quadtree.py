import kdtree
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
		self.data = np.zeros((width,height))
		# self.act_probs = softmax(gradient)
		self.nodes = {(0,0): [width, height, False]} #last parameter means been inverted or not
		# self.n_unit_node = 0

	
	def update(self, action, node_x, node_y, width, height):
		# action is element from [0,1,2,3,4] 0 meaning invert whole square, 1 invert top left, 2 top right, 3 bottom left, 4 bottom right
		if action==0:
			x1, x2, y1, y2 = node_x, node_x+width, node_y, node_y+height
			if width==1 or height==1:
				self.nodes.pop((x1,y1), None)
				# self.n_unit_node = self.n_unit_node+1
			else:
				self.nodes[(x1,y1)] = [width, height, True]


		else:
			if action==1:
				x1, x2, y1, y2 = node_x, node_x+width//2, node_y, node_y+height//2
				self.nodes[(x2,y1)] = [width//2, height//2, False]
				self.nodes[(x1,y2)] = [width//2, height//2, False]
				self.nodes[(x2,y2)] = [width//2, height//2, False]

			elif action==2:
				x1, x2, y1, y2 = node_x+width//2, node_x+width, node_y, node_y+height//2
				self.nodes[(node_x,y1)] = [width//2, height//2, False]
				self.nodes[(node_x,y2)] = [width//2, height//2, False]
				self.nodes[(x1,y2)] = [width//2, height//2, False]

			elif action==3:
				x1, x2, y1, y2 = node_x, node_x+width//2, node_y+height//2, node_y+height
				self.nodes[(x1,node_y)] = [width//2, height//2, False]
				self.nodes[(x2,node_y)] = [width//2, height//2, False]
				self.nodes[(x2,y1)] = [width//2, height//2, False]

			else:
				x1, x2, y1, y2 = node_x+width//2, node_x+width, node_y+height//2, node_y+height
				self.nodes[(node_x,y1)] = [width//2, height//2, False]
				self.nodes[(node_x,node_y)] = [width//2, height//2, False]
				self.nodes[(x1,node_y)] = [width//2, height//2, False]
			self.nodes[(x1,y1)] = [width//2, height//2, True]


		for i in range(x1, x2):
			for j in range(y1, y2):
				if self.data[i,j]==0:
					self.data[i,j] = 255
				else:
					self.data[i,j] = 0


	def start_play(self):
		key = random.sample(self.nodes.keys(),1)[0]
		value = self.nodes[key]
		if value[2]:
			action = random.randint(1,4)
		elif value[1]==1 or value[0]==1:
			action = 0
		else:
			action = random.randint(0,4)
		self.update(action,key[0],key[1],value[0],value[1])


	def end(self):
		if len(self.nodes)>=self.width//3:
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


	
# size = 128
# gradient = np.random.choice([-1, 0, 1], size=(size,size))

# image = Board(size, size)
# for i in range(size//3):
# 	image.start_play()
# 	print(len(image.nodes))

# image.visualize()