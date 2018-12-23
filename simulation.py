import numpy as np
import matplotlib.pyplot as plt
#from PIL import Image, ImageDraw


# Simulation parameters
# Velocity change per timestep
dv = 1

# Maybe try Langevin motion?
#m = 1
#l = 0.1

# Cell split rate
# Not stochastic anymore.
#split = 0.01
split_age = 25

# Death rate
death = 0.05

# Target pop
pop_target = 100.0

# Number of initial cells
init_cells = 10

# Size of image space
img_size = 256

# Nr of timesteps
# For now image timesteps are coupled to dynamics
nr_timesteps = 1000

# For debugging
np.set_printoptions(threshold=np.nan)

#-------------------------------

def add_gaussian(array,center,age):
	x=np.linspace(-4,4,9)
	y=np.linspace(-4,4,9)
	x,y = np.meshgrid(x, y)
	if age == split_age or age == 0 or age == 1:
		sigma_x = 0.5*(age+split_age)/(2.0*split_age)
		sigma_y = 2.0*(age+split_age)/(2.0*split_age)
	else:
		sigma_x = 2.0*(age+split_age)/(2.0*split_age)
		sigma_y = 2.0*(age+split_age)/(2.0*split_age)
	z = (1/(2*np.pi*sigma_x*sigma_y) * np.exp(-(x**2/(2*sigma_x**2)
	     + y**2/(2*sigma_y**2))))*2**12*(age+split_age)/(2.0*split_age)
	z = np.clip(z,0,255)
	z = np.uint8(z)
	
	c_int = center.astype(int)	
	x=[max(c_int[0]-4,0),min(c_int[0]+5,img_size)]
	y=[max(c_int[1]-4,0),min(c_int[1]+5,img_size)]
	try:
		array[x[0]:x[1],y[0]:y[1]] += z[0:x[1]-x[0],0:y[1]-y[0]]  
	except ValueError:
		print(x,y)

def check_cell_bounds(cell):
	if cell.coords[0] < 0:
		return True
	if cell.coords[0] > 255:
		return True
	if cell.coords[1] < 0:
		return True
	if cell.coords[1] > 255:
		return True
	return False
		

# Defines a single cell
# 	A cell sits in x,y,z space
#	Every timestep, a cell moves in a random (for now) direction
#	At every timestep, a cell splits with probability x
#	At every timestep, a cell dies with probability y
#	Each cell keeps track of its lineage for tracking

class Cell(object):

	# Add imaging parameters
	

	def __init__(self,coords, lineage, unique_id,age=-1):
		self.coords = coords
		self.lineage = lineage
		self.unique_id = unique_id
		if age == -1:
			self.age = int(np.random.rand()*10)
		else:
			self.age = age
		
		self.velocity = (np.random.rand(3)-0.5)*7


	# Take a timestep forward
	def movecell(self,vel_change):
		self.coords += self.velocity
		self.velocity += ((np.random.rand(3)*2-1)*vel_change)/(np.linalg.norm(self.velocity)*2)
		self.age += 1


if __name__ == '__main__':

	cells = set([])
	for i in range(init_cells):
		cells.add(Cell(np.random.rand(3)*256,i,i))

	unique_id = init_cells
	img = None

	for timesteps in range(nr_timesteps):
		remove = set([])
		add=set([])
		pop = cells.__len__()
		
		image = np.zeros((img_size,img_size),dtype=np.uint8)
		for cell in cells:
			add_gaussian(image,cell.coords,cell.age)
			if np.random.rand(1) < death*(pop/pop_target) or check_cell_bounds(cell):
				remove.add(cell)
			#if np.random.rand(1) < split*(pop_target/pop):
			cell.movecell(dv)
			if cell.age == split_age:
				cell.age = 0
				add.add(cell)
		cells = (cells - remove)

		for cell in add:
			cells.add(Cell(cell.coords+1,cell.lineage,unique_id,age=0))
			unique_id += 1

		if img is None:
			img = plt.imshow(image)
		else:
			img.set_data(image)		
		plt.pause(.5)
		plt.draw()

	

