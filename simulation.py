import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage import color
from skimage import img_as_float
import scipy
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
death = 0.0

# Target pop
pop_target = 5000.0

# Number of initial cells
init_cells = 5000

# Size of image space
img_size = np.array([800,800,200])

# Slice to view image at
img_slice = 100

# Nr of timesteps
# For now image timesteps are coupled to dynamics
nr_timesteps = 1

# Cell dims, given in radius
cell_dims = np.array([20,20,16],dtype=np.int)

# Intensity power
power = 20

# For debugging
np.set_printoptions(threshold=np.nan)

#-------------------------------

def generate_gaussian(age):
	x=np.linspace(-cell_dims[0],cell_dims[0],cell_dims[0]*2+1)
	y=np.linspace(-cell_dims[1],cell_dims[1],cell_dims[1]*2+1)
	z=np.linspace(-cell_dims[2],cell_dims[2],cell_dims[2]*2+1)
	x,y,z = np.meshgrid(y, x, z)
	if age == split_age or age == 0 or age == 1:
		sigma_x = 0.5*cell_dims[0]*(age+split_age)/(2.0*split_age)
		sigma_y = 0.5*cell_dims[1]*(age+split_age)/(2.0*split_age)
		sigma_z = 0.5*cell_dims[2]*(age+split_age)/(2.0*split_age)
	else:
		sigma_x = 0.5*cell_dims[0]*(age+split_age)/(2.0*split_age)
		sigma_y = 0.5*cell_dims[1]*(age+split_age)/(2.0*split_age)
		sigma_z = 0.5*cell_dims[2]*(age+split_age)/(2.0*split_age)
	k = (1/(2*np.pi*sigma_x*sigma_y*sigma_z) * np.exp(-(x**2/(2*sigma_x**2)
	     + y**2/(2*sigma_y**2) + z**2/(2*sigma_z**2))))*(2**power)*(((age+split_age)/(2.0*split_age))**3)


	k = np.clip(k,0,255)
	k = np.uint8(k) + 5
	return k

def add_cell_to_img(image,center,cell,rotate='True'):
	c_int = center.astype(int)	
	#print(c_int)
	x=[max(c_int[0]-cell_dims[0],0),min(c_int[0]+cell_dims[0]+1,img_size[0])]
	y=[max(c_int[1]-cell_dims[1],0),min(c_int[1]+cell_dims[1]+1,img_size[1])]
	z=[max(c_int[2]-cell_dims[2],0),min(c_int[2]+cell_dims[2]+1,img_size[2])]

	#if rotate:
	#	cell =  scipy.ndimage.rotate(cell,30)

	try:
		image[x[0]:x[1],y[0]:y[1],z[0]:z[1]] += cell[0:(x[1]-x[0]),0:(y[1]-y[0]),0:(z[1]-z[0])]  
	except ValueError:
		print(image.shape,image[x[0]:x[1],y[0]:y[1],z[0]:z[1]].shape)
		print(cell[0:(x[1]-x[0]),0:(y[1]-y[0]),0:(z[1]-z[0])].shape)

def check_cell_bounds(cell):
	if cell.coords[0] < 0:
		return True
	if cell.coords[0] > img_size[0]:
		return True
	if cell.coords[1] < 0:
		return True
	if cell.coords[1] > img_size[1]:
		return True
	if cell.coords[2] < 0:
		return True
	if cell.coords[2] > img_size[2]:
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
		cells.add(Cell(np.random.rand(3)*img_size,i,i))

	unique_id = init_cells
	img = None

	cell_kernel = generate_gaussian(2)

	for timesteps in range(nr_timesteps):
		remove = set([])
		add=set([])
		pop = cells.__len__()
		
		image = np.zeros((img_size[0],img_size[1],img_size[2]),dtype=np.uint8)
		for cell in cells:
			add_cell_to_img(image,cell.coords,cell_kernel)
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
			image = color.gray2rgb(image[:,:,img_slice]) *[0,1,0]
			img = plt.imshow(image)
			#img = plt.imshow(image[:,:,img_slice],cmap='gist_gray')
		#else:
			#img.set_data(image[:,:,img_slice])
		plt.show()		
		#plt.pause(.5)
		#plt.draw()

	

	

