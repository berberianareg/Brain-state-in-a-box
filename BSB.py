"""Brain-state-in-a-box (BSB) model for unsupervised categorization.

Notes
-----
  This script is version v0. It provides the base for all subsequent
  iterations of the project.

Requirements
------------
  See "requirements.txt"
  
Observations
------------
  Starting at some initial point of the square, positive feedback
  will drive the system toward one of the corners.
  
  The final state of the system will always be a corner.
  
  There are four distinct basins of attraction.
  
"""

#%% import libraries and modules
import numpy as np  
import matplotlib.pyplot as plt
import os

#%% figure parameters
plt.rcParams['figure.figsize'] = (6,6)
plt.rcParams['font.size']= 15
plt.rcParams['lines.linewidth'] = 3

#%% build BSB class

class BSB:
    """Brain-state-in-a-box class."""
    def __init__(self, feedback_factor=0.15, num_iterations=1000):
        self.feedback_factor = feedback_factor
        self.num_iterations = num_iterations
        
    def make_inputs(self):
        """Create input patterns."""
        input_patterns = np.array([[0.1, 0.2], [-0.2, 0.3], [-0.8, -0.4], [0.6, 0.1]])
        return input_patterns
    
    def activation_function(self, output_pattern):
        """Piecewise-linear activation function."""
        activation = np.piecewise(output_pattern, [output_pattern < -1, ((output_pattern >= -1) & (output_pattern <= 1)), output_pattern > 1], [-1, lambda output_pattern: output_pattern, 1])
        return activation
    
    def initialize_connection_weights(self):
        """Initialize connection weights."""
        weights = np.array([[0.035, -0.005],
                            [-0.005, 0.035]])
        return weights
    
    def run_model(self, input_patterns, weights):
        """Run BSB model."""
        num_input_patterns = len(input_patterns)
        output_patterns = [[] for _ in range(num_input_patterns)]
        
        for index, input_pattern in enumerate(input_patterns):
            iteration_count = 0
            while iteration_count < self.num_iterations:
                output_pattern = input_pattern + self.feedback_factor * np.dot(weights, input_pattern)
                input_pattern = self.activation_function(output_pattern)
                output_patterns[index].append(output_pattern)
                iteration_count += 1
        return output_patterns
    
    def show_attractor_basin_boundary(self, x_n, y_n, k_1, k_2, lambda_min, lambda_max, ax):
        """Show boundaries that separate distinct basins of attraction."""
        # first set of boundaries
        ax.plot(y_n, k_1 * y_n ** (lambda_min/lambda_max), c='k')
        ax.plot(-y_n, -k_1 * y_n ** (lambda_min/lambda_max), c='k')
        # second set of boundaries
        ax.plot(x_n, k_2 * x_n ** (lambda_max/lambda_min), c='k')
        ax.plot(-x_n, -k_2 * x_n ** (lambda_max/lambda_min), c='k')
        
    def show_attractor_basin(self, x_final_state, y_final_state, x_n, y_n, k_1, k_2, lambda_min, lambda_max, ax):
        """Show basin of attraction based on output pattern."""
        # top right corner
        if [x_final_state, y_final_state] == [1, 1]:
            ax.fill_between(x=y_n, y1=0, y2=1, color='grey', zorder=0)
            ax.fill_between(x=y_n, y1=0, y2=k_1 * y_n ** (lambda_min/lambda_max), color='white')
            ax.fill_between(x=x_n, y1=1, y2=k_2 * x_n ** (lambda_max/lambda_min), color='white')
        # top left corner
        if [x_final_state, y_final_state] == [-1, 1]:
            ax.fill_between(x=-y_n, y1=0, y2=1, color='grey', zorder=0)
            ax.fill_between(x=x_n,  y1=1, y2=k_2 * x_n ** (lambda_max/lambda_min), color='grey', zorder=0)
            ax.fill_between(x=-y_n, y1=0, y2=-k_1 * y_n ** (lambda_min/lambda_max), color='grey', zorder=0)
        # bottom left corner
        if [x_final_state, y_final_state] == [-1, -1]:
            ax.fill_between(x=-y_n, y1=0,  y2=-1, color='grey', zorder=0)
            ax.fill_between(x=-x_n, y1=-1, y2=-k_2 * x_n ** (lambda_max/lambda_min), color='white')
            ax.fill_between(x=-y_n, y1=0,  y2=-k_1 * y_n ** (lambda_min/lambda_max), color='white')
        # bottom right corner
        if [x_final_state, y_final_state] == [1, -1]:
            ax.fill_between(x=y_n,  y1=0,  y2=-1, color='grey', zorder=0)
            ax.fill_between(x=y_n,  y1=-1, y2=k_1 * y_n ** (lambda_min/lambda_max), color='grey', zorder=0)
            ax.fill_between(x=-x_n, y1=-1, y2=-k_2 * x_n ** (lambda_max/lambda_min), color='grey', zorder=0)
            
#%% instantiate BSB class
model = BSB()

#%% create input patterns
input_patterns = model.make_inputs()

#%% initialize connection weights
weights = model.initialize_connection_weights()

#%% train BSB model
output_patterns = model.run_model(input_patterns, weights)

#%% eigenvectors and corresponding eigenvalues of the two dimensional system
eigenvalues, eigenvectors = np.linalg.eigh(weights)
lambda_min = eigenvalues[0].round(2)
lambda_max = eigenvalues[1].round(2)
# determines the intersection with the boundary four (unstable) equilibrium points
x_value = lambda_max * np.sqrt(2) / (lambda_min + lambda_max)
# unstable equilibrium points
x_0 = 1-x_value
y_0 = 1
# x and y coordinates
n_points = 100
x_n = np.linspace(0, x_0, n_points)
y_n = np.linspace(0, y_0, n_points)
# attractor coefficients
k_1 = x_0/y_0**(lambda_min/lambda_max)
k_2 = y_0/x_0**(lambda_max/lambda_min)

#%% make figures
cwd = os.getcwd()                                                               # get current working directory
fileName = 'images'                                                             # specify filename

# filepath and directory specifications
if os.path.exists(os.path.join(cwd, fileName)) == False:                        # if path does not exist
    os.makedirs(fileName)                                                       # create directory with specified filename
    os.chdir(os.path.join(cwd, fileName))                                       # change cwd to the given path
    cwd = os.getcwd()                                                           # get current working directory
else:
    os.chdir(os.path.join(cwd, fileName))                                       # change cwd to the given path
    cwd = os.getcwd()                                                           # get current working directory

# figure 1 - activation function
fig, ax = plt.subplots()
output = np.linspace(-2.0, 2.0, 1000)
activation = BSB().activation_function(output)
ax.plot(output, activation)
ax.set_xticks([-1, 1])
ax.set_yticks([-1, 1])
ax.axvline(x=0, color='k', ls='--', lw=2)
ax.axhline(y=0, color='k', ls='--', lw=2)
ax.set_xlabel('y')
ax.set_ylabel('f(y)')
ax.set_title('piecewise-linear activation function')
fig.tight_layout()
fig.savefig(os.path.join(os.getcwd(), 'figure_1'))

# figure 2 - two neuron BSB model operating under 4 different initial conditions
fig, ax = plt.subplots(nrows=2, ncols=2)
ax = ax.ravel()
for i in range(len(input_patterns)):
    # trajectory of output patterns
    x = np.vstack(output_patterns[i])[::50, 0]
    y = np.vstack(output_patterns[i])[::50, 1]
    # final state of output pattern
    x_final_state = x[-1].astype(int)
    y_final_state = y[-1].astype(int)
    # show trajectory of output patterns
    ax[i].scatter(x, y, s=100)
    ax[i].scatter(x_final_state, y_final_state, s=150, color='r')
    ax[i].plot(x, y, color='k')
    # show basin of attraction boundary
    model.show_attractor_basin_boundary(x_n, y_n, k_1, k_2, lambda_min, lambda_max, ax=ax[i])
    # show basin of attraction
    model.show_attractor_basin(x_final_state, y_final_state, x_n, y_n, k_1, k_2, lambda_min, lambda_max, ax=ax[i])
    # add vertical and horizontal lines
    ax[i].axvline(x=0, color='k', ls='-', lw=2)
    ax[i].axhline(y=0, color='k', ls='-', lw=2)
    # set xticks and yticks
    ax[i].set_xticks([])
    ax[i].set_yticks([])
fig.suptitle('four distinct basins of attraction')
fig.tight_layout()
fig.savefig(os.path.join(os.getcwd(), 'figure_2'))


#%% remove variables
del fig, i, ax, n_points, x_value, output, activation, fileName
del x, x_0, x_n, x_final_state
del y, y_0, y_n, y_final_state
del lambda_min, lambda_max
del k_1, k_2
