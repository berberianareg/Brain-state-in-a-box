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
  There are four distinct basins of attraction.
  
  Starting at some initial point of the square, positive feedback
  will drive the system towards one of the four corners.
  
"""

#%% import libraries and modules
import numpy as np  
import matplotlib.pyplot as plt
import os

#%% figure parameters
plt.rcParams['figure.figsize'] = (6,6)
plt.rcParams['font.size']= 15
plt.rcParams['lines.linewidth'] = 2

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
        max_eigenvector = np.array([[1], [-1]])
        max_eigenvalue = 0.04

        min_eigenvector = np.array([[1], [1]])
        min_eigenvalue = 0.03
        
        max_weights = max_eigenvalue * 0.5 * np.dot(max_eigenvector, max_eigenvector.T)
        min_weights = min_eigenvalue * 0.5 * np.dot(min_eigenvector, min_eigenvector.T)
        
        weights = max_weights + min_weights
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
                        
    def plot_activation_function(self):
        """Plot piecewise-linear activation function."""
        fig, ax = plt.subplots()
        output = np.linspace(-2.0, 2.0, 1000)
        activation = self.activation_function(output)
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
        
    def plot_attractor_basin_boundary(self, lambda_min, lambda_max, ax):
        """Plot boundaries that separate distinct basins of attraction."""
        # attractor intersections
        x0 = (lambda_max * np.sqrt(2)) / (lambda_min + lambda_max)
        y0 = (lambda_min * np.sqrt(2)) / (lambda_min + lambda_max)
        # attractor coefficient
        k = y0 / x0 ** (lambda_max/lambda_min)
        # x-coordinates
        x = np.linspace(-1, 1, 100)
        # y-coordinates
        y_pos =  k * np.sign(x) * abs(x) ** (lambda_max/lambda_min)
        y_neg = -k * np.sign(x) * abs(x) ** (lambda_max/lambda_min)
        # rotation angle
        rotation_angle = np.pi/4
        # rotation matrix
        counterclockwise_rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                                     [np.sin(rotation_angle),  np.cos(rotation_angle)]])
        # rotate attractor basins
        x_bounary_pos, y_bounary_pos = np.dot(counterclockwise_rotation_matrix, np.array([x, y_pos]))
        x_bounary_neg, y_bounary_neg = np.dot(counterclockwise_rotation_matrix, np.array([x, y_neg]))
        # plot boundaries
        ax.plot(x_bounary_pos, y_bounary_pos, color='k')
        ax.plot(x_bounary_neg, y_bounary_neg, color='k')
        
    def plot_bsb_dynamics(self, lambda_min, lambda_max):
        """Plot two BSB neurons operating under 4 different initial conditions."""
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
            self.plot_attractor_basin_boundary(lambda_min, lambda_max, ax=ax[i])
            # add vertical and horizontal lines
            ax[i].axvline(x=0, color='k', ls='-', lw=2)
            ax[i].axhline(y=0, color='k', ls='-', lw=2)
            # set xticks and yticks
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].set_ylim(-1.1, 1.1)
            ax[i].set_xlim(-1.1, 1.1)
                        
        fig.suptitle('four distinct basins of attraction')
        fig.tight_layout()
        fig.savefig(os.path.join(os.getcwd(), 'figure_2'))
        
#%% instantiate BSB class
model = BSB()

#%% create input patterns
input_patterns = model.make_inputs()

#%% initialize connection weights
weights = model.initialize_connection_weights()

#%% train BSB model
output_patterns = model.run_model(input_patterns, weights)

#%% get eigenvectors and corresponding eigenvalues of the two dimensional system
eigenvalues, eigenvectors = np.linalg.eigh(weights)
lambda_min = eigenvalues[0].round(2)
lambda_max = eigenvalues[1].round(2)

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

# plot activation function
model.plot_activation_function()

# plot bsb dynamics
model.plot_bsb_dynamics(lambda_min, lambda_max)

