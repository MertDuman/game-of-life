import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


import scipy.signal
def update(i, img, grid, fade, kernel):
    # Count neighbors using convolution
    neighbor_count = scipy.signal.convolve2d(grid, kernel, mode='same', boundary='wrap')
    
    newGrid = grid.copy()
    # Apply Conway's rules
    newGrid[(grid == ON) & ((neighbor_count < 2) | (neighbor_count > 3))] = OFF
    newGrid[(grid == OFF) & (neighbor_count == 3)] = ON
    
    # Decay the cells that are off
    fade[grid == OFF] = np.maximum(0, fade[grid == OFF] - FADE_AMOUNT)
    fade[newGrid == ON] = FADE

    img.set_data(np.minimum(1, newGrid + fade))
    grid[:] = newGrid[:]
    return img,


# Constants
ON = 1.0
OFF = 0.0
FADE = 0.4

N = 100
FPS = 30
P_ON = 0.3
FADE_OVER_FRAMES = 5
FADE_AMOUNT = 0.5 / FADE_OVER_FRAMES
updateInterval = (1 / FPS) * 1000

# Create a grid with random initial states
grid = np.random.choice([ON, OFF], N*N, p=[P_ON, 1 - P_ON]).reshape(N, N)
fade = np.zeros_like(grid)
fade[grid == ON] = FADE
kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

# Set up the figure and axis
fig, ax = plt.subplots()
fig.patch.set_alpha(0)  # Make the figure background transparent
ax.axis('off')
img = ax.imshow(grid, interpolation='none', cmap='gray')
ani = animation.FuncAnimation(fig, update, fargs=(img, grid, fade, kernel), interval=updateInterval, frames=500, blit=True)
# ani.save('conways_game_of_life.gif', writer='pillow', savefig_kwargs={'facecolor': 'none'})
plt.show()
