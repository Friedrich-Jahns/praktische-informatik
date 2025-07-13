import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Configurations:
# 1:Plot Grid(random start)
# 2:Plot Grid(random start), B>0
# 3:Plot Grid(random start),Temperature, Magnetiziation  and Energy
# 4:Plot Grid(random start), B>0,Temperature, Magnetiziation  and Energy
config = 4


#---------------------Fuktionen---------------------#


def neighbors(i, j, mat):
    N = mat.shape[0]
    return np.array([
        mat[(i+1)%N, j],
        mat[(i-1)%N, j],
        mat[i, (j+1)%N],
        mat[i, (j-1)%N]
    ])


def denergy(spin,neighbors,J,B=0):
    return 2*spin*(J*sum(neighbors)+B)

def hamiltonian(mat, J=1.5,B=0):
    N = mat.shape[0]
    E = 0
    for i in range(N):
        for j in range(N):
            s = mat[i, j]
            right = mat[i, (j+1)%N]
            down  = mat[(i+1)%N, j]
            E -= J * s * right
            E -= J * s * down
            E -= B * s
    return E

def magnetization(mat):
    return np.sum(mat)/mat.shape[0]**2

def metropolis_step(mat,T,J,B):
    N = mat.shape[0]
    for n in range(N**2):
        i, j = np.random.randint(0, N), np.random.randint(0, N)
        dE = denergy(mat[i,j],neighbors(i,j,mat),J,B)
        # rand_flip = np.random.rand() < np.exp(-dE / T)
        if dE<0 or np.random.rand() < np.exp(-dE / T):
            mat[i,j] = mat[i,j]*-1

def update(frame):
    global T


    # ham.append(hamiltonian(grid,J,B))
    metropolis_step(grid, T,J,B)
    img.set_array(grid)

    if config == 3 or config == 4:
        T_plot.append(T)
        tp.set_data(np.arange(len(T_plot)), T_plot)
        axes[0,1].relim()
        axes[0,1].autoscale_view()


        mag.append(magnetization(grid))
        p1.set_data(np.arange(len(mag)), mag)
        axes[1,0].relim()
        axes[1,0].autoscale_view()

        eng.append(hamiltonian(grid,J=J,B=B))
        p2.set_data(np.arange(len(eng)), eng)
        axes[1,1].relim()
        axes[1,1].autoscale_view()


    if frame % 100 == 0:#frame >=39 and
        T += .2

    # frame_count += 1
    if config == 3 or config ==4:
        return [img,tp,p1,p2]
    if config == 2 or config ==1:
        return [img]
 


#---------------------Start config---------------------#

if False:
    N = 200 
    spins = np.ones((N, N), dtype=int)  # Alles +1

    center = N // 2
    radius = N // 4

    for i in range(N):
        for j in range(N):
            if (i - center)**2 + (j - center)**2 < radius**2:
                spins[i, j] = -1
    grid = spins

if config == 1 or config == 2 or config == 3 or config == 4:
    N = 100
    grid = np.random.choice([-1, 1], size=(N, N))
    print()

if False:
    N = 200
    grid = np.ones((N, N))

#---------------------Start params---------------------#

if config == 1 or config == 3:
    T = 0.00001
    J = 1
    B = 0

if config ==2 or config == 4:
    T = 0.00001
    J = 1
    B = 0.01

#---------------------Plot---------------------#


mag = []
eng = []
T_plot = []


if config == 1 or config ==2:
    fig, axes = plt.subplots(1,1,figsize=(6, 6))
    img = axes.imshow(grid, cmap="Reds", interpolation="nearest", animated=True)

    ani = FuncAnimation(fig, update, frames=10000, interval=200, blit=True,repeat=False)

    axes.axis('off')

    plt.tight_layout()
    plt.show()


if config == 3 or config == 4:
    fig, axes = plt.subplots(2,2,figsize=(6, 6))
    img = axes[0,0].imshow(grid, cmap="Reds", interpolation="nearest", animated=True)

    tp, = axes[0,1].plot([], [], 'r-', label="Temperatur")
    p1, = axes[1,0].plot([], [], 'r-', label="Magnetisierung")
    p2, = axes[1,1].plot([], [], 'r-', label="Energie")


    ani = FuncAnimation(fig, update, frames=10000, interval=200, blit=True,repeat=False)

    axes[0,0].axis('off')

    axes[1,0].set_title('Magnetisierung')
    axes[1,0].set_aspect('auto')

    axes[1,1].set_title('Energie')
    axes[1,1].set_aspect('auto')


    plt.tight_layout()
    plt.show()
