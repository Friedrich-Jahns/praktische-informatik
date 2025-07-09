import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

if True:
    N = 50
    grid = np.random.choice([-1, 1], size=(N, N))
    print()

if False:
    N = 200
    grid = np.ones((N, N))

#---------------------Start params---------------------#


T = 0.00001
J = 1
B = 0


# for _ in range(30):
#     metropolis_step(grid, T, J, B)

#---------------------Start params---------------------#

mag = []
eng = []
T_plot = []

fig, axes = plt.subplots(2,2,figsize=(6, 6))
img = axes[0,0].imshow(grid, cmap="Reds", interpolation="nearest", animated=True)

tp, = axes[0,1].plot([], [], 'r-', label="Temperatur")
p1, = axes[1,0].plot([], [], 'r-', label="Magnetisierung")
p2, = axes[1,1].plot([], [], 'r-', label="Energie")


def update(frame):
    global T


    # ham.append(hamiltonian(grid,J,B))
    metropolis_step(grid, T,J,B)
    img.set_array(grid)

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
    return [img,tp,p1,p2]

ani = FuncAnimation(fig, update, frames=10000, interval=200, blit=True,repeat=False)
# ani.save('ising_model_simulation.gif', writer='imagemagick', fps=10)

axes[0,0].axis('off')

# axes[0,1].axis('off')



axes[1,0].set_title('Magnetisierung')
axes[1,0].set_aspect('auto')

axes[1,1].set_title('Energie')
axes[1,1].set_aspect('auto')



plt.tight_layout()
plt.show()
# plt.subplot(1,2,1)
# plt.plot(ham)
# plt.subplot(1,2,2)
# plt.plot(mag)
# plt.show()
