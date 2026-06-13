import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import time
from pathlib import Path
import os
import multiprocessing as mp
from lattices import *

rng = np.random.default_rng()     # seed 123 for testing


### DEFINING FUNCTIONS ###

def backtrack_num(t, A=0.8):    # returns num b/w 1 and t with exp decay probability (larger A = favour lower number)
    """
    Use t as an upper bound on how many walks to backtrack by
    and determined using inverse transform sampling method
    """
    u = rng.random()
    if t > 1000 and u < 0.0001: return 500        # 0.1% of the time it'll kick it out of a winding configuration
    if t > 200 and u < 0.001: return 100        # 0.1% of the time it'll kick it out of a winding configuration
    inv_cdf = -np.log(1 - u * (1 - np.exp(-A * t))) / A
    backtrack_num = np.ceil(inv_cdf).astype(int)
    return backtrack_num


def check_v_eq(v1, v2):    # checking if 2 vectors are the same up to machine precision
    epsilon = 1e-9
    if norm(v1 - v2) < epsilon:
        return True
    return False


def check_v_in_vlist(v1, vlist):
    for v2 in vlist:
        if check_v_eq(v1, v2):
            return True
    return False


def get_SAW_naive_v1(vcv, vti, N):                  # works on any lattice (accounts for floating point precision but uses lists)
    """
    Can only reach small sized lattices with high variance of time required (N_max=2000 with times up to 2 minutes)
    """
    d_lattice = len(vcv[0][0])                      # using the dimension of the first vcv vector for the dimension of x
    x0 = np.array([0 for _ in range(d_lattice)])    # origin point
    x = [x0]                                        # list of 1d arrays
    t = 0
    vert_type = [0]
    n_iter = 0
    n_collision = 0

    start = time.time()

    while t < N - 1:
        n_iter += 1
        print("LENGTH:", len(x))

        current_vert_type = vert_type[-1]
        num_v_options = len(vcv[current_vert_type])
        idx = np.floor(rng.random() * num_v_options).astype(int)
        v = vcv[current_vert_type][idx]
        x_new = x[t] + v

        # updating the state before checking (will undo if theres a conflict)
        x.append(x_new)
        t += 1
        new_vert_type = vti[current_vert_type][idx]
        vert_type.append(new_vert_type)

        if check_v_in_vlist(x[t], x[:-1]):          # collision occurs
            t_reverse = backtrack_num(t)
            for _ in range(t_reverse):              # remove last t_reverse positions
                x.pop(-1)
                vert_type.pop(-1)
            t -= t_reverse
            n_collision += 1
            continue
    
    end = time.time()

    time_elapsed = end - start

    x_arr = np.array(x)
    return x_arr, n_iter, time_elapsed, n_collision


def get_SAW_naive_v2(
        vcv=VCV_SQUARE_2D, 
        vti=VTI_SQUARE_2D, 
        N=10000
        ):        # can only be used on integer lattices (due to floating point precision since we use sets here)
    """
    Can reach sizes of 10000 max very quickly (after that length its luck based how long it will take).
    """
    d_lattice = len(vcv[0][0])            # using the dimension of the first vcv vector for the dimension of x
    x0 = tuple(0 for _ in range(d_lattice))    # origin point (tuple, so that it can be hashed)
    x = [x0]
    visited = set([x0])
    t = 0
    vert_type = [0]
    n_iter = 0
    n_collision = 0

    start = time.time()

    while t < N:    # need to add N vertices for a total of N + 1 vetices 
        n_iter += 1
        # print("N_VERTICES:", len(x))

        current_vert_type = vert_type[-1]
        num_v_options = len(vcv[current_vert_type])
        idx = np.floor(rng.random() * num_v_options).astype(int)
        v = vcv[current_vert_type][idx]

        x_new = tuple(x[-1][i] + v[i] for i in range(d_lattice))

        # updating the state before checking (will undo if theres a conflict)
        x.append(x_new)
        t += 1
        new_vert_type = vti[current_vert_type][idx]
        vert_type.append(new_vert_type)
        
        if x[-1] in visited:                        # collision occurs
            t_reverse = backtrack_num(t)            # how far to backtrack by
            for i in range(t_reverse):              # remove last t_reverse positions
                if i > 0:                           # b/c the last added position never got added to visited
                    visited.remove(x[-1])
                x.pop(-1)
                vert_type.pop(-1)
            t -= t_reverse
            n_collision += 1
            continue

        visited.add(x_new)

        # if n_iter > 5e5:      # if it runs for too long (~20 seconds) (SAW is very tangled up) (designed for N=10000 in 2D)
        #     return get_SAW_naive_v2(vcv, vti, N)
    
    end = time.time()

    time_elapsed = end - start

    x_arr = np.array([np.array(xt) for xt in x])
    return x_arr, n_iter, time_elapsed, n_collision


def plot_SAW(x):
    d = len(x[0])
    if d == 2:
        plt.plot(x[:, 0], x[:, 1])
    if d == 3:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot(x[:, 0], x[:, 1], x[:, 2])

    plt.show()


### SAW DATA PROCESSING FUNCTIONS ###


def ete_dist(x):
    return (norm(x[-1] - x[0])) ** 2


# For large SAWs these values take much longer to calculate due to the loops
def rad_gyr(x):
    x_mean = np.mean(x, axis=0)
    val = 0
    for i in range(len(x)):
        val += norm(x[i] - x_mean) ** 2
    return val / len(x)


def mean_ete_dist(x):
    val = 0
    for i in range(len(x)):
        val += norm(x[i] - x[0]) ** 2 + norm(x[-1] - x[i]) ** 2
    return val / (2 * len(x))


def calc_forall_lengths(x: np.ndarray, func: callable) -> np.ndarray:
    """
    Used to calculate statistical values for all truncations of a SAW.
    """
    values = []       # in ascending order
    N_len = len(x)    # n_vertices - 1 = steps
    for n in range(1, N_len):
        x_truncated = x[:n]
        val = func(x_truncated)
        values.append(val)
        # print("PROCESSED:", n)
    return np.array(values)


def save_data(saw_data: tuple, id: int, N_max: int, lattice_type: str) -> None:
    """
    This will append the inputted saw_data, from a generation of a set length (in this case 10,000
    for testing purposes). 
    """
    x_arr = saw_data[0]
    ete_dists_new = calc_forall_lengths(x_arr, ete_dist)
    
    saw_dir_path = Path("./SAW_data_files")
    if not saw_dir_path.is_dir():      # checking if directory exits and making it if not
        saw_dir_path.mkdir(exist_ok=True)

    try:                               # checking if the file exists in the directory
        with np.load(f"./SAW_data_files/saw_data_id_{id}_Nmax_{N_max}_{lattice_type}.npz") as previous_data:
            N_old = previous_data["N"]
            sm_ete_dists_old = previous_data["sm_ete_dists"]
            sv_ete_dists_old = previous_data["sv_ete_dists"]
    except FileNotFoundError:          # the file hasn't been made yet
        np.savez(file=f"./SAW_data_files/saw_data_id_{id}_Nmax_{N_max}_{lattice_type}.npz", 
                 N=1, 
                 sm_ete_dists=ete_dists_new,
                 sv_ete_dists=np.zeros(shape=np.shape(ete_dists_new)))
        return None

    # applying incremental updates to sample means and sample variances 
    N_new = N_old + 1
    sm_ete_dists_new = sm_ete_dists_old + (ete_dists_new - sm_ete_dists_old) / N_new
    sv_ete_dists_new = (sv_ete_dists_old + (ete_dists_new - sm_ete_dists_new) * (ete_dists_new - sm_ete_dists_old)) / (N_new - 1) 

    # adding the new data to a tmp file
    np.savez(file=f"./SAW_data_files/saw_data_id_{id}_Nmax_{N_max}_{lattice_type}_tmp.npz",
             N=N_new,
             sm_ete_dists=sm_ete_dists_new,
             sv_ete_dists=sv_ete_dists_new)
    # atomic operation safe for when code stops running in the middle of a file override 
    os.replace(src=f"./SAW_data_files/saw_data_id_{id}_Nmax_{N_max}_{lattice_type}_tmp.npz",
               dst=f"./SAW_data_files/saw_data_id_{id}_Nmax_{N_max}_{lattice_type}.npz")


def accumulate_data(vcv, vti, N, id, lattice_type):
    data = get_SAW_naive_v2(N=N, vcv=vcv, vti=vti)
    save_data(saw_data=data, id=id, N_max=N, lattice_type=lattice_type)


def plot_stats(stat_arr):
    count = np.arange(1, len(stat_arr) + 1)
    plt.plot(count, stat_arr)
    plt.show()


def pool_stats(N_max: int, lattice_type: str):
    """
    Iterates through all npz files of matching N_max and pools their statistics into one file with id=0.
    """
    p = Path("./SAW_data_files/")
    saw_paths = list(p.glob(f"*Nmax_{N_max}_{lattice_type}.npz"))

    # pooled statistics
    N_pool = 0
    sm_ete_dists_pool = np.zeros(N_max)
    sv_ete_dists_pool = np.zeros(N_max)

    # pool statistics from all separate files
    for path in saw_paths:
        with np.load(path) as data:
            N_old = data["N"]
            sm_ete_dists_old = data["sm_ete_dists"]
            sv_ete_dists_old = data["sv_ete_dists"]

        if N_old < 2:   # to avoid 0 division errors 
            # ADD SOME CODE TO DELETE THE FILE B/C ITS SMALL
            continue 

        w11 = (N_pool / (N_pool + N_old)) * sm_ete_dists_pool
        w21 = (N_old / (N_pool + N_old)) * sm_ete_dists_old
        w12 = ((N_pool - 1) / (N_pool + N_old - 2)) * sv_ete_dists_pool
        w22 = ((N_old - 1) / (N_pool + N_old - 2)) * sv_ete_dists_old

        sm_ete_dists_pool = w11 + w21
        sv_ete_dists_pool = w12 + w22
        N_pool += N_old
    
    np.savez(f"./SAW_data_files/saw_data_id_0_Nmax_{N_max}_{lattice_type}.npz",
             N=N_pool,
             sm_ete_dists=sm_ete_dists_pool,
             sv_ete_dists=sv_ete_dists_pool)
        

def delete_tmp():
    p = Path("./SAW_data_files/")
    tmp_paths = list(p.glob("*tmp.npz"))
    for path in tmp_paths:
        os.remove(path)
    


### RUNNING THE CODE ###

# Max for 2D = 10000 (speed diminishes quickly for bigger N)
# No max for 3D, appears to steadily grow at a rate of 12,000 vertices/second (diminishes negligibly over larger N)

delete_tmp()   # running this to always remove temporary files 

pool_stats(N_max=10000, lattice_type="3dsq")

# max_n = 34
# for n in range(max_n):
#     accumulate_data(N=10000, vcv=VCV_SQUARE_3D, vti=VTI_SQUARE_3D, id=6157, lattice_type="3dsq")
#     print(f"{n} / {max_n} done")

data = np.load("./SAW_data_files/saw_data_id_0_Nmax_10000_3dsq.npz")
print(data["N"])
print(np.shape(data["sm_ete_dists"]))
print(np.shape(data["sv_ete_dists"]))
print(data["sm_ete_dists"][:10])
print(data["sv_ete_dists"][:10])

plot_stats(data["sm_ete_dists"])
plot_stats(data["sv_ete_dists"])