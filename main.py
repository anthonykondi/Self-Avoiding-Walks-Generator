import numpy as np
from numpy import sqrt
from numpy.linalg import norm
import matplotlib.pyplot as plt
import time

rng = np.random.default_rng()     # seed 123 for testing

### VARIABLES ###
"""
Consist of VCVs (vertex connection vectors), and VTIs (vertex transition instructions).
VCV: 0-idx = starting type, 1-idx = connection vector
VTI: 0-idx = starting type, 1-idx = transitioning type
"""

# 2D LATTICES

VCV_SQUARE_2D = [[[1, 0], [0, 1], [-1, 0], [0, -1]]]
VCV_SQUARE_2D = [[np.array(vector) for vector in list] for list in VCV_SQUARE_2D]
VTI_SQUARE_2D = [[0, 0, 0, 0]]

VCV_TRIANGLE_2D = [[[1, 0], [1 / 2, sqrt(3) / 2], [-1 / 2, sqrt(3) / 2],
                    [-1, 0], [-1 / 2, -sqrt(3) / 2], [1 / 2, -sqrt(3) / 2]]]
VCV_TRIANGLE_2D = [[np.array(vector) for vector in list] for list in VCV_TRIANGLE_2D]
VTI_TRIANGLE_2D = [[0, 0, 0, 0, 0, 0]]

VCV_HONEYCOMB_2D = [[[0, 1], [-sqrt(3) / 2, -1 / 2], [sqrt(3) / 2, -1 / 2]], 
                    [[0, -1], [-sqrt(3) / 2, 1 / 2], [sqrt(3) / 2, 1 / 2]]]
VCV_HONEYCOMB_2D = [[np.array(vector) for vector in list] for list in VCV_HONEYCOMB_2D]
VTI_HONEYCOMB_2D = [[1, 1, 1], 
                    [0, 0, 0]]

VCV_OCTAGON_2D = [[[1, 0], [-sqrt(2) / 2, sqrt(2) / 2], [-sqrt(2) / 2, -sqrt(2) / 2]], 
                  [[0, 1], [sqrt(2) / 2, -sqrt(2) / 2], [-sqrt(2) / 2, -sqrt(2) / 2]],
                  [[-1, 0], [sqrt(2) / 2, sqrt(2) / 2], [sqrt(2) / 2, -sqrt(2) / 2]],
                  [[0, -1], [-sqrt(2) / 2, sqrt(2) / 2], [sqrt(2) / 2, sqrt(2) / 2]]]
VCV_OCTAGON_2D = [[np.array(vector) for vector in list] for list in VCV_OCTAGON_2D]
VTI_OCTAGON_2D = [[2, 1, 3],
                  [3, 0, 2],
                  [0, 1, 3],
                  [1, 2, 0]]

# 3D LATTICES

VCV_SQUARE_3D = [[[1, 0, 0], [0, 1, 0], [-1, 0, 0], 
                  [0, -1, 0], [0, 0, 1], [0, 0, -1]]]
VCV_SQUARE_3D = [[np.array(vector) for vector in list] for list in VCV_SQUARE_3D]
VTI_SQUARE_3D = [[0, 0, 0, 0, 0, 0]]

VCV_TRIANGLE_3D = [[[1, 0, 0], [1 / 2, sqrt(3) / 2, 0], [-1 / 2, sqrt(3) / 2, 0], [-1, 0, 0], 
                    [-1 / 2, -sqrt(3) / 2, 0], [1 / 2, -sqrt(3) / 2, 0], [0, 0, 1], [0, 0, -1]]]
VCV_TRIANGLE_3D = [[np.array(vector) for vector in list] for list in VCV_TRIANGLE_3D]
VTI_TRIANGLE_2D = [[0, 0, 0, 0, 0, 0, 0, 0]]

VCV_HONEYCOMB_3D = [[[0, 1, 0], [-sqrt(3) / 2, -1 / 2, 0], [sqrt(3) / 2, -1 / 2, 0], [0, 0, 1], [0, 0, -1]], 
                    [[0, -1, 0], [-sqrt(3) / 2, 1 / 2, 0], [sqrt(3) / 2, 1 / 2, 0], [0, 0, 1], [0, 0, -1]]]
VCV_HONEYCOMB_3D = [[np.array(vector) for vector in list] for list in VCV_HONEYCOMB_3D]
VTI_HONEYCOMB_3D = [[1, 1, 1, 0, 0], 
                    [0, 0, 0, 1, 1]]

VCV_OCTAGON_3D = [[[1, 0, 0], [-sqrt(2) / 2, sqrt(2) / 2, 0], [-sqrt(2) / 2, -sqrt(2) / 2, 0], [0, 0, 1], [0, 0, -1]], 
                  [[0, 1, 0], [sqrt(2) / 2, -sqrt(2) / 2, 0], [-sqrt(2) / 2, -sqrt(2) / 2, 0], [0, 0, 1], [0, 0, -1]],
                  [[-1, 0, 0], [sqrt(2) / 2, sqrt(2) / 2, 0], [sqrt(2) / 2, -sqrt(2) / 2, 0], [0, 0, 1], [0, 0, -1]],
                  [[0, -1, 0], [-sqrt(2) / 2, sqrt(2) / 2, 0], [sqrt(2) / 2, sqrt(2) / 2, 0], [0, 0, 1], [0, 0, -1]]]
VCV_OCTAGON_3D = [[np.array(vector) for vector in list] for list in VCV_OCTAGON_3D]
VTI_OCTAGON_3D = [[2, 1, 3, 0, 0],
                  [3, 0, 2, 1, 1],
                  [0, 1, 3, 2, 2],
                  [1, 2, 0, 3, 3]]

### DEFINING FUNCTIONS ###

### SAW GENERATION FUNCTIONS ###


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
        print("N_VERTICES:", len(x))

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


def rad_gyr(x):        # Note: this and mean end to end distance take way longer to calcualte
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
        print("PROCESSED:", n)
    return np.array(values)


### RUNNING THE CODE ###

# Max for 2D = 10000 (speed diminishes quickly for bigger N)
# No max for 3D, appears to steadily grow at a rate of 12,000 vertices/second (diminishes negligibly over larger N)

data = get_SAW_naive_v2(N=1000000, vcv=VCV_SQUARE_3D, vti=VTI_SQUARE_3D)

start_t = time.time()
processed_data = calc_forall_lengths(data[0], ete_dist)
end_t = time.time()

print("GENERATION TIME: ", round(data[2], 2))
print("CALCULATION TIME:", round(end_t - start_t), 2)

count_arr = np.arange(1, len(data[0]))

plot_SAW(data[0])
plt.plot(count_arr, processed_data)
plt.show()
