import numpy as np
from numpy import sqrt

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