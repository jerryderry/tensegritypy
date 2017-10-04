# tensegritypy
TensegrityPy is a project collecting functions related to tensegrity structures for scientific research. 

**This is an ongoing project**.

# Examples
## Create a tensegrity
A tensegrity is created by providing three parameters: nodal coordinates, connectivity and member types. Nodal coordinates and connectivity are Numpy arrays, while member types is a list of enum tensegrity.Member.

The coordinates of each node form a row vector. For a 3D structure, the 1st column of the nodal coordinate matrix corresponds to x coordinate, the 2nd and 3rd correpond to y and z coordinates, respectively.

Connectivity is a matrix with dimension b by 2, where b is the number of members. Each row specifies which nodes the corresponding member connects. The node number counts from 1. For example, if Member 3 connects Node 1 and Node 2, then the 3rd row (index 2) should be [1, 2]. One should put the node number with smaller value at the first column.

Member types are specified with an enum whose members are `CABLE` and `BAR`.

```python
import numpy as np

import tensegrity

config = {'nodal_coordinates': np.array([[0.2588, 0.9659, 0.5],      # Node 1
                                       [-0.9659, -0.2588, 0.5],    # Node 2
                                       [0.7071, -0.7071, 0.5],     # Node 3
                                       [0.2588, -0.9659, -0.5],    # Node 4
                                       [0.7071, 0.7071, -0.5],     # Node 5
                                       [-0.9659, 0.2588, -0.5]]),  # Node 6
          'connectivity': np.array([[1, 2],
                                  [2, 3],
                                  [1, 3],
                                  [4, 5],
                                  [5, 6],
                                  [4, 6],
                                  [1, 5],
                                  [2, 6],
                                  [3, 4],
                                  [1, 4],
                                  [2, 5],
                                  [3, 6]]),
          'member_types': [tensegrity.Member.CABLE, tensegrity.Member.CABLE, tensegrity.Member.CABLE, tensegrity.Member.CABLE,
                         tensegrity.Member.CABLE, tensegrity.Member.CABLE, tensegrity.Member.CABLE, tensegrity.Member.CABLE,
                         tensegrity.Member.CABLE, tensegrity.Member.BAR, tensegrity.Member.BAR, tensegrity.Member.BAR]}
                         
structure = tensegrity.Tensegrity(**config)
