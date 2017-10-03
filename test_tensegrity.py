from unittest import TestCase
from tensegrity import Tensegrity, Member
import numpy as np


class TestTensegrity(TestCase):
    def setUp(self):
        nodal_coordinates = np.array([[0.2588, 0.9659, 0.5],  # Node 1
                                      [-0.9659, -0.2588, 0.5],  # Node 2
                                      [0.7071, -0.7071, 0.5],  # Node 3
                                      [0.2588, -0.9659, -0.5],  # Node 4
                                      [0.7071, 0.7071, -0.5],  # Node 5
                                      [-0.9659, 0.2588, -0.5]])  # Node 6
        connectivity = np.array([[1, 2],
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
                                 [3, 6]])
        member_types = [Member.CABLE, Member.CABLE, Member.CABLE, Member.CABLE,
                        Member.CABLE, Member.CABLE, Member.CABLE, Member.CABLE,
                        Member.CABLE, Member.BAR, Member.BAR, Member.BAR]
        self.structure = Tensegrity(nodal_coordinates, connectivity, member_types)

    def test_get_placement(self):
        p = np.array([0.2588, 0.9659, 0.5, -0.9659, -0.2588, 0.5,
                      0.7071, -0.7071, 0.5, 0.2588, -0.9659, -0.5,
                      0.7071, 0.7071, -0.5, -0.9659, 0.2588, -0.5])
        np.testing.assert_allclose(p, self.structure.get_placement())

    def test_fix_dofs(self):
        dofs_to_be_fixed = [10, 11, 12,  # Node 4
                            13, 14, 15,  # Node 5
                            16, 17, 18]  # Node 6
        self.structure.fix_dofs(dofs_to_be_fixed)
        self.assertEqual(self.structure.fixed_dofs, {10, 11, 12,
                                                     13, 14, 15,
                                                     16, 17, 18})
        self.assertEqual(self.structure.free_dofs, {1, 2, 3,
                                                    4, 5, 6,
                                                    7, 8, 9})

    def test_set_nodal_coordinates_from_placement_full(self):
        self.structure.set_nodal_coordinates_from_placement(np.array([1, 2, 3,
                                                                      4, 5, 6,
                                                                      7, 8, 9,
                                                                      10, 11, 12,
                                                                      13, 14, 15,
                                                                      16, 17, 18]))
        np.testing.assert_allclose(self.structure.nodal_coordinates, np.array([[1, 2, 3],
                                                                               [4, 5, 6],
                                                                               [7, 8, 9],
                                                                               [10, 11, 12],
                                                                               [13, 14, 15],
                                                                               [16, 17, 18]]))

    def test_set_nodal_coordinates_from_placement_partial(self):
        dofs_to_be_fixed = [10, 11, 12,  # Node 4
                            13, 14, 15,  # Node 5
                            16, 17, 18]  # Node 6
        self.structure.fix_dofs(dofs_to_be_fixed)
        self.structure.set_nodal_coordinates_from_placement(np.array([1, 2, 3,
                                                                      4, 5, 6,
                                                                      7, 8, 9]))
        np.testing.assert_allclose(self.structure.nodal_coordinates, np.array([[1, 2, 3],
                                                                               [4, 5, 6],
                                                                               [7, 8, 9],
                                                                               [0.2588, -0.9659, -0.5],
                                                                               [0.7071, 0.7071, -0.5],
                                                                               [-0.9659, 0.2588, -0.5]]))

    def test_get_incidence_matrix(self):
        np.testing.assert_array_equal(self.structure.get_incidence_matrix(), np.array([[1, -1, 0, 0, 0, 0],
                                                                                  [0, 1, -1, 0, 0, 0],
                                                                                  [1, 0, -1, 0, 0, 0],
                                                                                  [0, 0, 0, 1, -1, 0],
                                                                                  [0, 0, 0, 0, 1, -1],
                                                                                  [0, 0, 0, 1, 0, -1],
                                                                                  [1, 0, 0, 0, -1, 0],
                                                                                  [0, 1, 0, 0, 0, -1],
                                                                                  [0, 0, 1, -1, 0, 0],
                                                                                  [1, 0, 0, -1, 0, 0],
                                                                                  [0, 1, 0, 0, -1, 0],
                                                                                  [0, 0, 1, 0, 0, -1]]))

    def test_get_edm(self):
        np.testing.assert_allclose(self.structure.get_edm(), np.array([[0, 1.7321, 1.7321, 2.1753, 1.1260, 1.7321],
                                                                       [1.7321, 0, 1.7321, 1.7321, 2.1753, 1.1260],
                                                                       [1.7321, 1.7321, 0, 1.1260, 1.7321, 2.1753],
                                                                       [2.1753, 1.7321, 1.1260, 0, 1.7321, 1.7321],
                                                                       [1.1260, 2.1753, 1.7321, 1.7321, 0, 1.7321],
                                                                       [1.7321, 1.1260, 2.1753, 1.7321, 1.7321, 0]]),
                                   rtol=1e-4)

    def test_get_equilibrium_matrix(self):
        should_be = np.array([[7.0710678e-01,  0,  -2.5881905e-01,   0,   0,
                               0,  -3.9811261e-01,   0,   0,   0,   0,   0],
                              [7.0710678e-01,   0,   9.6592583e-01,   0,   0,
                               0,   2.2985042e-01,   0,   0,   8.8807383e-01,   0,   0],
                              [0,   0,   0,   0,   0,
                               0,   8.8807383e-01,   0,   0,   4.5970084e-01,   0,   0],
                              [-7.0710678e-01,  -9.6592583e-01,   0,   0,   0,
                               0,   0,   0,   0,   0,    -7.6909450e-01,   0],
                              [-7.0710678e-01,   2.5881905e-01,   0,   0,   0,
                               0,   0,  -4.5970084e-01,   0,   0,   -4.4403692e-01,   0],
                              [0,   0,   0,   0,   0,
                               0,   0,   8.8807383e-01,   0,   0,   4.5970084e-01,   0],
                              [0,   9.6592583e-01,   2.5881905e-01,   0,   0,
                               0,   0,   0,   3.9811261e-01,   0,   0,   7.6909450e-01],
                              [0,  -2.5881905e-01,  -9.6592583e-01,   0,   0,
                               0,   0,   0,   2.2985042e-01,   0,  0,  -4.4403692e-01],
                              [0,  0,  0,   0,   0,
                               0,   0,   0,   8.8807383e-01,   0,   0,   4.5970084e-01],
                              [0,   0,   0,  -2.5881905e-01,   0,
                               7.0710678e-01,   0,   0,  -3.9811261e-01,  0,   0,   0],
                              [0,   0,   0,  -9.6592583e-01,   0,
                               -7.0710678e-01,   0,   0,  -2.2985042e-01,  -8.8807383e-01,   0,   0],
                              [0,   0,   0,   0,   0,
                               0,   0,   0,  -8.8807383e-01,  -4.5970084e-01,   0,   0],
                              [0,   0,   0,   2.5881905e-01,   9.6592583e-01,
                               0,   3.9811261e-01,   0,   0,   0,   7.6909450e-01,   0],
                              [0,   0,   0,   9.6592583e-01,   2.5881905e-01,
                               0,  -2.2985042e-01,   0,   0,   0,   4.4403692e-01,   0],
                              [0,   0,   0,  0,   0,
                               0,  -8.8807383e-01,   0,   0,   0,   -4.5970084e-01,   0],
                              [0,   0,   0,   0,  -9.6592583e-01,
                               -7.0710678e-01,   0,  0,   0,   0,   0,  -7.6909450e-01],
                              [0,   0,   0,   0,  -2.5881905e-01,
                               7.0710678e-01,   0,   4.5970084e-01,   0,   0,   0,   4.4403692e-01],
                              [0,   0,   0,   0,  0,
                               0,   0,  -8.8807383e-01,   0,   0,   0,  -4.5970084e-01]])
        np.testing.assert_allclose(self.structure.get_equilibrium_matrix(False), should_be, rtol=1e-4)

        self.structure.fix_dofs([10, 11, 12, 13, 14, 15, 16, 17, 18])
        np.testing.assert_allclose(self.structure.get_equilibrium_matrix(), should_be[[0, 1, 2, 3, 4, 5, 6, 7, 8]],
                                   rtol= 1e-4)