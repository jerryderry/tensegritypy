import numpy as np
from enum import Enum, unique


@unique
class Member(Enum):
    BAR = 0
    CABLE = 1


class Tensegrity:
    """Tensegrity class is used for representing classical straight member tensegrity
    structures."""

    def __init__(self, nodal_coordinates: np.ndarray, connectivity: np.ndarray, member_types: list):
        self.nodal_coordinates = nodal_coordinates
        self.connectivity = connectivity
        self.member_types = member_types
        self.number_of_nodes = nodal_coordinates.shape[0]
        self.dimension = nodal_coordinates.shape[1]
        self.number_of_members = connectivity.shape[0]
        self.free_dofs = set(range(1, self.dimension * self.number_of_nodes + 1))
        self.fixed_dofs = set()

    def get_placement(self) -> np.ndarray:
        """Get the current placement vector of the structure.
        The coordinate components of each node are grouped together, i.e., it is
        x_1, y_1, z_1, x_2, y_2, z_2, ..., x_n, y_n, z_n

        :return: the placement vector
        :rtype: np.ndarray
        """
        return self.nodal_coordinates.flatten()

    def set_nodal_coordinates_from_placement(self, placement: np.ndarray) -> None:
        """Set the structure's nodal coordinates from a placement vector.
        The placement vector can only contain values for free DOFs if the list
        for fixed_dof is not empty.

        :param placement: the placement vector
        :type placement: np.ndarray
        """
        if placement.shape[0] == self.number_of_nodes * self.dimension:
            self.nodal_coordinates = placement.reshape((self.number_of_nodes, self.dimension))
            return
        elif len(self.fixed_dofs) != 0 and placement.shape[0] == self.number_of_nodes * self.dimension - len(
                self.fixed_dofs):
            temp_placement = self.get_placement()
            temp_placement[np.array(list(self.free_dofs)) - 1] = placement
            self.nodal_coordinates = temp_placement.reshape((self.number_of_nodes, self.dimension))
            return

    def fix_dofs(self, dofs: list) -> None:
        """Constrain the DOFs given by the list of DOF numbers.

        :param dofs: DOF numbers to be fixed
        :type dofs: list
        """
        if dofs:
            for dof in dofs:
                self.fixed_dofs.add(dof)
                self.free_dofs.remove(dof)

    def free_dofs(self, dofs: list) -> None:
        """Free the DOFs given by the list of DOF numbers.

        :param dofs: DOF numbers to be fixed
        :type dofs: list
        """
        if dofs:
            for dof in dofs:
                self.free_dofs.add(dof)
                self.fixed_dofs.remove(dof)

    # Get structural related matrices

    def get_incidence_matrix(self) -> np.ndarray:
        """This method returns the incidence matrix [C] of the tensegrity.
        [C] is b-by-n, where b is the number of members, and n is the number
        of nodes. If Member k is connected to Node i and Node j, then C_ki
        and C_kj are non-zero, while other entries on Row k are all zeros. The
        convention used here is that if i < j, then C_ki = 1 whereas C_kj = -1.

        :return: the incidence matrix [C]
        :rtype: np.ndarray
        """
        incidence_matrix = np.zeros((self.number_of_members, self.number_of_nodes))
        for i in range(self.number_of_members):
            node1 = self.connectivity[i, 0]
            node2 = self.connectivity[i, 1]
            if node1 < node2:
                incidence_matrix[i, node1 - 1] = 1
                incidence_matrix[i, node2 - 1] = -1
            else:
                incidence_matrix[i, node1 - 1] = -1
                incidence_matrix[i, node2 - 1] = 1
        return incidence_matrix

    def get_edm(self) -> np.ndarray:
        """This method returns the Euclidean Distance Matrix (EDM).
        The matrix is n-by-n, where n is the number of nodes. Its ij element is
        the Euclidean distance between Node i and Node j. The diagonal entries
        are all zeros.

        :return: the Euclidean distance matrix
        :rtype: np.ndarray
        """
        edm = np.zeros((self.number_of_nodes, self.number_of_nodes))
        for i in range(self.number_of_nodes - 1):
            for j in range(i + 1, self.number_of_nodes):
                edm[i, j] = np.linalg.norm(self.nodal_coordinates[i, :] - self.nodal_coordinates[j, :])
                edm[j, i] = edm[i, j]
        return edm

    def get_equilibrium_matrix(self, reduce_rows: bool = True) -> np.ndarray:
        """This method returns the equilibrium matrix [A] of the tensegrity.
        The equilibrium condition of a tensegrity structure is controlled by
        [A]{t} = {f}, where [A] is the equilibrium matrix, {t} is the internal
        member force vector, and {f} is the external nodal force vector.

        The full equilibrium matrix is dn-by-b, where d is the dimension of the
        ambient space, n is the number of nodes, and b is the number of members.
        However, by default the method will not return the full matrix if there
        are DOF constraints. Instead, the rows corresponding to the constrained
        DOFs will be deleted. This behaviour can be changed by setting the parameter
        reduce_rows to be False, which gives the full matrix.

        :param reduce_rows: bool
        :return: the equilibrium matrix [A]
        :rtype: np.ndarray
        """
        equilibrium_matrix = np.zeros((self.dimension * self.number_of_nodes, self.number_of_members))
        for member_index in range(self.number_of_members):
            node1 = self.connectivity[member_index, 0]
            node2 = self.connectivity[member_index, 1]
            if node1 > node2:
                node1, node2 = node2, node1
            nodal_difference = self.nodal_coordinates[node1 - 1, :] - self.nodal_coordinates[node2 - 1, :]
            member_direction = nodal_difference / np.linalg.norm(nodal_difference)
            equilibrium_matrix[self.dimension * (node1 - 1):self.dimension * node1,
                               member_index] = member_direction
            equilibrium_matrix[self.dimension * (node2 - 1):self.dimension * node2,
                               member_index] = -member_direction

        if reduce_rows:
            indices = np.array(sorted(list(self.free_dofs))) - 1
            return equilibrium_matrix[indices, :]
        return equilibrium_matrix




if __name__ is "__main__":
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
    s = Tensegrity(nodal_coordinates, connectivity, member_types)
    print(s.get_equilibrium_matrix())
