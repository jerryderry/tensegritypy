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
                if not isinstance(dof, int):
                    raise TypeError("DOF number must be an integer.")
                elif dof <= 0:
                    raise ValueError("DOF number must be a positive integer.")
                self.fixed_dofs.add(dof)
                self.free_dofs.remove(dof)

    def free_dofs(self, dofs: list) -> None:
        """Free the DOFs given by the list of DOF numbers.

        :param dofs: DOF numbers to be fixed
        :type dofs: list
        """
        if dofs:
            for dof in dofs:
                if not isinstance(dof, int):
                    raise TypeError("DOF number must be an integer.")
                elif dof <= 0:
                    raise ValueError("DOF number must be a positive integer.")
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

        Each row of the equilibrium matrix corresponds to a DOF of the structure.
        The DOFs associated with one node are grouped together. Therefore, for example,
        the first three rows correspond to x, y and z directions of Node 1.

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

    def get_geometric_matrix(self, reduce_rows: bool = True) -> np.ndarray:
        """This method returns the geometric matrix :math:`[\Pi]` of the tensegrity.
        The equilibrium condition of a tensegrity structure is controlled by
        :math:`[\Pi]{q} = {f}`, where :math:`[\Pi]` is the geometric matrix,
        :math:`{q}` is the force density vector, and :math:`{f}` is the external
        nodal force vector.

        The full geometric matrix is dn-by-b, where d is the dimension of the
        ambient space, n is the number of nodes, and b is the number of members.
        However, by default the method will not return the full matrix if there
        are DOF constraints. Instead, the rows corresponding to the constrained
        DOFs will be deleted. This behaviour can be changed by setting the parameter
        reduce_rows to be False, which gives the full matrix.

        Each row of the geometric matrix corresponds to a DOF of the structure.
        The DOFs associated with one node are grouped together. Therefore, for example,
        the first three rows correspond to x, y and z directions of Node 1.

        :param reduce_rows: bool
        :return: the geometric matrix :math:`[\Pi]`
        :rtype: np.ndarray
        """
        geometric_matrix = np.zeros((self.dimension * self.number_of_nodes, self.number_of_members))
        for member_index in range(self.number_of_members):
            node1 = self.connectivity[member_index, 0]
            node2 = self.connectivity[member_index, 1]
            if node1 > node2:
                node1, node2 = node2, node1
            nodal_difference = self.nodal_coordinates[node1 - 1, :] - self.nodal_coordinates[node2 - 1, :]
            geometric_matrix[self.dimension * (node1 - 1):self.dimension * node1,
                             member_index] = nodal_difference
            geometric_matrix[self.dimension * (node2 - 1):self.dimension * node2,
                             member_index] = -nodal_difference

        if reduce_rows:
            indices = np.array(sorted(list(self.free_dofs))) - 1
            return geometric_matrix[indices, :]
        return geometric_matrix

    def get_compatibility_matrix(self, reduce_columns: bool = True) -> np.ndarray:
        """This method returns the compatibility matrix [B] of the tensegrity.
        The compatibility condition is controlled by :math:`[B]{d} = {\Delta}`,
        where :math:`[B]` is the compatibility matrix, :math:`{d}` is the nodal
        displacement vector, and :math:`{\Delta}` is the member elongation vector.

        The full compatibility matrix is b-by-nd, where d is the dimension of the
        ambient space, n is the number of nodes, and b is the number of members.
        The compatibility matrix is the transpose of the equilibrium matrix.
        However, by default the method will not return the full matrix if there
        are DOF constraints. Instead, the columns corresponding to the constrained
        DOFs will be deleted. This behaviour can be changed by setting the parameter
        reduce_rows to be False, which gives the full matrix.

        Each column of the geometric matrix corresponds to a DOF of the structure.
        The DOFs associated with one node are grouped together. Therefore, for example,
        the first three columns correspond to x, y and z directions of Node 1.

        :param reduce_columns: bool
        :return: the compatibility matrix :math:`[B]`
        :rtype: np.ndarray
        """
        if reduce_columns:
            return self.get_equilibrium_matrix().transpose()
        else:
            return self.get_equilibrium_matrix(False).transpose()

    def get_rigidity_matrix(self, reduce_columns: bool = True) -> np.ndarray:
        """This method returns the rigidity matrix [R] of the tensegrity.
        The compatibility condition is controlled by :math:`[R]{v} = {\epsilon}`,
        where :math:`[R]` is the rigidity matrix, :math:`{v}` is the nodal
        velocity vector, and :math:`{\epsilon}` is the member strain vector.

        The full rigidity matrix is b-by-nd, where d is the dimension of the
        ambient space, n is the number of nodes, and b is the number of members.
        The rigidity matrix is the transpose of the geometric matrix.
        However, by default the method will not return the full matrix if there
        are DOF constraints. Instead, the columns corresponding to the constrained
        DOFs will be deleted. This behaviour can be changed by setting the parameter
        reduce_rows to be False, which gives the full matrix.

        Each column of the geometric matrix corresponds to a DOF of the structure.
        The DOFs associated with one node are grouped together. Therefore, for example,
        the first three columns correspond to x, y and z directions of Node 1.

        :param reduce_columns: bool
        :return: the rigidity matrix :math:`[R]`
        :rtype: np.ndarray
        """
        if reduce_columns:
            return self.get_geometric_matrix().transpose()
        else:
            return self.get_geometric_matrix(False).transpose()


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
