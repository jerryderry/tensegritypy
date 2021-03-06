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
        self.force_densities = None
        self.youngs_modulii = None
        self.cross_section_areas = None
        self.original_lengths = None

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

        :param reduce_rows: whether to delete rows corresponding to fixed DOFs
        :type reduce_rows: bool
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

        :param reduce_rows: whether to delete rows corresponding to fixed DOFs
        :type reduce_rows: bool
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

        :param reduce_columns: whether to delete columns corresponding to fixed DOFs
        :type reduce_columns: bool
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

        :param reduce_columns: whether to delete columns corresponding to fixed DOFs
        :type reduce_columns: bool
        :return: the rigidity matrix :math:`[R]`
        :rtype: np.ndarray
        """
        if reduce_columns:
            return self.get_geometric_matrix().transpose()
        else:
            return self.get_geometric_matrix(False).transpose()

    def get_laplacian_matrix(self) -> np.ndarray:
        """This method returns the Laplacian matrix associated with a tensegrity.
        The equilibrium condition of a 3D tensegrity can also be express by
        .. math::
        [\mathbf{L}]{\mathbf{x}] &= {\mathbf{f}_x},\\
        [\mathbf{L}]{\mathbf{y}] &= {\mathbf{f}_y},\\
        [\mathbf{L}]{\mathbf{z}] &= {\mathbf{f}_z},

        where :math:`[\mathbf{L}]` is the Laplacian matrix, :math:`{\mathbf{x}}`,
        :math:`{\mathbf{y}}` and :math:`{\mathbf{z}}` are nodal coordinate vectors
        in directions x, y and z, respectively, and similarly, :math:`{\mathbf{f}_x}`,
        :math:`{\mathbf{f}_y}` and :math:`{\mathbf{f}_z}` are external nodal
        force vectors in the three directions.

        The Laplacian matrix is n-by-n, where n is the number of nodes. Its off-diagonal
        element :math:`L_{ij}` is the force density of Member ij with negative sign. If
        there is no member between Node i and Node j, then :math:`L_{ij}` is zero. The
        diagonal entries are the sum of the other entries on the same row with an
        opposite sign, such that the row sums of the matrix are zeros.

        The Laplacian matrix is symmetric.

        :return: the Laplacian matrix :math:`[\mathbf{L}]`
        :rtype: np.ndarray
        """
        if not self.force_densities:
            # TODO: Raise exception
            pass
        incidence_matrix = self.get_incidence_matrix()
        force_density_matrix = np.diagonal(self.force_densities)
        return np.linalg.multi_dot([incidence_matrix.transpose(), force_density_matrix, incidence_matrix])

    def get_stress_matrix(self, reduced: bool = True, grouped: str = "by_node") -> np.ndarray:
        """This method returns the stress matrix [S] associated with a tensegrity.
        If the rows of the matrix are grouped by nodes, i.e., the rows corresponding
        to different directions of the same node are grouped together, then the stress
        matrix is calculated by
        .. math::
        [\mathbf{S}] = [\mathbf{L}] \otimes [\mathbf{I}_d],

        where d is the dimension of the structure. However, if rows are grouped by
        directions, then it is calculated by
        .. math::
        [\mathbf{S}] = [\mathbf{I}_d] \otimes [\mathbf{L}].

        The full stress matrix is dn-by-dn, where d is the dimension of the
        ambient space, n is the number of nodes.

        However, by default the method will not return the full matrix if there
        are DOF constraints. Instead, the rows and columns corresponding to the
        constrained DOFs will be deleted. This behaviour can be changed by setting
        the parameter reduced to be False, which gives the full matrix.

        :param reduced: whether to delete rows and columns corresponding to constrained
        DOFs
        :type reduced: bool
        :param grouped: how the rows are grouped, can be "by_node" or "by_direction"
        :type grouped: str
        :return: the stress matrix [S]
        :rtype: np.ndarray
        """
        if grouped == "by_node":
            stress_matrix = np.kron(self.get_laplacian_matrix(), np.identity(self.dimension))
            if reduced:
                indices = np.array(sorted(list(self.free_dofs))) - 1
                return stress_matrix[indices, indices]
            return stress_matrix
        elif grouped == "by_direction":
            stress_matrix = np.kron(np.identity(self.dimension), self.get_laplacian_matrix())
            if reduced:
                indices = []
                dofs = sorted(list(self.free_dofs))
                for dof in dofs:
                    node_zero_base = (dof - 1) // self.dimension
                    direction_zero_base = (dof - 1) % self.dimension
                    indices.append(node_zero_base * self.dimension + direction_zero_base)
                indices = np.array(indices)
                return stress_matrix[indices, indices]
            return stress_matrix
        else:
            raise ValueError("The way of grouping can only be by_node or by_direction.")

    def get_axial_stiffness_matrix(self) -> np.ndarray:
        """This method returns a diagonal matrix [G] whose diagonal entries are the
        axial stiffness of the members.

        :return: the axial stiffness matrix [G]
        :rtype: np.ndarray
        """
        return np.diagonal(np.divide(np.multiply(self.youngs_modulii, self.cross_section_areas),
                                     self.original_lengths))

    def get_tangent_stiffness_matrix(self, reduced: bool = True, kind: str = "default") -> np.ndarray:
        """This method returns the tangent stiffness matrix of a tensegrity structure.
        The default kind of matrix is calculated as:
        .. math::
        [\mathbf{K}_t] = [\mathbf{A}]([\mathbf{G}] - [\mathbf{Q}])[\mathbf{A}]^T + [\mathbf{S}],

        where :math:`[\mathbf{K}_t]` is the tangent stiffness matrix,
        :math:`[\mathbf{A}]` is the equilibrium matrix, :math:`[\mathbf{G}]` is the
        axial stiffness matrix, :math:`[\mathbf{Q}]` is the diagonal force density matrix,
        and :math:`[\mathbf{S}]` is the stress matrix.

        If kind is "modified", then the matrix is calculated as:
        .. math::
        [\mathbf{K}_t] = [\mathbf{A}][\mathbf{G}][\mathbf{A}]^T + [\mathbf{S}].

        If kind is "Murakami", then it is calculated as described in Eqn (36b) and
        Eqn (36c) of the paper by Murakami: "Static and Dynamic Analyses of Tensegrity
        Structures. Part 1. Nonlinear equations of motion (2001)".

        By default the method will not return the full matrix if there
        are DOF constraints. Instead, the rows and columns corresponding to the
        constrained DOFs will be deleted. This behaviour can be changed by setting
        the parameter reduced to be False, which gives the full matrix.

        :param reduced: whether to delete rows and columns corresponding to fixed DOFs
        :type reduced: bool
        :param kind: which kind of tangent stiffness matrix to return
        :type kind: str
        :return: the tangent stiffness matrix
        :rtype: np.ndarray
        """
        if kind == "default":
            equilibrium_matrix = self.get_equilibrium_matrix(reduce_rows=reduced)
            force_density_matrix = np.diagonal(self.force_densities)
            axial_stiffness_matrix = self.get_axial_stiffness_matrix()
            zero_member_force = self.force_densities == 0
            axial_stiffness_matrix[zero_member_force, zero_member_force] = 0
            stress_matrix = self.get_stress_matrix(reduced=reduced)
            return np.linalg.multi_dot([equilibrium_matrix, (axial_stiffness_matrix - force_density_matrix),
                                        equilibrium_matrix.transpose()]) + stress_matrix
        elif kind == "modified":
            equilibrium_matrix = self.get_equilibrium_matrix(reduce_rows=reduced)
            axial_stiffness_matrix = self.get_axial_stiffness_matrix()
            zero_member_force = self.force_densities == 0
            axial_stiffness_matrix[zero_member_force, zero_member_force] = 0
            stress_matrix = self.get_stress_matrix(reduced=reduced)
            return np.linalg.multi_dot([equilibrium_matrix, axial_stiffness_matrix,
                                        equilibrium_matrix.transpose()]) + stress_matrix
        elif kind == "Murakami":
            tangent_stiffness_matrix = np.zeros((self.number_of_nodes * self.dimension,
                                                 self.number_of_nodes * self.dimension))
            member_lengths = self.get_member_lengths()
            for i in range(self.number_of_members):
                element_stiffness_matrix = np.zeros((2 * self.dimension, 2 * self.dimension))
                node1 = self.connectivity[i, 0]
                node2 = self.connectivity[i, 1]
                member_direction = self.nodal_coordinates[node1 - 1, :] - self.nodal_coordinates[node2 - 1, :]
                unit_member_direction = member_direction / np.linalg.norm(member_direction)
                element_stiffness_matrix[0:self.dimension-1,
                                         0:self.dimension-1] = np.dot(unit_member_direction.transpose(),
                                                                      unit_member_direction)
                element_stiffness_matrix[self.dimension:-1,
                                         self.dimension:-1] = np.dot(unit_member_direction.transpose(),
                                                                     unit_member_direction)
                element_stiffness_matrix[self.dimension:-1,
                                         0:self.dimension-1] = -np.dot(unit_member_direction.transpose(),
                                                                       unit_member_direction)
                element_stiffness_matrix[0:self.dimension-1,
                                         self.dimension:-1] = -np.dot(unit_member_direction.transpose(),
                                                                      unit_member_direction)
                element_stiffness_matrix = (self.youngs_modulii[i] * self.cross_section_areas[i]
                                            / member_lengths[i]) * element_stiffness_matrix

                element_stress_matrix = np.identity(2 * self.dimension)
                element_stress_matrix[0:self.dimension-1, self.dimension:-1] = -np.identity(self.dimension)
                element_stress_matrix[self.dimension:-1, 0:self.dimension-1] = -np.identity(self.dimension)
                element_stress_matrix = self.force_densities[i] * element_stress_matrix

                element_stiffness_matrix = element_stiffness_matrix + element_stress_matrix

                global_local_map = np.zeros(2 * self.dimension, self.number_of_nodes * self.dimension)
                global_local_map[0:self.dimension-1,
                                 (node1-1)*self.dimension:node1*self.dimension-1] = np.identity(self.dimension)
                global_local_map[self.dimension:-1,
                                 (node2-1)*self.dimension:node2*self.dimension-1] = np.identity(self.dimension)

                tangent_stiffness_matrix = tangent_stiffness_matrix + np.linalg.multi_dot([global_local_map.transpose(),
                                                                                           element_stiffness_matrix,
                                                                                           global_local_map])
            if reduced:
                indices = np.array(sorted(list(self.free_dofs))) - 1
                return tangent_stiffness_matrix[indices, indices]
            return tangent_stiffness_matrix
        else:
            raise ValueError("Does not recognize the matrix kind.")

    def get_member_lengths(self) -> np.ndarray:
        """This method returns a vector of member lengths.

        If a member is a cable, and the nodal distance is smaller than the original
        length of the cable, then its member length will be set to equal its
        original length.

        :return: a vector of member lengths
        :rtype: np.ndarray
        """
        if not self.original_lengths:
            return self.get_nodal_distances()
        member_lengths = np.zeros(self.number_of_members)
        for i in range(self.number_of_members):
            node1 = self.connectivity[i, 0]
            node2 = self.connectivity[i, 1]
            distance = np.linalg.norm(self.nodal_coordinates[node1 - 1, :] - self.nodal_coordinates[node2 - 1, :])
            if self.member_types[i] is Member.CABLE and distance <= self.original_lengths[i]:
                member_lengths[i] = self.original_lengths[i]
            else:
                member_lengths[i] = distance
        return member_lengths

    def get_nodal_distances(self) -> np.ndarray:
        """This method returns a vector of nodal distances of the structure,
        if there is a member connected between two nodes.

        The difference between this method and get_member_lengths() is that
        nodal distances will not be compared with original lengths.

        :return: a vector of nodal distances between nodes that are connected
        :rtype: np.ndarray
        """
        nodal_distances = np.zeros(self.number_of_members)
        for i in range(self.number_of_members):
            node1 = self.connectivity[i, 0]
            node2 = self.connectivity[i, 1]
            nodal_distances[i] = np.linalg.norm(self.nodal_coordinates[node1 - 1, :]
                                                - self.nodal_coordinates[node2 - 1, :])
        return nodal_distances


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
