""" This module contains objective function implementations of the DTLZ test
suite, derived from the implementations in ParMOO:

Chang and Wild. "ParMOO: A Python library for parallel multiobjective
simulation optimization." Journal of Open Source Software 8(82):4468, 2023.

------------------------------------------------------------------------------

For further references, the DTLZ test suite was originally proposed in:

Deb, Thiele, Laumanns, and Zitzler. "Scalable test problems for
evolutionary multiobjective optimization" in Evolutionary Multiobjective
Optimization, Theoretical Advances and Applications, Ch. 6 (pp. 105--145).
Springer-Verlag, London, UK, 2005. Abraham, Jain, and Goldberg (Eds).

The original implementation was appropriate for testing randomized algorithms,
but for many deterministic algorithms, the global solutions represent either
best- or worst-case scenarios, so an configurable offset was introduced in:

Chang. "Mathematical Software for Multiobjective Optimization Problems."
Ph.D. dissertation, Virginia Tech, Dept. of Computer Science, 2020.

------------------------------------------------------------------------------

The full list of public classes in this module includes the 7 unconstrained
DTLZ problems:
 * ``dtlz1``
 * ``dtlz2``
 * ``dtlz3``
 * ``dtlz4``
 * ``dtlz5``
 * ``dtlz6``
 * ``dtlz7``
 * ``dtlz8``
 * ``dtlz9``

"""

import numpy as np


class __dtlz_base__():
    """ Base class implements re-used constructor """

    def __init__(self, num_des, num_obj=3, offset=0.5):
        """ Constructor for all DTLZ classes.

        Args:
            num_des (int): The number of design variables.

            num_obj (int, optional): The number of objectives.

            offset (optional, float): The location of the global minimizers
                is x_i = offset for i = num_objectives, ..., num_des.
                The default offset is 0.5.

        """

        self.n = num_des
        self.o = num_obj
        self.offset = offset
        return

    def __call__(self, x):
        raise NotImplementedError("The call method must be implemented...")


class __g1__(__dtlz_base__):
    """ Class defining 1 of 4 kernel functions used in the DTLZ problem suite.

    g1 = 100 ( (n - o + 1) +
               sum_{i=o}^n ((x_i - offset)^2 - cos(20pi(x_i - offset))) )

    Contains 2 methods:
     * ``__init__(num_des, num_obj)``
     * ``__call__(x)``

    The ``__init__`` method creates a new kernel.

    The ``__call__`` method performs an evaluation of the g1 kernel.

    """

    def __call__(self, x):
        """ Define objective evaluation.

        Args:
            x (numpy.array): A numpy.ndarray containing the design point
                to evaluate.

        Returns:
            float: The output of this objective for the input x.

        """

        return (1 + self.n - self.o +
                np.sum((x[self.o-1:self.n] - self.offset) ** 2 -
                        np.cos(20.0 * np.pi *
                               (x[self.o-1:self.n] - self.offset)))) * 100.0


class __g2__(__dtlz_base__):
    """ Class defining 2 of 4 kernel functions used in the DTLZ problem suite.

    g2 = (x_o - offset)^2 + ... + (x_n - offset)^2

    Contains 2 methods:
     * ``__init__(num_des, num_obj)``
     * ``__call__(x)``

    The ``__init__`` method inherits from the __dtlz_base__ ABC.

    The ``__call__`` method performs an evaluation of the g2 problem.

    """

    def __call__(self, x):
        """ Define objective evaluation.

        Args:
            x (numpy.array): A numpy.ndarray containing the design point
                to evaluate.

        Returns:
            float: The output of this objective for the input x.

        """

        return np.sum((x[self.o-1:self.n] - self.offset) ** 2)


class __g3__(__dtlz_base__):
    """ Class defining 3 of 4 kernel functions used in the DTLZ problem suite.

    g3 = |x_o - offset|^.1 + ... + |x_n - offset|^.1

    Contains 2 methods:
     * ``__init__(num_des, num_obj)``
     * ``__call__(x)``

    The ``__init__`` method inherits from the __dtlz_base__ ABC.

    The ``__call__`` method performs an evaluation of the g3 problem.

    """

    def __init__(self, num_des, num_obj=3, offset=0.0):
        """ Constructor for g3, with modified default offset.

        Args:
            num_des (int): The number of design variables.

            num_obj (int, optional): The number of objectives.

            offset (optional, float): The location of the global minimizers
                is x_i = offset for i = num_objectives, ..., num_des.
                The default offset is 0.0.

        """

        super().__init__(num_des=num_des, num_obj=num_obj, offset=offset)
        return

    def __call__(self, x):
        """ Define objective evaluation.

        Args:
            x (numpy.array): A numpy.ndarray containing the design point
                to evaluate.

        Returns:
            float: The output of this objective for the input x.

        """

        return np.sum(np.abs(x[self.o-1:self.n] - self.offset) ** 0.1)


class __g4__(__dtlz_base__):
    """ Class defining 4 of 4 kernel functions used in the DTLZ problem suite.

    g4 = 1 + (9 * (|x_o - offset| + ... + |x_n - offset|) / (n + 1 - o))

    Contains 2 methods:
     * ``__init__(num_des, num_obj)``
     * ``__call__(x)``

    The ``__init__`` method inherits from the __dtlz_base__ ABC.

    The ``__call__`` method performs an evaluation of the g4 problem.

    """

    def __init__(self, num_des, num_obj=3, offset=0.0):
        """ Constructor for g4, with modified default offset.

        Args:
            num_des (int): The number of design variables.

            num_obj (int, optional): The number of objectives.

            offset (optional, float): The location of the global minimizers
                is x_i = offset for i = num_objectives, ..., num_des.
                The default offset is 0.0.

        """

        super().__init__(num_des=num_des, num_obj=num_obj, offset=offset)
        return

    def __call__(self, x):
        """ Define objective evaluation.

        Args:
            x (numpy.array): A numpy.ndarray containing the design point
                to evaluate.

        Returns:
            float: The output of this objective for the input x.

        """

        return (9 * np.sum(np.abs(x[self.o-1:self.n] - self.offset))
                / float(self.n + 1 - self.o)) + 1.0


class dtlz1(__dtlz_base__):
    """ Class defining the DTLZ1 problem with offset minimizer.

    DTLZ1 has a linear Pareto front, with all nondominated points
    on the hyperplane F_1 + F_2 + ... + F_o = 0.5.
    DTLZ1 has 11^k - 1 "local" Pareto fronts where k = n - o + 1, and
    1 "global" Pareto front.

    Contains 2 methods:
     * ``__init__(num_des, num_obj)``
     * ``__call__(x)``

    The ``__init__`` method inherits from the __dtlz_base__ ABC.

    The ``__call__`` method performs an evaluation of the DTLZ1 problem.

    """

    def __call__(self, x):
        """ Define objective evaluation.

        Args:
            x (numpy.ndarray): A numpy.ndarray containing the design point
                to evaluate.

        Returns:
            numpy float array: The output of this objective for the input x.

        """

        # Initialize kernel function
        ker = __g1__(self.n, self.o, self.offset)
        # Initialize output array
        fx = np.zeros(self.o)
        fx[:] = (1.0 + ker(x)) / 2.0
        # Calculate the output array
        for i in range(self.o):
            for j in range(self.o - 1 - i):
                fx[i] *= x[j]
            if i > 0:
                fx[i] *= (1.0 - x[self.o - 1 - i])
        return fx


class dtlz2(__dtlz_base__):
    """ Class defining the DTLZ2 problem with offset minimizer.

    DTLZ2 has a concave Pareto front, given by the unit sphere in
    objective space, restricted to the positive orthant.
    DTLZ2 has no "local" Pareto fronts, besides the true Pareto front.

    Contains 2 methods:
     * ``__init__(num_des, num_obj)``
     * ``__call__(x)``

    The ``__init__`` method inherits from the __dtlz_base__ ABC.

    The ``__call__`` method performs an evaluation of the DTLZ2 problem.

    """

    def __call__(self, x):
        """ Define objective evaluation.

        Args:
            x (numpy.ndarray): A numpy.ndarray containing the design point
                to evaluate.

        Returns:
            numpy float array: The output of this objective for the input x.

        """

        # Initialize kernel function
        ker = __g2__(self.n, self.o, self.offset)
        # Initialize output array
        fx = np.zeros(self.o)
        fx[:] = (1.0 + ker(x))
        # Calculate the output array
        for i in range(self.o):
            for j in range(self.o - 1 - i):
                fx[i] *= np.cos(np.pi * x[j] / 2)
            if i > 0:
                fx[i] *= np.sin(np.pi * x[self.o - 1 - i] / 2)
        return fx


class dtlz3(__dtlz_base__):
    """ Class defining the DTLZ3 problem with offset minimizer.

    DTLZ3 has a concave Pareto front, given by the unit sphere in
    objective space, restricted to the positive orthant.
    DTLZ3 has 3^k - 1 "local" Pareto fronts where k = n - o + 1, and
    1 "global" Pareto front.

    Contains 2 methods:
     * ``__init__(num_des, num_obj)``
     * ``__call__(x)``

    The ``__init__`` method inherits from the __dtlz_base__ ABC.

    The ``__call__`` method performs an evaluation of the DTLZ3 problem.

    """

    def __call__(self, x):
        """ Define objective evaluation.

        Args:
            x (numpy.ndarray): A numpy.ndarray containing the design point
                to evaluate.

        Returns:
            numpy float array: The output of this objective for the input x.

        """


        # Initialize kernel function
        ker = __g1__(self.n, self.o, self.offset)
        # Initialize output array
        fx = np.zeros(self.o)
        fx[:] = (1.0 + ker(x))
        # Calculate the output array
        for i in range(self.o):
            for j in range(self.o - 1 - i):
                fx[i] *= np.cos(np.pi * x[j] / 2)
            if i > 0:
                fx[i] *= np.sin(np.pi * x[self.o - 1 - i] / 2)
        return fx


class dtlz4(__dtlz_base__):
    """ Class defining the DTLZ4 problem with offset minimizer.

    DTLZ4 has a concave Pareto front, given by the unit sphere in
    objective space, restricted to the positive orthant.
    DTLZ4 has no "local" Pareto fronts, besides the true Pareto front,
    but by tuning the optional parameter alpha, one can adjust the
    solution density, making it harder for MOO algorithms to produce
    a uniform distribution of solutions.

    Contains 2 methods:
     * ``__init__(num_des, num_obj)``
     * ``__call__(x)``

    The ``__init__`` method inherits from the __dtlz_base__ ABC.

    The ``__call__`` method performs an evaluation of the DTLZ4 problem.

    """

    def __init__(self, num_des, num_obj=3, offset=0.0, alpha=100.0):
        """ Constructor for DTLZ7, with modified default offset.

        Args:
            num_des (int): The number of design variables.

            num_obj (int, optional): The number of objectives.

            offset (optional, float): The location of the global minimizers
                is x_i = offset for i = num_objectives, ..., num_des.
                The default offset is 0.0.

            alpha (optional, float or int): The uniformity parameter used for
                controlling the uniformity of the distribution of solutions
                across the Pareto front. Must be greater than or equal to 1.
                A value of 1 results in DTLZ2. Default value is 100.0.

        """

        super().__init__(num_des=num_des, num_obj=num_obj, offset=offset)
        self.alpha = alpha
        return

    def __call__(self, x):
        """ Define objective evaluation.

        Args:
            x (numpy.ndarray): A numpy.ndarray containing the design point
                to evaluate.

        Returns:
            numpy float array: The output of this objective for the input x.

        """

        # Initialize kernel function
        ker = __g2__(self.n, self.o, self.offset)
        # Initialize output array
        fx = np.zeros(self.o)
        fx[:] = (1.0 + ker(x))
        # Calculate the output array
        for i in range(self.o):
            for j in range(self.o - 1 - i):
                fx[i] *= np.cos(np.pi * x[j] ** self.alpha / 2)
            if i > 0:
                fx[i] *= np.sin(np.pi * x[self.o - 1 - i] ** self.alpha / 2)
        return fx


class dtlz5(__dtlz_base__):
    """ Class defining the DTLZ5 problem with offset minimizer.

    DTLZ5 has a lower-dimensional Pareto front embedded in the objective
    space, given by an arc of the unit sphere in the positive orthant.
    DTLZ5 has no "local" Pareto fronts, besides the true Pareto front.

    Contains 2 methods:
     * ``__init__(num_des, num_obj)``
     * ``__call__(x)``

    The ``__init__`` method inherits from the __dtlz_base__ ABC.

    The ``__call__`` method performs an evaluation of the DTLZ5 problem.

    """

    def __call__(self, x):
        """ Define objective evaluation.

        Args:
            x (numpy.ndarray): A numpy.ndarray containing the design point
                to evaluate.

        Returns:
            numpy float array: The output of this objective for the input x.

        """

        # Initialize kernel function
        ker = __g2__(self.n, self.o, self.offset)
        # Calculate theta values
        theta = np.zeros(self.o - 1)
        g2x = ker(x)
        for i in range(self.o - 1):
            theta[i] = np.pi * (1 + 2 * g2x * x[i]) / (4 * (1 + g2x))
        # Initialize output array
        fx = np.zeros(self.o)
        fx[:] = (1.0 + g2x)
        # Calculate the output array
        for i in range(self.o):
            for j in range(self.o - 1 - i):
                fx[i] *= np.cos(np.pi * theta[j] / 2)
            if i > 0:
                fx[i] *= np.sin(np.pi * theta[self.o - 1 - i] / 2)
        return fx


class dtlz6(__dtlz_base__):
    """ Class defining the DTLZ6 problem with offset minimizer.

    DTLZ6 has a lower-dimensional Pareto front embedded in the objective
    space, given by an arc of the unit sphere in the positive orthant.
    DTLZ6 has no "local" Pareto fronts, but tends to show very little
    improvement until the algorithm is very close to its solution set.

    Contains 2 methods:
     * ``__init__(num_des, num_obj)``
     * ``__call__(x)``

    The ``__init__`` method inherits from the __dtlz_base__ ABC.

    The ``__call__`` method performs an evaluation of the DTLZ6 problem.

    """

    def __init__(self, num_des, num_obj=3, offset=0.0):
        """ Constructor for DTLZ6, with modified default offset.

        Args:
            num_des (int): The number of design variables.

            num_obj (int, optional): The number of objectives.

            offset (optional, float): The location of the global minimizers
                is x_i = offset for i = num_objectives, ..., num_des.
                The default offset is 0.0.

        """

        super().__init__(num_des=num_des, num_obj=num_obj, offset=offset)
        return

    def __call__(self, x):
        """ Define objective evaluation.

        Args:
            x (numpy.ndarray): A numpy.ndarray containing the design point
                to evaluate.

        Returns:
            numpy float array: The output of this objective for the input x.

        """

        # Initialize kernel function
        ker = __g3__(self.n, self.o, self.offset)
        # Calculate theta values
        theta = np.zeros(self.o - 1)
        g3x = ker(x)
        for i in range(self.o - 1):
            theta[i] = np.pi * (1 + 2 * g3x * x[i]) / (4 * (1 + g3x))
        # Initialize output array
        fx = np.zeros(self.o)
        fx[:] = (1.0 + g3x)
        # Calculate the output array
        for i in range(self.o):
            for j in range(self.o - 1 - i):
                fx[i] *= np.cos(np.pi * theta[j] / 2)
            if i > 0:
                fx[i] *= np.sin(np.pi * theta[self.o - 1 - i] / 2)
        return fx


class dtlz7(__dtlz_base__):
    """ Class defining the DTLZ7 problem with offset minimizer.

    DTLZ7 has a discontinuous Pareto front, with solutions on the 
    2^(o-1) discontinuous nondominated regions of the surface:

    F_m = o - F_1 (1 + sin(3pi F_1)) - ... - F_{o-1} (1 + sin3pi F_{o-1}).

    Contains 2 methods:
     * ``__init__(num_des, num_obj)``
     * ``__call__(x)``

    The ``__init__`` method inherits from the __dtlz_base__ ABC.

    The ``__call__`` method performs an evaluation of the DTLZ7 problem.

    """

    def __init__(self, num_des, num_obj=3, offset=0.0):
        """ Constructor for DTLZ7, with modified default offset.

        Args:
            num_des (int): The number of design variables.

            num_obj (int, optional): The number of objectives.

            offset (optional, float): The location of the global minimizers
                is x_i = offset for i = num_objectives, ..., num_des.
                The default offset is 0.0.

        """

        super().__init__(num_des=num_des, num_obj=num_obj, offset=offset)
        return

    def __call__(self, x):
        """ Define objective evaluation.

        Args:
            x (numpy.ndarray): A numpy.ndarray containing the design point
                to evaluate.

        Returns:
            numpy float array: The output of this objective for the input x.

        """

        # Initialize kernel function
        ker = __g4__(self.n, self.o, self.offset)
        # Initialize first o-1 entries in the output array
        fx = np.zeros(self.o)
        fx[:self.o-1] = x[:self.o-1]
        # Calculate kernel functions
        gx = 1.0 + ker(x)
        hx = (-np.sum(x[:self.o-1] *
                      (1.0 + np.sin(3.0 * np.pi * x[:self.o-1]) / gx))
                      + float(self.o))
        # Calculate the last entry in the output array
        fx[self.o-1] = gx * hx
        return fx
