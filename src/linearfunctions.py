import numpy as np

def compute_lower(weights_minus, weights_plus, input_lower, input_upper):
    return weights_plus.dot(input_lower) + weights_minus.dot(input_upper)


def compute_upper(weights_minus, weights_plus, input_lower, input_upper):
    return weights_plus.dot(input_upper) + weights_minus.dot(input_lower)


# def compute_lower_and_upper_equations(lower_equation, upper_equation, weights_minus, weights_plus, bias):
    # lower_matrix = lower_equation.get_matrix()
    # upper_matrix = upper_equation.get_matrix()

    # lower_offset = lower_equation.get_offset()
    # upper_offset = upper_equation.get_offset()

    # return LinearFunctions(compute_lower(weights_minus, weights_plus, lower_matrix, upper_matrix),
                           # compute_lower(weights_minus, weights_plus, lower_offset, upper_offset) + bias), \
           # LinearFunctions(compute_upper(weights_minus, weights_plus, lower_matrix, upper_matrix),
                           # compute_upper(weights_minus, weights_plus, lower_offset, upper_offset) + bias)


# def compute_lower_and_upper_relu_equations(lower_equation, upper_equation, lower_l, lower_u, upper_l, upper_u):
    # k_lower, b_lower = get_array_lin_lower_bound_coefficients(lower_l, lower_u)
    # k_upper, b_upper = get_array_lin_upper_bound_coefficients(upper_l, upper_u)

    # lower_matrix = get_transformed_matrix(lower_equation.get_matrix(), k_lower)
    # upper_matrix = get_transformed_matrix(upper_equation.get_matrix(), k_upper)
    # #
    # lower_offset = get_transformed_offset(lower_equation.get_offset(), k_lower, b_lower)
    # upper_offset = get_transformed_offset(upper_equation.get_offset(), k_upper, b_upper)

    # lower = LinearFunctions(lower_matrix, lower_offset)
    # upper = LinearFunctions(upper_matrix, upper_offset)

    # return lower, upper


# def get_transformed_matrix(matrix, k):
    # return matrix * k[:, None]


# def get_transformed_offset(offset, k, b):
    # return offset * k + b


# def get_array_lin_lower_bound_coefficients(lower, upper):
    # ks = np.zeros(len(lower))
    # bs = np.zeros(len(lower))

    # for i in range(len(lower)):
        # k, b = get_lin_lower_bound_coefficients(lower[i], upper[i])
        # ks[i] = k
        # bs[i] = b

    # return ks, bs


# def get_array_lin_upper_bound_coefficients(lower, upper):
    # ks = np.zeros(len(lower))
    # bs = np.zeros(len(lower))

    # for i in range(len(lower)):
        # k, b = get_lin_upper_bound_coefficients(lower[i], upper[i])
        # ks[i] = k
        # bs[i] = b

    # return ks, bs


# def get_lin_lower_bound_coefficients(lower, upper):
    # if lower >= 0:
        # return 1, 0

    # if upper <= 0:
        # return 0, 0

    # mult = upper / (upper - lower)

    # return mult, 0


# def get_lin_upper_bound_coefficients(lower, upper):
    # if lower >= 0:
        # return 1, 0

    # if upper <= 0:
        # return 0, 0

    # mult = upper / (upper - lower)
    # add = -mult*lower

    # return mult, add


class LinearFunctions:
    """
    matrix is an (n x m) np array
    offset is an (n) np array

    An object represents n linear functions f(i) of m input variables x

    f(i) = matrix[i]*x + offset[i]

    """
    def __init__(self, matrix, offset):
        self.size = matrix.shape[0]
        self.matrix = matrix
        self.offset = offset

    def clone(self):
        return LinearFunctions(self.matrix.copy(), self.offset.copy())

    def get_size(self):
        return self.size

    def get_matrix(self):
        return self.matrix

    def get_offset(self):
        return self.offset

    def compute_min_max_values(self, input_lower, input_upper):
        weights_plus = np.maximum(self.matrix, np.zeros(self.matrix.shape))
        weights_minus = np.minimum(self.matrix, np.zeros(self.matrix.shape))
        return compute_lower(weights_minus, weights_plus, input_lower, input_upper) + self.offset,\
               compute_upper(weights_minus, weights_plus, input_lower, input_upper) + self.offset

    def compute_max_values(self, input_bounds):
        input_lower = input_bounds['out']['l']
        input_upper = input_bounds['out']['u']
        weights_plus = np.maximum(self.matrix, np.zeros(self.matrix.shape))
        weights_minus = np.minimum(self.matrix, np.zeros(self.matrix.shape))
        return compute_upper(weights_minus, weights_plus, input_lower, input_upper) + self.offset

    def compute_min_values(self, input_bounds):
        input_lower = input_bounds['out']['l']
        input_upper = input_bounds['out']['u']
        weights_plus = np.maximum(self.matrix, np.zeros(self.matrix.shape))
        weights_minus = np.minimum(self.matrix, np.zeros(self.matrix.shape))
        return compute_lower(weights_minus, weights_plus, input_lower, input_upper) + self.offset

    def getLowerReluRelax(self, input_bounds):
        # get lower and upper bounds for the bound equation
        lower =  self.compute_min_values(input_bounds)
        upper =  self.compute_max_values(input_bounds)
        # matrix and offset for the relaxation
        matrix = self.matrix
        offset = self.offset

        # compute the coefficients of the linear approximation of out
        # bound equations
        for i in range(self.size):
            if  lower[i] >= 0:
                # Active node - Propagate lower bound equation unaltered
                pass
            elif upper[i] <=0: 
                # Inactive node - Propagate the zero function
                matrix[i,:]  = 0
                offset[i] = 0
            else:
                # Unstable node - Propagate linear relaxation of
                # lower bound equations

                # Standard approach
                # adj =  upper[i] / (upper[i] - lower[i])
                # matrix[i,:]  = matrix[i,:] * adj
                # offset[i]  = offset[i] * adj

                # Approach based on overapproximation areas
                # lb = lower[i]
                # ub = upper[i]
                # u_area = (pow(ub,2)/2) - (pow(ub,3)/(2*(ub-lb)))
                # l_area = (pow(lb,2) * ub) / (2*(ub-lb))
                # area = pow(ub,2)/2
                # if area < l_area + u_area:
                    # matrix[i,:] = 0
                    # offset[i] = 0
                # else:
                    # adj =  upper[i] / (upper[i] - lower[i])
                    # matrix[i,:]  = matrix[i,:] * adj
                    # offset[i]  = offset[i] * adj

                # Approach based on overapproximation areas
                lb = lower[i]
                lb2=pow(lb,2)
                ub = upper[i]
                ub2=pow(ub,2)
                # over-approximation area based on double
                # linear relaxation
                area1 = ((ub * lb2) - (ub2 * lb))/(2*(ub-lb))
                # over-approximation area based on triangle
                # relaxation
                area2 = pow(ub,2)/2
                # choose the least over-approximation
                if  area1 < area2:
                   adj =  upper[i] / (upper[i] - lower[i])
                   matrix[i,:]  = matrix[i,:] * adj
                   offset[i]  = offset[i] * adj
                else:
                   matrix[i,:] = 0
                   offset[i] = 0
 
        return LinearFunctions(matrix,offset)


    def getUpperReluRelax(self, input_bounds):
        # get lower and upper bounds for the bound equation
        lower =  self.compute_min_values(input_bounds)
        upper =  self.compute_max_values(input_bounds)
        # matrix and offset for the relaxation
        matrix = self.matrix
        offset = self.offset

        # compute the coefficients of the linear approximation of out
        # bound equations
        for i in range(self.size):
            if  lower[i] >= 0:
                # Active node - Propagate lower bound equation unaltered
                pass
            elif upper[i] <=0: 
                # Inactive node - Propagate the zero function
                matrix[i,:]  = 0
                offset[i] = 0
            else:
                # Unstable node - Propagate linear relaxation of
                # lower bound equations
                adj =  upper[i] / (upper[i] - lower[i])
                matrix[i,:]  = matrix[i,:] * adj
                offset[i]  = offset[i] * adj - adj * lower[i]
            
        return LinearFunctions(matrix,offset)

