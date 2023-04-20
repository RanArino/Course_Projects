import math
import sympy


# to make the calculation easier, every float value (output) is rounded up by 1
def get_distance(pos1, pos2):
    return math.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)


def get_degree(current_pos, target_pos, center_pos):
    """
    get theta(degree) between two points and close to center
    """
    x1, y1 = current_pos
    x2, y2 = target_pos
    cx, cy = center_pos
    # Lengths of the sides of the triangle
    a = math.sqrt((x1 - cx) ** 2 + (y1 - cy) ** 2)
    b = math.sqrt((x2 - cx) ** 2 + (y2 - cy) ** 2)
    c = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    eql = round((a ** 2 + b ** 2 - c ** 2) / (2 * a * b), 1)
    radian = math.acos(eql)
    degree = radian * (180 / math.pi)
    return degree


def arc_length(current_pos, target_pos, center_pos, radius):
    """
    Given the current & target xy coordinates and degree,
    Return arc length(m).
    """
    degree = get_degree(current_pos, target_pos, center_pos)
    arced_length = 2*math.pi * radius * (degree/360)
    return round(arced_length, 0)


def circle_by_center_radius(center, radius):
    x, y = sympy.symbols('x y')  # this is target
    return sympy.simplify((x - center[0]) ** 2 + (y - center[1]) ** 2 - radius ** 2)


def line_by_point_length(start_xy, length):
    x, y = sympy.symbols('x y')  # this is target
    sx, sy = round(start_xy[0], 1), round(start_xy[1], 1)
    return sympy.simplify(sympy.sqrt((x - sx) ** 2 + (y - sy) ** 2) - round(length, 1))


def solve_two_equations(e1, e2):
    """
    return the solution which is closer to the target xy position
    """
    x, y = sympy.symbols('x y')
    # Define the equations
    eq1 = sympy.Eq(e1, 0)
    eq2 = sympy.Eq(e2, 0)
    # Solve equations
    solutions = sympy.solve((eq1, eq2), (x, y), dict=True, check=False, minimal=True)
    sols = []
    for s in solutions:
        try:
            sx = round(float(s[x]), 1)
            sy = round(float(s[y]), 1)
        except TypeError:  # cannot convert complex to float
            # dealing with the complex type; only focusing on the real number
            sx = round(s[x].as_real_imag()[0], 1)
            sy = round(s[y].as_real_imag()[0], 1)

        sols.append((sx, sy))
    return sols


def deceleration(init_speed, target_speed, distance, max_dcl_kmh):
    """
    Adjusting the deceleration rate (km/h) per one second,
     based on the current car speed, what a driver should decrease speed to a target speed within certain distance

    init_speed: the initial car speed (km/h)
    target_speed: the target car speed (km/h)
    distance: meters(m)
    max_dcl: the maximum of the deceleration rate (km/h); the limitation of the car's braking; taken as negative int
    return: deceleration rate (km/h) per one second
    """
    # if distance is less than or equal to 0, car should be stopped
    if distance <= 0:
        return init_speed * -1
    i_s = init_speed * 1000 / 3600
    t_s = target_speed * 1000 / 3600
    dcl = (t_s ** 2 - i_s ** 2) / (2 * distance)
    dcl_kmh = round(dcl * 3600 / 1000, 1)
    return max(dcl_kmh, max_dcl_kmh)


def moving_average(data, steps):
    """
    Calculate a user-given moving averages based on two inputs
    - data: the list of numbers
    - steps: an integer to retrieve data

    return a list of numbers
    """
    moving_averages = []
    for i in range(len(data)):
        if i >= steps:
            # Calculate the average of the past a user-given steps
            average = sum(data[i-steps:i])/steps
            moving_averages.append(average)
        else:
            # if there are not enough data for a user-given steps average, add NaN
            moving_averages.append(float('nan'))

    return moving_averages
