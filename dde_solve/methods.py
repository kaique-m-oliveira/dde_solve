from .cerk_step import *


class CERK3(RungeKutta):
    """ Bockagi Shampine method 3(2) with Cubic Hermite Interpolation"""

    A = np.array([
        [0, 0, 0, 0],
        [1/2, 0, 0, 0],
        [0, 3/4, 0, 0],
        [2/9, 1/3, 4/9, 0]
    ], dtype=np.float64)

    b = np.array([2/9, 1/3, 4/9, 0], dtype=np.float64)
    b_err = np.array([7/24, 1/4, 1/3, 1/8], dtype=np.float64)
    c = np.array([0, 1/2, 3/4, 1], dtype=np.float64)
    D = np.array([[0, 1, -4 / 3, 5 / 9],
                  [0, 0, 1, -2/3],
                  [0, 0, 4/3, -8/9],
                  [0, 0, -1, 1]])
    D_err = D
    D_ovl = D

    order = {
        "discrete_method": 3,
        "discrete_err_est_method": 2,
        "continuous_method": 3,
        "continuous_err_est_method": 3,
        "continuous_ovl_method": 3
    }

    n_stages = {
        "discrete_method": 4,
        # "discrete_method": 3,
        "discrete_err_est_method": 4,
        "continuous_method": 4,
        "continuous_err_est_method": 4,
        "continuous_ovl_method": 4
    }



class CERK4(RungeKutta):
    """ Method from zennaro and bellen's book Numerical methods for delay differential equations"""
    A = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1/2, 0, 0, 0, 0, 0, 0, 0],
        [0, 1/2, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [1/6, 1/3, 1/3, 1/6, 0, 0, 0, 0],
        [31/162, 7/81, 7/81, 7/162, -2/27, 0, 0, 0],
        [(37 - np.sqrt(5))/300, (7 * (1 - np.sqrt(5)))/150, (7 * (1 - np.sqrt(5))) /
         150, (7 * (1 - np.sqrt(5)))/300, (-1 + 2 * np.sqrt(5))/100, 27/100, 0, 0],
        [(np.sqrt(5) + 37)/300, (7 * (1 + np.sqrt(5)))/150, (7 * (1 + np.sqrt(5))) /
         150, (7 * (1 + np.sqrt(5)))/300, -(1 + 2 * np.sqrt(5))/100, 27/100, 0, 0]
    ], dtype=np.float64)

    b_err = np.array(
        [1/12, 0, 0, 0, 1/12, 0, 5/12, 5/12], dtype=np.float64)

    b = np.array([1/6, 1/3, 1/3, 1/6], dtype=np.float64)

    c = np.array([0, 1/2, 1/2, 1, 1, 1/3, (5 - np.sqrt(5)) /
                  10, (5 + np.sqrt(5))/10], dtype=np.float64)

    D = np.array([
        [0, 1, -3, 11/3, -3/2],
        [0, 0, -2, 16/3, -3],
        [0, 0, -2, 16/3, -3],
        [0, 0, -1, 8/3, -3/2],
        [0, 0, 5/4, -7/2, 9/4],
        [0, 0, 27/4, -27/2, 27/4]
    ], dtype=np.float64)

    D_err = np.array([
        [0, 1, -3/2, 2/3],
        [0, 0, 1, -2/3],
        [0, 0, 1, -2/3],
        [0, 0, 1/2, -1/3],
        [0, 0, -1, 1]
    ], dtype=np.float64)

    D_ovl = np.array([
        [0, 1, -3/2, 2/3],
        [0, 0, 1, -2/3],
        [0, 0, 1, -2/3],
        [0, 0, -1/2, 2/3]
    ], dtype=np.float64)

    order = {"discrete_method": 4, "discrete_err_est_method": 5,
             "continuous_method": 4, "continuous_err_est_method": 3, "continuous_ovl_method": 3}

    n_stages = {"discrete_method": 4, "discrete_err_est_method": 8,
                "continuous_method": 6, "continuous_err_est_method": 5, "continuous_ovl_method": 4}



class CERK5(RungeKutta):
    """ Dormand Prince method with 2 Hermite interpolation """
    A = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0, 0, 0],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0, 0, 0],
        [-33728713/104693760, 2, -30167461/21674880, 7739027/17448960, -19162737/123305984, 0, -26949/363520, 0, 0],
        [7157/75776, 0, 70925/164724, 10825/113664, -220887/4016128, 80069/3530688, -107/5254, -5/74, 0]
        ], dtype=np.float64)

    b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0], dtype=np.float64)

    b_err = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40], dtype=np.float64)

    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1, 1/2, 1/2], dtype=np.float64)

    D = np.array([
        [0, 1, -6839/1776, 24433/3552, -81685/14208, 29/16],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 413200/41181, -398800/13727, 1245700/41181, -4000/371],
        [0, 0, 225/37, -44725/1776, 83775/2368, -125/8],
        [0, 0, -98415/31376, 798255/62752, -4428675/251008, 6561/848],
        [0, 0, 23529/18389, -285659/55167, 527571/73556, -22/7],
        [0, 0, -3483/2627, 14847/2627, -21872/2627, 4],
        [0, 0, -40/37, 80/37, -40/37, 0],
        [0, 0, -8, 32, -40, 16]
        ], dtype=np.float64)

    D_err = np.array([
        [0, 1, -8048581381/2820520608, 8663915743/2820520608, -12715105075/11282082432],
        [0, 0, 0, 0, 0],
        [0, 0, 131558114200/32700410799, -68118460800/10900136933, 87487479700/32700410799],
        [0, 0, -1754552775/470086768, 14199869525/1410260304, -10690763975/1880347072],
        [0, 0, 127303824393/49829197408, -318862633887/49829197408, 701980252875 / 199316789632],
        [0, 0, -282668133/205662961, 2019193451/616988883, -1453857185/822651844],
        [0, 0, 40617522/29380423, -110615467/29380423, 69997945/29380423]])


    D_ovl = D_err

    order = {
        "discrete_method": 5,
        "discrete_err_est_method": 4,
        "continuous_method": 5,
        "continuous_err_est_method": 4,
        "continuous_ovl_method": 4
    }

    n_stages = {
        "discrete_method": 7,
        "discrete_err_est_method": 7,   
        "continuous_method": 9,   
        "continuous_err_est_method": 7,
        "continuous_ovl_method": 7
    }

