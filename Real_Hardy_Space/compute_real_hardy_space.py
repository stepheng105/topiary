import numpy as np
import heapq
import matplotlib.pyplot as plt
from modern_portfolio_theory.topiary import topiary_main_algorithm
from modern_portfolio_theory.empirical_solution_helpers import max_eta
import time
import math
from PIL import Image, ImageFile
from packaging import version
import imageio
from scipy.spatial import cKDTree
import matplotlib.colors as mcolors


def generate_random_points(num_points):
    # Generate random points in the unit disk according to some complicated distribution
    agg = np.random.rand(num_points)-1/2
    agg = agg/np.abs(agg)
    theta = ((40/50) * np.pi * np.random.rand(num_points)+20/50)*agg  # Angle
    r0 = 23/30 + np.cos(20*theta)/120 + 6*np.cos(7*theta)/120 - np.sin(3*theta)/60 - np.cos(theta)/3
    #r0=1
    r =np.sqrt(10*np.random.rand(num_points)/36+25/36)*r0  # Radius
    r=r*.8
    #theta = np.abs(theta+r*agg)
    # r = r*np.exp(-np.abs(theta-np.pi)/5)
    z = r * np.exp(1j * theta)  # Convert to complex numbers
    #z=(np.random.rand(num_points)+2.2*agg-1/2) /8 + 1j * (np.random.rand(num_points)+agg2*1.5-1/2) /8
    #z=(z-1/3)/(1-z/3)
    z=z+.6
    return z

def generate_random_points_2(num_points):
    # Generate random points in the unit disk according to some complicated distribution
    agg = np.random.rand(num_points)-1/2
    agg = agg/np.abs(agg)
    theta = ((40/50) * np.pi * np.random.rand(num_points)+20/50)*agg  # Angle
    r0 = 23/30 + np.cos(20*theta)/120 + 6*np.cos(7*theta)/120 - np.sin(3*theta)/60 - np.cos(theta)/3
    #r0=1
    r =np.sqrt(10*np.random.rand(num_points)/36+25/36)*r0  # Radius
    r=r*.8
    theta = np.abs(theta+r*agg)
    r = r*np.exp(-np.abs(theta-np.pi)/5)
    # z = r * np.exp(1j * theta)  # Convert to complex numbers
    z=(np.random.rand(num_points)+2.2*agg-1/2) /8 + 1j * (np.random.rand(num_points)+agg*1.5-1/2) /8
    z=(z-1/3)/(1-z/3)
    z=z+.6
    return z

def generate_random_points_pi(num_points):
    points = np.array([0] * num_points, dtype=np.complex_)
    for i in range(num_points):
        done = False
        while not done:
            p = np.random.rand(2)
            x = np.sqrt(p[1]) * np.cos(2 * np.pi * p[0])
            y = np.sqrt(p[1]) * np.sin(2 * np.pi * p[0])
            z = x + 1j * y
            if (((x-.2)**2 + (y+.05+.3)**2 <= 0.0125) or (x+.4)**2 + (y+.05+.3)**2 <= 0.0125
                    or (-.1*x +.4 -.3 <= y and y <= -100*math.exp(-8*x-11)+.7+.01*x -.3 and x <= -0.525869)
                    or (-100*math.exp(-8*x-11) + .1 + .01*x+.47 -.3 <= y and y <= -100*math.exp(-8*x-11)+.7+.01*x -.3 and -0.525869 <= x and x <= .6)
                    or (math.tan(3*x + .79665217700435547) -.3 <= y and y <= math.tan(2*x + 1) - .3 and -0.2938491 <= x and x <= -0.246457)
                    or (-0.412802*x - 0.206401 -.3 <= y and y <= math.tan(2*x + 1) -.3 and -.5 <= x and x <= -0.2938491)
                    or (math.tan(3*x + .79665217700435547) -.3 <= y and y <= -100*math.exp(-8*x-11) + .1 + .01*x + .47 -.3 and -0.246457 <= x and x <= -0.093991)
                    or (-0.369078*x+0.0369078 -.3 <= y and y <= math.tan(2*x-.2) -.3 and .1 <= x and x <= 0.308505)
                    or (math.tan(4*x - 1.3108253396761014) -.3 <= y and y <= math.tan(2*x - .2) -.3 and 0.308505 <= x and x <= 0.360357)
                    or (math.tan(4*x - 1.3108253396761014) -.3 <= y and y <= -100*math.exp(-8*x-11) + .1 + .01*x + .47 -.3 and 0.360357 <= x and x <= 0.458078)):
                done = True
        points[i] = z
    return points

def generate_random_points_julia(num_points):
    # z^2 + c
    points = np.array([0] * num_points, dtype=np.complex_)
    c = -0.4 + 0.6j
    a = math.sqrt(c.real**2 + c.imag**2)
    R = round(math.ceil((1 + math.sqrt(1 + 4*a))*50)/100, 2)
    max_iteration = 1000
    for i in range(num_points):
        done = False
        while not done:
            p = np.random.rand(2)
            x = np.sqrt(p[1]) * np.cos(2 * np.pi * p[0])
            y = np.sqrt(p[1]) * np.sin(2 * np.pi * p[0])
            z = x + 1j * y
            z_temp = z.real + 1j*z.imag
            iteration = 0
            while z_temp.real * z_temp.real + z_temp.imag * z_temp.imag < R*R and iteration < max_iteration:
                xtemp = z_temp.real * z_temp.real - z_temp.imag * z_temp.imag
                z_temp = xtemp + c.real + 1j * (2*z_temp.real * z_temp.imag + c.imag)
                iteration += 1
            if iteration == max_iteration:
                done = True
        points[i] = z
    return points

def generate_random_points_heart(num_points):
    points = np.array([0] * num_points, dtype=np.complex_)
    for i in range(num_points-1):
        done = False
        while not done:
            p = np.random.rand(2)
            x = np.sqrt(p[1]) * np.cos(2 * np.pi * p[0])
            y = np.sqrt(p[1]) * np.sin(2 * np.pi * p[0])
            z = x + 1j * y
            C = 1 - (4*(x**2))**(1/3)
            if C*((19-8*C)**2) - (6 - 2*C - 8 * (C)**2 - 32*y)**2 >= 0:
                done = True
        points[i] = z
    points = np.append(points, [0 + 1j * 0])
    return points

def generate_random_points_donut(num_points):
    points = np.array([0] * num_points, dtype=np.complex_)
    for i in range(num_points):
        done = False
        while not done:
            p = np.random.rand(2)
            x = np.sqrt(p[1]) * np.cos(2 * np.pi * p[0])
            y = np.sqrt(p[1]) * np.sin(2 * np.pi * p[0])
            z = x + 1j * y
            if x**2 + y**2 >= .3 and x**2 + y**2 <= .6:
                done = True
        points[i] = z
    return points

def generate_random_points_shifted_donut(num_points):
    points = np.array([0] * num_points, dtype=np.complex_)
    for i in range(num_points):
        done = False
        while not done:
            p = np.random.rand(2)
            x = np.sqrt(p[1]) * np.cos(2 * np.pi * p[0])
            y = np.sqrt(p[1]) * np.sin(2 * np.pi * p[0])
            z = x + 1j * y
            if x**2 + (y-.1)**2 >= .3 and x**2 + (y-.1)**2 <= .6:
                done = True
        points[i] = z
    return points


def generate_random_points_maze(num_points):
    points = np.array([0]*num_points, dtype=np.complex_)
    for i in range(num_points):
        done = False
        while not done:
            p = np.random.rand(2)
            x = np.sqrt(p[1]) * np.cos(2 * np.pi * p[0])
            y = np.sqrt(p[1]) * np.sin(2 * np.pi * p[0])
            z = x + 1j*y
            if (((abs(z) <= .7 and abs(z) >= .55 and (x <= 0 or abs(y)>=.1))
                    or (.3 <= x and x <= .4 and abs(y) <= .2))
                    or (abs(z) <= .1 and abs(z) >= .5 and (y >= 0 or abs(x) >= .03)) or
                    (abs(x) <= .06 and .3 <= y and y <= .6) or
                    (.3 <= y and y <= .4 and -.25 <= x and x <= 0) or
                    (abs(y) <= .04 and -.6 <= x and x <= -.3) or
                    (x-y <= .6 and .4 <= x-y and x <= .2 and y >= -.6) or
                    (3*x + y >= .1 and 3*x + y <= .2 and -.05 >= y and y >= -.3) or
                    (y-x <= .1 and y-x >= -.01 and y>=-.3 and x <= -.06)):
                done = True
        points[i] = z
    return points

def generate_random_points_i_love_comittee(num_points):
    points = np.array([0]*num_points, dtype=np.complex_)
    for i in range(num_points):
        done = False
        while not done:
            p = np.random.rand(2)
            x = np.sqrt(p[1]) * np.cos(2 * np.pi * p[0])
            y = np.sqrt(p[1]) * np.sin(2 * np.pi * p[0])
            z = x + 1j*y
            # Good luck code golfers!
            # If statement so long it lags my IDE LMAO
            if ((.85 <= y and y <= .9 and -.2 <= x and x <= .2)
                    or (-.025 <= x and x <= .025 and .6 <= y and y <= .85)
                    or (.55 <= y and y <= .6 and -.2 <= x and x <= .2)
                    or ((1 - ((16*(x**2))**(1/3)))*(19-8*(1 - ((16*(x**2))**(1/3))))**2
                        - (6-2*(1 - ((16*(x**2))**(1/3)))-8*(1 - ((16*(x**2))**(1/3)))**2-16*4*(y-.3))**2 >= 1)
                    or (-.9 <= x and x <= -.85 and -.35 <= y and y <= -.05)
                    or (-.6 <= x and x <= -.55 and -.35 <=y and y <= -.05)
                    or (-x -.95 <= y and y <= -x - .9 and -.85 <= x and x <= -.725)
                    or (x+.5 <= y and y <= x + .55 and -.725 <= x and x <= -.6)
                    or (-.55-y <= x and x <= -.5-y and -.15 <= y and y <= -.05)
                    or (-.4 <= x and x <= -.35 and -.35 <= y and y <= -.15)
                    or (-.2 + y >= x and x >= -.25 + y and -.15 <=y and y <= -.05)
                    or (-.1 <= y and y <= -.05 and -.2 <= x and x <=.05)
                    or (-.1 <= x and x <= -.05 and -.35 <=y and y <= -.1)
                    or (.1 <= x and x <= .15 and -.35 <=y and y <= -.05)
                    or (.25 <= x and x <= .3 and -.35 <=y and y <= -.05)
                    or (-.175 >= y and y >= -.225 and .15 <=x and x <=.25)
                    or (.35 <=x and x <= .4 and -.35 <=y and y <= -.05)
                    or ( -.175 >=y and y >= -.225 and .4 <= x and x <= .5)
                    or (-.1 <= y and y <= -.05 and .4 <= x and x <= .5)
                    or ( -.35 <=y and y <= -.3 and .4 <= x and x <=.5)
                    or (3 * (x-.575)**2 + (y+ .1375)**2 <= 0.00765625 and 3 * (x-.575)**2 + (y+ .1375)**2 >= 0.00140625 and x <= .575)
                    or (3 * (x-.575)**2 + (y+0.2625)**2 <= 0.00765625 and 3 * (x-.575)**2 + (y+0.2625)**2 >= 0.00140625 and x >= .575)
                    or (-.1 <= y and y <= -.05 and .66 <= x and x <= .8 )
                    or (-.35 <=y and y <= -.3 and .66 <= x and x <= .8)
                    or (.71 <=x and x <= .75 and -.3 <=y and y <= -.1)
                    or (3*(x-.875)**2 + (y + 0.1375)**2 <= 0.00765625 and 3*(x-.875)**2 + (y + 0.1375)**2 >= 0.00140625 and x <= .875)
                    or (3 * (x - .875)**2 + (y+0.2625)**2 <= 0.00765625 and 3 * (x - .875)**2 + (y+0.2625)**2 >= 0.00140625 and x >= .875)
                    or ((x+.7)**2 + (y+.5)**2 <= .01 and (x+.7)**2 + (y+.5)**2 >= .003 and x <= -.66)
                    or ((x+.53)**2 + (y+.5)**2 <= .01 and (x+.53)**2 + (y+.5)**2 >= .003)
                    or (-.4 <= x and x <= -.35 and -.6 <= y and y <= -.4)
                    or (-2*(x+.35) - .45 <= y and y <= -2*(x+.35) - .4 and -.5 <=y and y <= -.4)
                    or (2*(x+.3)-.5 <= y and y<= 2*(x+.325) -.5 and -.5 <= y and y <= -.4)
                    or (-.275 <= x and x <= -.225 and -.6 <= y and y <= -.4)
                    or (-.16 <= x and x <= -.11 and -.6 <= y and y <= -.4)
                    or (-2 * (x+0.11) - .45 <= y and y <= -2*(x+0.11) -.4 and -.5 <= y and y <= -.4)
                    or (2* (x+.06) - .5 <=y and y <= 2*(x+.085) - .5 and -.5 <=y and y <= -.4)
                    or (-.035 <= x and x <= .015 and -.6 <= y and y <= -.4)
                    or (.1 <= x and x <= .15 and -.55 <= y and y <= -.45)
                    or (-.45 <= y and y <= -.4 and .05 <= x and x <= .2)
                    or (-.6 <= y and y <= -.55 and .05 <=x and x <= .2)
                    or (.275 <= x and x <= .325 and -.6 <= y and y <= -.45)
                    or (-.45 <= y and y <= -.4 and .225 <= x and x <= .375)
                    or (.45 <= x and x <= .5 and -.6 <=y and y <= -.45)
                    or (-.45 <=y and y <= -.4 and .4 <= x and x <= .55)
                    or (.57 <= x and x <= .62 and -.6 <= y and y <= -.4)
                    or (-.45 <= y and y <= -.4 and .62 <= x and x <= .675)
                    or (-.6 <= y and y <= -.55 and .62 <= x and x <= .675)
                    or (-.525 <= y and y <= -.475 and .62 <= x and x <= .675)
                    or (.7 <= x and x <= .74 and -.6 <= y and y <= -.4)
                    or (-.45 <= y and y <= -.4 and .74 <= x and x <= .79)
                    or (-.6 <= y and y <= -.55 and .74 <= x and x <= .79)
                    or (-.525 <= y and y <= -.474 and .74 <= x and x <= .79)):
                done = True
        points[i] = z
    return points

def generate_random_points_disk(num_points):
    # Generate random points in the unit disk
    theta = 2 * np.pi * np.random.rand(num_points)  # Angle
    r = np.sqrt(np.random.rand(num_points))  # Radius
    z = r * np.exp(1j * theta)  # Convert to complex numbers
    return z

def compute_gram_matrix(points):
    num_points = points.shape[0]

    # Compute z_bar and w
    z_bar = np.conj(points)
    w = points[:, np.newaxis]
    outer =z_bar * w


    # Compute Gram matrix for real hardy space
    #gram_matrix = np.real(((1 + outer) / (1 - outer))+((10*outer +1) / (10*outer - 1)))
    gram_matrix =np.real(((1 + outer) / (1 - outer)))
    return gram_matrix


def compute_real_hardy_space(num_points, num_points_fill, precision, region_type, compute_scipy, create_gif=False):
    t0 = time.time()
    print("generating random points")
    if region_type == 'pascoe':
        points = generate_random_points(num_points)
    elif region_type == 'weird':
        points = generate_random_points_2(num_points)
    elif region_type == 'pi':
        points = generate_random_points_pi(num_points)
    elif region_type == 'heart':
        points = generate_random_points_heart(num_points)
    elif region_type == 'maze':
        points = generate_random_points_maze(num_points)
    elif region_type == 'committee':
        points = generate_random_points_i_love_comittee(num_points)
    elif region_type == 'donut':
        points = generate_random_points_donut(num_points)
    elif region_type == 'shifted_donut':
        points = generate_random_points_shifted_donut(num_points)
    elif region_type == 'julia':
        points = generate_random_points_julia(num_points)
    else:
        raise Exception('Unknown region type "' + region_type + '". Please choose from "pascoe", "weird", "pi", "heart", "maze", "donut", "shifted_donut", "julia", or "committee".')

    print("Time taken to generate points:", time.time() - t0)
    gram_matrix = compute_gram_matrix(points)
    cov_matrix = np.array(gram_matrix)
    starting_col = np.argmin(cov_matrix.diagonal())
    starting_weight = np.zeros(cov_matrix[0].shape)
    starting_weight[starting_col] = 1

    t0 = time.time()
    print("computing topiary")
    if create_gif:
        mu, past_mus = topiary_main_algorithm(np.zeros(cov_matrix[0].shape), cov_matrix, 'max', precision, True, 0, keep_track=True)
    else:
        mu, past_mus = topiary_main_algorithm(np.zeros(cov_matrix[0].shape), cov_matrix, 'max', precision, True, 0, keep_track=False)
        # r = np.abs(points)
        # angle = np.angle(points)
        # psi = (.5 -1j)*np.square(r) * np.exp(-2j * angle) + (1+1j)*r * np.exp(-1j * angle)+.1+(1 -1j)*r * np.exp(1j * angle) + (.5 +1j)*np.square(r) * np.exp(2j * angle)
        # mu, past_mus = topiary_main_algorithm(np.zeros(cov_matrix[0].shape), cov_matrix, 'max', precision, True, 0,
        #                                       keep_track=False)
    print("Time taken to compute topiary:", time.time() - t0)

    newpoints = generate_random_points_disk(num_points_fill)
    wx = newpoints[:, np.newaxis]

    if create_gif:
        images = []
        nu_prev = np.zeros_like(mu)
        for i in range(len(past_mus)):
            nu = past_mus[i]
            last_added_point = np.argmax(nu - nu_prev)
            nonzero_indexes = np.nonzero(nu)
            goodpoints = points[nonzero_indexes]
            norm_nu = nu.transpose() @ (cov_matrix @ nu)

            z_barx = np.conj(goodpoints)
            outerx = z_barx * wx
            kernx = nu[nonzero_indexes].T @ np.real((1 + outerx) / (1 - outerx)).T
            with np.errstate(divide='ignore', invalid='ignore'):
                kernx_temp = np.true_divide(norm_nu - kernx, np.abs(norm_nu - kernx))
                kernx_temp[kernx_temp == np.inf] = 0
                kernx_temp = np.nan_to_num(kernx_temp)
            print("NUM POINTS USED:", len(goodpoints))
            print("NORM:", norm_nu)
            pos_indices = []
            neg_indices = []
            zero_indices = []
            for i in range(len(kernx_temp)):
                if kernx_temp[i] > 0:
                    pos_indices.append(i)
                elif kernx_temp[i] < 0:
                    neg_indices.append(i)
                else:
                    zero_indices.append(i)

            np.conj(goodpoints)
            plt.figure()
            plt.scatter(newpoints[pos_indices].real, newpoints[pos_indices].imag, color='yellow', s=2,
                        label='Random Points')
            plt.scatter(newpoints[neg_indices].real, newpoints[neg_indices].imag, color='purple', s=2,
                        label='Random Points')
            plt.scatter(newpoints[zero_indices].real, newpoints[zero_indices].imag, color='blue', s=2,
                        label='Random Points')
            plt.scatter(points.real, points.imag, color="pink", s=3, label='Random Points')
            plt.scatter(goodpoints.real, goodpoints.imag, color='red', s=3, label='Random Points')
            plt.scatter(points[last_added_point].real, points[last_added_point].imag, color='limegreen', s=3, label='Random Points')
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig('Topiary_Hardy.png', dpi=100)
            images.append(imageio.imread('Topiary_Hardy.png'))
            nu_prev = nu.copy()
        imageio.mimsave('animated_plot.gif', images)


    nonzero_indexes = np.nonzero(mu)
    goodpoints = points[nonzero_indexes]
    print("mu: " + str(mu[nonzero_indexes]))
    print("points: " + str(goodpoints))
    norm_mu = mu.transpose() @ (cov_matrix @ mu)

    z_barx = np.conj(goodpoints)
    outerx = z_barx * wx
    kernx = mu[nonzero_indexes].T @ np.real((1 + outerx) / (1 - outerx)).T
    with np.errstate(divide='ignore', invalid='ignore'):
        kernx_temp = np.true_divide(norm_mu-kernx, np.abs(norm_mu-kernx))
        kernx_temp[kernx_temp == np.inf] = 0
        kernx_temp = np.nan_to_num(kernx_temp)

    print("NUM POINTS USED:", len(goodpoints))
    print("NORM:", norm_mu)
    pos_indices = []
    neg_indices = []
    zero_indices = []
    for i in range(len(kernx_temp)):
        if kernx_temp[i] > 0:
            pos_indices.append(i)
        elif kernx_temp[i] < 0:
            neg_indices.append(i)
        else:
            zero_indices.append(i)

    np.conj(goodpoints)
    plt.figure()
    plt.scatter(newpoints[pos_indices].real, newpoints[pos_indices].imag, color='yellow', s=2,
                    label='Random Points')
    plt.scatter(newpoints[neg_indices].real, newpoints[neg_indices].imag, color='purple', s=2,
                    label='Random Points')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(newpoints[zero_indices].real, newpoints[zero_indices].imag, color='blue', s=2,
                label='Random Points')
    plt.savefig('Topiary_Hardy_2.png', dpi=300)
    plt.scatter(points.real, points.imag, color="pink", s=3, label='Random Points')
    plt.scatter(goodpoints.real, goodpoints.imag, color='red', s=3, label='Random Points')
    plt.savefig('Topiary_Hardy.png', dpi=300)
    if compute_scipy:
        t0 = time.time()
        mu_scipy = max_eta(np.zeros(cov_matrix[0].shape), cov_matrix, 0, precision)
        print("Time taken to compute scipy:", time.time() - t0)

        nonzero_indexes = np.nonzero(mu_scipy)
        goodpoints = points[nonzero_indexes]
        newpoints = generate_random_points_disk(num_points_fill)
        norm_mu = mu_scipy.transpose() @ (cov_matrix @ mu_scipy)

        z_barx = np.conj(goodpoints)
        wx = newpoints[:, np.newaxis]
        outerx = z_barx * wx
        kernx = mu_scipy[nonzero_indexes].T @ np.real((1 + outerx) / (1 - outerx)).T
        print("NORM" + str(norm_mu))
        kernx = (kernx - norm_mu) / np.abs(kernx - norm_mu)
        pos_indices = []
        neg_indices = []
        for i in range(len(kernx)):
            if kernx[i] >= 0:
                pos_indices.append(i)
            else:
                neg_indices.append(i)

        np.conj(goodpoints)
        plt.figure()
        plt.scatter(newpoints[pos_indices].real, newpoints[pos_indices].imag, color='yellow', s=2, label='Random Points')
        plt.scatter(newpoints[neg_indices].real, newpoints[neg_indices].imag, color='purple', s=2, label='Random Points')
        plt.scatter(points.real, points.imag, color="pink", s=5, label='Random Points')
        plt.scatter(goodpoints.real, goodpoints.imag, color='red', s=10, label='Random Points')
        plt.savefig('SciPy_Hardy.png', dpi=300)


if __name__ == '__main__':
    compute_real_hardy_space(10000, 1**(-10), compute_scipy=False)