#! /usr/bin/env python3
"""
kinematics classes for 3 DOF robots
@author: Achille
"""
import numpy as np
from numpy import cos, sin, arctan2, pi, arcsin, arccos
np.seterr(all='raise')  # raise error iso warning
from random import randint
from angles import normalize_angle
import sympy

DEFAULT_SIN_COS_TOLERANCE = 0.01


class ThreeDOFKinematics(object):

    @classmethod
    def origins(cls, thetas):
        """Return the positions of each of the origins o1 -> o3
        :param thetas: list of theta1 -> theta3
        :returns: np.array of size 3 by 3, each row a position
        """
        # grab the first three elements from the last column of all poses
        return cls.forward_kinematics(thetas)[:, 0:3, 3]

    @classmethod
    def end_effector_pose(cls, thetas):
        """Forward kinematics to get the end-effector pose"""
        # grab H_0_3
        return cls.forward_kinematics(thetas)[2, :, :]

    @classmethod
    def forward_kinematics(cls, thetas):
        """Forward kinematics to get all origin poses
        :param thetas: list of theta1 -> theta3
        :returns: np.array of size 3 by 4 by 4
        """
        A1 = cls.dh_matrix(cls.a1, cls.alpha1, cls.d1, thetas[0])
        A2 = cls.dh_matrix(cls.a2, cls.alpha2, cls.d2, thetas[1])
        A3 = cls.dh_matrix(cls.a3, cls.alpha3, cls.d3, thetas[2])

        H_0_2 = np.matmul(A1, A2)
        H_0_3 = np.matmul(H_0_2, A3)

        return np.array([A1, H_0_2, H_0_3])

    @classmethod
    def random_valid_pose(cls):
        """Return a random valid end-effector pose
        Values are within joint limits (for unconstrained joints: [-pi, pi])
        """
        return cls.end_effector_pose(cls.random_valid_config())

    @classmethod
    def random_valid_config(cls):
        """Return a random valid set of joint values
        Values are within joint limits (for unconstrained joints: [-pi, pi])
        """
        random_valid_joints = np.zeros(3)
        for joint_ind in range(3):
            (llimit, ulimit) = cls.joint_limits_feasible[joint_ind]
            random_valid_joint_val = np.random.rand() * (ulimit - llimit) - abs(
                llimit)  # scale from 0-1 to limit ranges
            random_valid_joints[joint_ind] = random_valid_joint_val

        return random_valid_joints

    @classmethod
    def dh_matrix(cls, a, alpha, d, theta):
        """Return the homogeneous transformation matrix based off the given DH parameters
        :param a, alpha, d, theta: standard convention DH parameters
        :returns: np.array of size 4 by 4
        """
        return np.array([[cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta)],
                         [sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
                         [         0,             sin(alpha),             cos(alpha),            d],
                         [         0,                      0,                      0,            1]])


    @classmethod
    def ik(cls, target_pos):
        """
        Solve the closed form IK
        :param target_pos: 3 by 1 np.array that describes the desired position of the end-effector wrt 
                           the the arm's base frame 
        :return: list of lists of joint angles defined wrt the DH frames
        """
        thetas = [0, 0, 0]

        return thetas


class OrthoManip3RKinematics(ThreeDOFKinematics):
    joint_limits = pi/180*np.array([[-np.inf, np.inf],
                                    [-np.inf, np.inf],
                                    [-np.inf, np.inf]])
    joint_limits_feasible = pi/180*np.array([[-180, 180],
                                    [-180,       180],
                                    [-180,       180]])

    @classmethod
    def determinant_jacobian(cls, theta2, theta3):
        return cls.a3*(cls.a2 + cls.a3*sympy.cos(theta3))*(cls.a1*sympy.sin(theta3) - cls.d2*sympy.cos(theta2)*sympy.cos(theta3) + cls.a2*sympy.cos(theta2)*sympy.sin(theta3))

    @classmethod
    def quartic(cls, rho, zee):
        R = rho**2 + zee**2
        L = cls.a3**2 + cls.a2**2 + cls.d2**2
        m0 = -R + zee**2 + cls.d2**2 + ((R+1-L)**2)/4
        m1 = 2*cls.d2*cls.a3 + (L - R - 1)*cls.a3*cls.d2
        m2 = (L - R - 1)*cls.a3*cls.a2
        m3 = 2*cls.d2*cls.a2*cls.a3**2
        m4 = cls.a3**2*(cls.d2**2 + 1)
        m5 = cls.a3**2 * cls.a2**2

        a = m5-m2+m0
        b = -2*m3+2*m1
        c = -2*m5+4*m4+2*m0
        d = 2*m3+2*m1
        e = m5+m2+m0

        return a, b, c, d, e

    @classmethod
    def quartic_discriminant(cls, rho, zee):
        a, b, c, d, e = cls.quartic(rho, zee)

        discriminant = 256*a**3*e**3 - 192*a**2*b*d*e**2 - 128*a**2*c**2*e**2 + 144*a**2*c*d**2*e - 27*a**2*d**4 + 144*a*b**2*c*e**2 - 6*a*b**2*d**2*e \
                    - 80*a*b*c**2*d*e + 18*a*b*c*d**3 + 16*a*c**4*e - 4*a*c**3*d**2 - 27*b**4*e**2 + 18*b**3*c*d*e - 4*b**3*d**3 - 4*b**2*c**3*e + b**2*c**2*d**2

        return discriminant

    @classmethod
    def delta_0(cls, rho, zee):
        a, b, c, d, e = cls.quartic(rho, zee)
        return c**2 - 3*b*d + 12*a*e

    @classmethod
    def delta_1(cls, rho, zee):
        a, b, c, d, e = cls.quartic(rho, zee)
        return 2*c**3 - 9*b*c*d + 27*b**2*e + 27*a*d**2 - 72*a*c*e

    @classmethod
    def is_cuspidal(cls, rho, zee):
        """Use this with symbolic rho, zee"""
        delta_0 = cls.delta_0(rho, zee)
        print("is_cuspidal: delta_0 = {}".format(delta_0))
        
        return delta_0 == 0

    def ik(cls, xee, yee, zee):
        """
        Inverse Kinematics for a given rho, zee. Does not work for symbolic values

        Does not implement all edge cases properly.
        """
        rho = np.sqrt(xee**2 + yee**2)

        a, b, c, d, e = cls.quartic(rho, zee)

        delta_0 = cls.delta_0(rho, zee)
        delta_1 = cls.delta_1(rho, zee)

        if delta_1 == 0:
            raise NotImplementedError()

        p = (8*a*c - 3*b**2) / (8*a**2)
        q = (b**3 - 4*a*b*c + 8*a**2*d) / (8*a**3)

        discr = cls.quartic_discriminant(rho, zee)
        if discr < 0:
            Q = np.cbrt((delta_1 + np.sqrt(delta_1**2 - 4*delta_0**3)) / 2)
            S = 0.5 * np.sqrt(-2./3*p + 1./(3*a) * (Q + delta_0/Q))
        elif discr > 0:
            phi = arccos(delta_1 / (2 * np.sqrt(delta_0**3)))
            S = 0.5 * np.sqrt(-2./3*p + 2./(3*a) * np.sqrt(delta_0) * cos(phi/3))
            # S = 0.5 * np.sqrt(2*)
        else:
            raise NotImplementedError()

        # try replacing these with Wolfram Alpha's roots 
        # https://mathworld.wolfram.com/QuarticEquation.html 
        t1 = -b/(4*a) - S + 0.5*np.sqrt(-4*S**2 - 2*p + q/S)
        t2 = -b/(4*a) - S - 0.5*np.sqrt(-4*S**2 - 2*p + q/S)
        t3 = -b/(4*a) + S + 0.5*np.sqrt(-4*S**2 - 2*p - q/S)
        t4 = -b/(4*a) + S - 0.5*np.sqrt(-4*S**2 - 2*p - q/S)

        th3_1 = 2 * np.arctan(t1)
        th3_2 = 2 * np.arctan(t2)
        th3_3 = 2 * np.arctan(t3)
        th3_4 = 2 * np.arctan(t4)

        th2_1 = arcsin(-zee / (cls.a2 + cls.a3 * cos(th3_1)))
        th2_2 = arcsin(-zee / (cls.a2 + cls.a3 * cos(th3_2)))
        th2_3 = arcsin(-zee / (cls.a2 + cls.a3 * cos(th3_3)))
        th2_4 = arcsin(-zee / (cls.a2 + cls.a3 * cos(th3_4)))

        l1 = cls.a1 + cls.a3*cos(th3_1)*cos(th2_1) + cls.a2*cos(th2_1)
        l2 = cls.a1 + cls.a3*cos(th3_2)*cos(th2_2) + cls.a2*cos(th2_2)
        l3 = cls.a1 + cls.a3*cos(th3_3)*cos(th2_3) + cls.a2*cos(th2_3)
        l4 = cls.a1 + cls.a3*cos(th3_4)*cos(th2_4) + cls.a2*cos(th2_4)
        m1 = cls.d2 + cls.a3*sin(th3_1)
        m2 = cls.d2 + cls.a3*sin(th3_2)
        m3 = cls.d2 + cls.a3*sin(th3_3)
        m4 = cls.d2 + cls.a3*sin(th3_4)

        # th1_1 = arccos((l1*xee + m1*yee) / (l1**2 + m1**2))
        # th1_2 = arccos((l2*xee + m2*yee) / (l2**2 + m2**2))
        # th1_3 = arccos((l3*xee + m3*yee) / (l3**2 + m3**2))
        # th1_4 = arccos((l4*xee + m4*yee) / (l4**2 + m4**2))

        # th1_1 = arcsin((l1*yee - m1*xee) / (l1**2 + m1**2))
        # th1_2 = arcsin((l2*yee - m2*xee) / (l2**2 + m2**2))
        # th1_3 = arcsin((l3*yee - m3*xee) / (l3**2 + m3**2))
        # th1_4 = arcsin((l4*yee - m4*xee) / (l4**2 + m4**2))

        th1_1 = arctan2(l1*yee - m1*xee, l1*xee + m1*yee)
        th1_2 = arctan2(l2*yee - m2*xee, l2*xee + m2*yee)
        th1_3 = arctan2(l3*yee - m3*xee, l3*xee + m3*yee)
        th1_4 = arctan2(l4*yee - m4*xee, l4*xee + m4*yee)

        solns = [[th1_1, th2_1, th3_1],
                 [th1_2, th2_2, th3_2],
                 [th1_3, th2_3, th3_3],
                 [th1_4, th2_4, th3_4]]

        return solns


    def forward_kinematics_2D(self, thetas):
        ee_pos = self.origins(thetas)[-1, :]
        rho = np.sqrt(ee_pos[0]**2 + ee_pos[1]**2)
        
        return rho, ee_pos[2]


class Manipulator1(OrthoManip3RKinematics):
    # DH parameters. All a's and alphas are zero.
    d1 = 0
    d2 = 1
    d3 = 0
    alpha1 = -pi/2
    alpha2 = pi/2
    alpha3 = 0
    a1 = 1.
    a2 = 2
    a3 = 1.5


class Manipulator2(OrthoManip3RKinematics):
    # DH parameters. All a's and alphas are zero.
    d1 = 0
    d2 = 0.7
    d3 = 0
    alpha1 = -pi/2
    alpha2 = pi/2
    alpha3 = 0
    a1 = 1.
    a2 = 1.5
    a3 = 0.7


def minmax(val, tol = DEFAULT_SIN_COS_TOLERANCE, limit = 1):
    """ 
    If -(limit + tol) <= val <= (limit + tol), val is bounded between (-limit, +limit)
    else val remains unchanged
    """
    val_minmax = val
    if (abs(val)) <= limit + tol:
        val_minmax = min(val, limit)
        val_minmax = max(val_minmax, -1*limit)
    return val_minmax


if __name__ == '__main__':
    solver = Manipulator1()
    # random_valid_target = solver.random_valid_ee_pose()
    xee = 1.79
    yee = 1.75
    zee = 0.6
    print("rho original: {}".format(np.sqrt(xee**2 + yee**2)))

    all_solns = solver.ik(xee, yee, zee)
    for soln in all_solns:
        x, y, z = solver.end_effector_pose(soln)[:3, 3]
        print("angles: {}".format(soln))
        print("x: {}, y: {}, z: {}".format(x,y,z))
        print("rho soln: {}".format(np.sqrt(x**2 + y**2)))
