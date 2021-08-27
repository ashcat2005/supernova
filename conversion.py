# Conversion between orbital parameters and initial conditions

from numpy.linalg import norm
from numpy import pi, sqrt, array, dot, cross, transpose
from numpy import sin, cos, arcsin, arccos, isnan
from scipy.optimize import newton


################################################################################
# Global Constants
################################################################################


################################################################################

def op_to_coords(mu, a, ecc, i, omega, Omega, t_0, t):
    '''
    Transforms from the orbital parameters to the coordinates of the position
    vector and the components of the velocity at time t.
    Verify the system of units for the arguments mu, a and t.
    ----------------------------------------------------------------------------
    Arguments: a, ecc, i, omega, Omega, t_0, t
    ----------------------------------------------------------------------------
    Returns: position (array) , velocity (array)
    ----------------------------------------------------------------------------
    '''
    # Mean Motion
    n = sqrt(mu/a**3)
    # Mean Anomaly
    M = n*(t-t_0)
    # Eccentric Anomaly
    # Kepler problem
    f = lambda x: x - ecc*sin(x) - M
    f_prime = lambda x: 1. - ecc*cos(x)
    E = newton(f, 0., f_prime)
    # Radial coordinate at time t
    r = a*(1-ecc*cos(E))
    # Coordinates of the body at time t in the orbital plane
    r_XYZ = array([a*(cos(E) - ecc), a*sqrt(1 - ecc**2)*sin(E), 0.])
    # Components of the velocity of the body at time t in the orbital plane
    v_XYZ = array([-(sqrt(mu*a)/r)*sin(E),
                   (sqrt(mu*a*(1 - ecc**2))/r)*cos(E),
                   0.])
    # Rotation Matrix to transform the components of a vector from the
    # orbital plane to the general reference system
    R = array([[cos(omega)*cos(Omega) - sin(omega)*sin(Omega)*cos(i),
                -sin(omega)*cos(Omega) - cos(omega)*sin(Omega)*cos(i),
                sin(Omega)*sin(i)],
               [cos(omega)*sin(Omega) + sin(omega)*cos(Omega)*cos(i),
                -sin(omega)*sin(Omega) + cos(omega)*cos(Omega)*cos(i),
                -cos(Omega)*sin(i)],
               [sin(omega)*sin(i),
                cos(omega)*sin(i),
                cos(i)]])
    
    return R.dot(r_XYZ), R.dot(v_XYZ)


def coords_to_op(mu, r_xyz, v_xyz):
    '''
    Transforms from the coordinates of the position and the components of the
    velocity at t=0. to the orbital parameters.
    Verify the system of units for the arguments.
    ----------------------------------------------------------------------------
    Arguments: position (array) , velocity (array)
    ----------------------------------------------------------------------------
    Returns: a, ecc, i, omega, Omega
    ----------------------------------------------------------------------------
    '''
    # Norm of the position and velocity vectors
    r = norm(r_xyz)
    v = norm(v_xyz)
    
    # Radial velocity
    v_r = dot(r_xyz,v_xyz)/r
    # Angular momentum
    h_xyz = cross(r_xyz, v_xyz)
    # Norm of the angular momentum
    h = norm(h_xyz)
    
    # Eccentricity
    ecc_xyz = ((v**2 - mu/r)*r_xyz - r*v_r*v_xyz)/mu
    ecc = norm(ecc_xyz)
    
    # Semi-major axis
    a = h**2/(mu*(1-ecc**2))
    
    # Inclination of the orbit
    i = arccos(h_xyz[2]/h)
    if i==0.:
        omega=0.
        Omega=0.
    else:
        # Line of the Nodes
        N_xyz = cross(array([0., 0., 1.]), h_xyz)
        N = norm(N_xyz)
        # Longitude of the ascending node
        if N==0.:
            Omega=0.
        else:
            if N_xyz[1] >= 0:
                Omega = arccos(N_xyz[0]/N)
            else:
                Omega = 2*pi - arccos(N_xyz[0]/N)
        # Argument of the pericenter
        aux =  dot(N_xyz,ecc_xyz)/(N*ecc)
        if abs(aux) >1.:
            omega = 0.
        else:
            if ecc_xyz[2] >=0.:
                omega = arccos(aux)
            else:
                omega = 2*pi - arccos(aux)
    return a, ecc, i, omega, Omega


def twoBodyCoords(m1, m2, r_xyz, v_xyz, Rcm_xyz = array([0.,0.,0.]), Vcm_xyz = array([0.,0.,0.])):
    '''
    Returns the position and velocity vectors for a 2-body system calculated
    from the relative and center-of-mass positions and velocities.
    '''
    r1_xyz = -m2*r_xyz/(m1+m2)
    r2_xyz = m1*r_xyz/(m1+m2)
    v1_xyz = -m2*v_xyz/(m1+m2)
    v2_xyz = m1*v_xyz/(m1+m2)
    return r1_xyz, r2_xyz, v1_xyz, v2_xyz

def CM_relativePosition(m1, m2, r1_xyz, r2_xyz, v1_xyz, v2_xyz):
    '''
    Returns the position and velocity vectors for a 2-body system calculated
    from the relative and center-of-mass positions and velocities.
    '''
    Rcm_xyz = (m1*r1_xyz + m2*r2_xyz)/(m1+m2)
    r_xyz = r2_xyz - r1_xyz
    Vcm_xyz = (m1*v1_xyz + m2*v2_xyz)/(m1+m2)
    v_xyz = v2_xyz - v1_xyz
    return Rcm_xyz, r_xyz, Vcm_xyz, v_xyz


if __name__ == "__main__":
    '''
    Example to test the code: Sun-Earth system
    '''
    # Working with the system of units [au], [yr], [solar masses]
    G = 4*pi**2   # Newtonian Gravitational Constant
    m1 = 1.       # [solar masses]
    m2 = 3.00273e-6    # [solar masses]
        
    # Orbital Parameters
    a = 1.000001018      # semi-major axis [au]
    ecc = 0.0 #1671   # eccentricity
    i = 0.      # orbital plane inclination
    Omega = 0.     # Argument of the ascendent node
    omega = 0.     # Argument of the pericenter
    t_0 = 0.       # Epoch [yr]
    t = 0.       # Time to calculate the position and velocity [yr]
    
    mu = G*(m1 + m2)
    r_xyz, v_xyz = op_to_coords(mu, a, ecc, i, omega, Omega, t_0, t)
    print(f'\nr = ',r_xyz)
    print(f'\nv = ',v_xyz)
    print(f'\n')
    #new_r1, new_r2, new_v1, new_v2 = twoBodyCoords(m1, m2, r_xyz, v_xyz)
    #print(CM_relativePosition(m1, m2, new_r1, new_r2, new_v1, new_v2))
    new_a, new_ecc, new_i, new_omega, new_Omega = coords_to_op(mu, r_xyz, v_xyz)
    print(a, new_a)
    print(ecc, new_ecc)
    print(i, new_i)
    print(omega, new_omega)
    print(Omega, new_Omega)
    
    

