# Supernova Code

from numpy import array, ones, empty, random, sqrt, pi, zeros
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from conversion import *


##### Simulation Parameters ########################################################

# Gravitational constant in units of au^3 M_sun^-1 yr-2
G = 4.*pi**2

# Discrete time step.
dt = 1.e-4 # yr

####################################################################################

class Body:
    '''
    Each body object will represent a star or particle resulting from the
    supernova explosion.
    '''
    def __init__(self, m, position, momentum):
        '''
        Creates a particle using the arguments
        .mass : scalar
        .position : NumPy array  with the coordinates [x,y,z]
        .momentum : NumPy array  with the components [px,py,pz]
        '''
        self.m = m
        self.m_pos = m * position
        self.momentum = momentum
        self.marker = 'o'
    
    def position(self):
        '''
        Returns the physical coordinates of the node.
        '''
        return self.m_pos / self.m


def distance_between(body1, body2):
    '''
    Returns the distance between node1 and node2.
    '''
    return norm(body1.position() - body2.position())


def gravitational_force(body1, body2):
    '''
    Returns the gravitational force on body1 due to body 2.
    A short distance cutoff is introduced in order to avoid numerical
    divergences in the gravitational force.
    '''
    cutoff_dist = 2.e-4
    d = distance_between(body1, body2)
    if d < cutoff_dist:
        #print('Collision!')
        # Returns no Force!
        return array([0., 0., 0.])
    else:
        # Gravitational force
        return -G*body1.m*body2.m*(body1.position() - body2.position())/d**3


def force_on(p, stars):
    '''
    This function computes the net force on a body exerted by all other bodies.
    '''
    force = zeros(3)
    for s in stars:
        force += gravitational_force(p, s)
    return force #sum(gravitational_force(p, s) for s in stars)


def verlet(particles, stars, dt):
    '''
    Verlet method for time evolution.
    Force on eahc of the particles due to the stars.
    '''
    for p in particles:
        force = force_on(p, stars)
        p.momentum += force*dt
        p.m_pos += p.momentum*dt
   
    
def system_init(star1M, star1_ini_pos, star1_ini_momentum, star2M, star2_ini_pos, star2_ini_momentum):
    '''
    This function initialize the 2-body system by creating the corresponding
    objects of the Body class
    '''
    # The stars list contains only the two stars in the binary system
    stars = []
    stars.append(Body(star1M, star1_ini_pos, star1_ini_momentum))
    stars.append(Body(star2M, star2_ini_pos, star2_ini_momentum))
    return stars

    
def evolution_explosion(stars, n, center, plot_limit, img_step,
                        image_folder='images/', video_name='my_video.mp4'):
    '''
    This function evolves the system modelling the supernova explosion
    at a random time time_of_explosion
    '''
    # Initially there are no particles (besides the two stars)
    particles = []
    # Random time of the explosion
    random.seed(413)
    time_of_explosion = int((random.random()*0.3*n)//img_step)*img_step
    # Range of time before the explosion
    range_before = range(time_of_explosion)
    # Range of time after the explosion
    range_after = range(time_of_explosion, n+1)
    
    # Evolution before the explosion
    evolve(stars, particles, range_before, center, plot_limit, img_step,
           image_folder='images/', video_name='my_video.mp4')
    
    # Explosion
    # Percentage of mass expelled during the explosion
    q = 0.1
    particles = explosion(stars[1], q)
    
    # Evolution after the explosion
    evolve(stars, particles, range_after, center, plot_limit, img_step,
           image_folder='images/', video_name='my_video.mp4')

def evolve(stars, particles, time_range, center, plot_limit, img_step, image_folder='images/', video_name='my_video.mp4'):
    '''
    This function evolves the system in time using the Verlet algorithm
    (No explosion involved here!)
    '''
    # Limits for the axes in the plot
    axis_limit = 1.5*plot_limit
    lim_inf = [center[0]-axis_limit, center[1]-axis_limit, center[2]-axis_limit]
    lim_sup = [center[0]+axis_limit, center[1]+axis_limit, center[2]+axis_limit]
    
    # Main loop over time iterations.
    for i in time_range:
        verlet(stars, stars, dt) # Update for the stars in the binary
        verlet(particles, stars, dt) # Update for the rest of particles
        
        if i%img_step==0:
            print("Writing image at time {0}".format(i))
            plot_bodies(stars+particles, i//img_step, 2*i*dt, lim_inf, lim_sup, image_folder)

def explosion(destroyed_star, q):
    '''
    This function models the explosion by updating the data of the
    destroyed star and creating the expelled particles
    '''
    particles = []
    #particles.append(Body(1.5,exploding_star.position()+[0.1,0.,0.],1.5*(exploding_star.momentum/exploding_star.m)))
    #particles.append(Body(1.5, array([0.,0.,0.]),array([0.,0.,0.])))
    # Update of the parameters of the star after the explosion
    destroyed_star.momentum *=(1.-q)/destroyed_star.m # New momentum
    destroyed_star.m *= (1-q) # New mass
    destroyed_star.marker = '.' # New marker indicating the remnant
    return particles
    

def plot_bodies(bodies, i, time, lim_inf, lim_sup, image_folder='images/'):
    '''
    Writes an image file with the current position of the bodies
    '''
    plt.rcParams['grid.color'] = 'dimgray'
    #plt.rcParams['axes.edgecolor'] = 'dimgray'
    #plt.rcParams['axes.labelcolor'] = 'dimgray'
    fig = plt.figure(figsize=(10,10), facecolor='black')
    ax = plt.gcf().add_subplot(111, projection='3d')
    ax.set_xlim([lim_inf[0], lim_sup[0]])
    ax.set_ylim([lim_inf[1], lim_sup[1]])
    ax.set_zlim([lim_inf[2], lim_sup[2]])
    #ax.set_proj_type('ortho')
    ax.set_facecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('dimgray')
    ax.yaxis.pane.set_edgecolor('dimgray')
    ax.zaxis.pane.set_edgecolor('dimgray')
    #ax.xaxis.label.set_color('dimgray')
    #ax.yaxis.label.set_color('dimgray')
    #ax.zaxis.label.set_color('dimgray')
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    #ax.set_zlabel('z')
    #ax.grid(False)
    for body in bodies:
        pos = body.position()
        ax.scatter(pos[0], pos[1], pos[2], marker=body.marker, color='lightcyan')
    ax.set_title('Time: {:.3f} years'.format(time), color='dimgray')
    plt.gcf().savefig(image_folder+'bodies3D_{0:06}.png'.format(i))
    plt.close()


def create_video(image_folder='images/', video_name='my_video.mp4'):
    '''
    Creates a .mp4 video using the stored files images
    '''
    from os import listdir
    import moviepy.video.io.ImageSequenceClip
    fps = 15
    image_files = [image_folder+img for img in sorted(listdir(image_folder)) if img.endswith(".png")]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(video_name)








if __name__=="__main__":
    '''
    Example of a binary system
    '''
    
    # Stars initial data
    star1M = 20. # Solar masses
    star2M = 15. # Solar masses
    # Orbital Parameters
    a = 3.         # semi-major axis [au]
    ecc = 0.0   # eccentricity
    i = 0.      # orbital plane inclination
    Omega = 0.     # Argument of the ascendent node
    omega = 0.     # Argument of the pericenter
    t_0 = 0.       # Epoch [yr]
    t = 0.       # Time to calculate the position and velocity [yr]
    
    mu = G*(star1M + star2M)
    r_xyz, v_xyz = op_to_coords(mu, a, ecc, i, omega, Omega, t_0, t)
    star1_ini_pos, star2_ini_pos, star1_ini_v, star2_ini_v = twoBodyCoords(star1M, star2M, r_xyz, v_xyz)
    star1_ini_momentum, star2_ini_momentum = star1M*star1_ini_v, star2M*star2_ini_v
    
    # center of the plot
    center = array([0., 0., 0.])
    # Limit for the plot
    plot_limit = a # [au]
    
    # Number of time-iterations executed by the program.
    n = 20000 # Time steps
    # Frequency at which .PNG images are written.
    img_step = 100
    # Folder to save the images
    image_folder = 'images/'
    # Name of the generated video
    video_name = 'my_video.mp4'
    
    stars = system_init(star1M, star1_ini_pos, star1_ini_momentum,
                                    star2M, star2_ini_pos, star2_ini_momentum)
    evolution_explosion(stars, n, center, plot_limit, img_step, image_folder, video_name)
    #i_part = [Body(1., array([0., 1.5, 0.]), array([0.,0.,1.]))]
    #i_part = [Body(star2M, star2_ini_pos, star2_ini_momentum)]
    #i_part.append(Body(1., array([0., 1.5, 0.]), array([0.,0.,6.])))
    #evolve(stars, i_part, range(n+1), center, plot_limit, img_step, image_folder, video_name)
    create_video(image_folder, video_name)
