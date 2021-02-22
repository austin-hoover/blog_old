import numpy as np
import numpy.linalg as la
import pandas as pd
from scipy.stats import truncnorm
from scipy.integrate import odeint # works better than `solve_ivp`
import matplotlib.pyplot as plt

from scipy.constants import epsilon_0, elementary_charge, speed_of_light, pi
proton_mass = 0.938272029 # [GeV/c^2]
classical_proton_radius = 1.53469e-18 # [m]
k0 = 0.35 # [m^-1]


class FODO:
    """Class for FODO lattice.
    
    The order is : half-qf, drift, qd, drift, half-qf. Both magnets are
    upright and have the same strength.
    
    Attributes
    ----------
    k0 : float
        Focusing strength of both quadrupoles [m^-1].
    length : float
        Period length [m].
    fill_fac : float
        Fraction of cell filled with quadrupoles.
    """
    def __init__(self, k0=k0, length=5.0, fill_fac = 0.5):
        self.k0, self.length, self.fill_fac = k0, length, fill_fac
        
    def foc_strength(self, s):
        """Return x and y focusing strength at position `s`. We assume the 
        lattice repeats forever."""
        kx, ky = 0., 0
        s %= self.length # assume infinite repeating cells
        s /= self.length # fractional position in cell
        delta = 0.25 * self.fill_fac
        if s < delta or s > 1 - delta:
            kx, ky = self.k0, -self.k0
        elif 0.5 - delta <= s < 0.5 + delta:
            kx, ky = -self.k0, +self.k0
        return kx, ky
        
        
class EnvelopeSolver:
    """Class to track the rms beam envelope assuming a uniform density ellipse.
    
    Attributes
    ----------
    positions : ndarray, shape (nsteps + 1,)
        Positions at which to evaluate.
    Sigma0 : ndarray, shape (4, 4):
        Initial covariance matrix.
    sigma0 : ndarray, shape (10,)
        Initial moment vector (upper-triangular elements of `Sigma`. Order is :
        ['x2','xxp','xy','xyp','xp2','yxp','xpyp','y2','yyp','yp2'].
    sigma : ndarray, shape (nsteps + 1, 10)
        Beam moment vector at each position.
    perveance : float
        Dimensionless beam perveance.
    ext_foc : callable
        Function which returns the external focusing strength at position s.
        Call signature is `kx, ky = ext_foc(s)`.
    mm_mrad : bool
        Whether to convert units to mm-mrad.
    """
    def __init__(self, Sigma0, positions, perveance, ext_foc=None, 
                 mm_mrad=True, atol=1e-14):
        self.sigma0 = Sigma0[np.triu_indices(4)]
        self.positions, self.perveance = positions, perveance
        self.mm_mrad = mm_mrad
        self.atol = atol
        self.ext_foc = ext_foc
        if self.ext_foc is None:
            self.ext_foc = lambda s: (0.0, 0.0)
        
    def reset(self):
        self.moments = []
        
    def set_perveance(self, perveance):
        self.perveance = perveance
        
    def derivs(self, sigma, s): 
        """Return derivative of 10 element moment vector."""
        k0x, k0y = self.ext_foc(s)
        k0xy = 0.
        # Space charge terms
        s11, s12, s13, s14, s22, s23, s24, s33, s34, s44 = sigma
        S0 = np.sqrt(s11*s33 - s13**2)
        Sx, Sy = s11 + S0, s33 + S0
        D = S0 * (Sx + Sy)
        psi_xx, psi_yy, psi_xy = Sy/D, Sx/D, -s13/D
        # Modified focusing strength
        kx = k0x - 0.5 * self.perveance * psi_xx
        ky = k0y - 0.5 * self.perveance * psi_yy
        kxy = k0xy + 0.5 * self.perveance * psi_xy
        # Derivatives
        sigma_p = np.zeros(10)
        sigma_p[0] = 2 * s12
        sigma_p[1] = s22 - kx*s11 + kxy*s13
        sigma_p[2] = s23 + s14
        sigma_p[3] = s24 + kxy*s11 - ky*s13
        sigma_p[4] = -2*kx*s12 + 2*kxy*s23
        sigma_p[5] = s24 - kx*s13 + kxy*s33
        sigma_p[6] = -kx*s14 + kxy*(s34+s12) - ky*s23
        sigma_p[7] = 2 * s34
        sigma_p[8] = s44 + kxy*s13 - ky*s33
        sigma_p[9] = 2*kxy*s14 - 2*ky*s34
        return sigma_p
                
    def integrate(self):
        """Integrate the envelope equations."""
        self.moments = odeint(self.derivs, self.sigma0, self.positions, atol=self.atol)
        if self.mm_mrad:
            self.moments *= 1e6
            
            
def rotation_matrix(phi):
    """2D rotation matrix (cw)."""
    C, S = np.cos(phi), np.sin(phi)
    return np.array([[C, S], [-S, C]])


def apply(M, X):
    """Apply matrix M to all rows of X."""
    return np.apply_along_axis(lambda x: np.matmul(M, x), 1, X)


def norm_rows(X):
    """Normalize all rows of X to unit length."""
    return np.apply_along_axis(lambda x: x/la.norm(x), 1, X)
    
    
def rand_rows(X, k):
    """Return k random rows of X."""
    idx = np.random.choice(X.shape[0], k, replace=False)
    return X[idx, :]


def Vmat2D(alpha, beta):
    """Return symplectic normalization matrix for 2D phase space."""
    return np.sqrt(1/beta)* np.array([[beta, 0], [alpha, 1]])
    
    
def get_perveance(line_density, beta, gamma):
    """Return the dimensionless space charge perveance."""
    return 2 * classical_proton_radius * line_density / (beta**2 * gamma**3)
    
    
def get_sc_factor(charge, mass, beta, gamma):
    """Return factor defined by x'' = factor * (x electric field component).
    
    Units of charge are Coulombs, and units of mass are GeV/c^2. 
    """
    mass_kg = mass * 1.782662e-27
    velocity = beta * speed_of_light
    return charge / (mass_kg * velocity**2 * gamma**3)


class DistGenerator:
    """Class to generate particle distributions in 4D phase space.
    
    The four dimensions are the transverse positions and slopes {x, x', y, y'}.
    Note that 'normalized space', is referring to normalization in the 2D 
    sense, in which the x-x' and y-y' ellipses are upright, as opposed the 4D
    sense, in which the whole covariance matrix is diagonal. In other words,
    only the regular Twiss parameters are used.
    
    Attributes
    ----------
    ex, ey : float
        Rms emittances: eu = sqrt(<u^2><u'^2> - <uu'>^2)
    ax, ay, bx, by : float
        Alpha and beta functions: au = <uu'> / eu, bu = <u^2> / eu
    V : ndarray, shape (4, 4)
        Symplectic normalization matrix.
    A : ndarray, shape (4, 4)
        Emittance scaling matrix.
    kinds : list
        List of the available distributions.
    """
    
    def __init__(self, twiss=(0., 0., 10., 10.), eps=(100e-6, 100e-6)):
        self.set_eps(*eps)
        self.set_twiss(*twiss)
        self.kinds = ['kv', 'gauss', 'waterbag', 'danilov']
        self._gen_funcs = {'kv':self._kv, 
                           'gauss':self._gauss, 
                           'waterbag':self._waterbag,
                           'danilov':self._danilov}
        
    def set_twiss(self, ax, ay, bx, by):
        """Set Twiss parameters"""
        self.ax, self.ay, self.bx, self.by = ax, ay, bx, by
        self.V = np.zeros((4, 4))
        self.V[:2, :2] = Vmat2D(ax, bx)
        self.V[2:, 2:] = Vmat2D(ay, by)
        
    def set_eps(self, ex, ey):
        """Set emittance."""
        self.ex, self.ey = ex, ey
        self.A = np.sqrt(np.diag([ex, ex, ey, ey]))
        
    def get_cov(self, ex, ey):
        Sigma = np.zeros((4, 4))
        gx = (1 + self.ax**2) / self.bx
        gy = (1 + self.ay**2) / self.by
        Sigma[:2, :2] = ex * np.array([[self.bx, -self.ax], [-self.ax, gx]])
        Sigma[2:, 2:] = ey * np.array([[self.by, -self.ay], [-self.ay, gy]])
        return Sigma
    
    def unnormalize(self, X):
        """Transform coordinates out of normalized phase space.
        
        X : ndarray, shape (nparts, 4)
        """
        return apply(np.matmul(self.V, self.A), X)
    
    def normalize(self, X):
        """Transform coordinates into normalized phase space.
        
        X : ndarray, shape (nparts, 4)
        """
        return apply(la.inv(np.matmul(self.V, self.A)), X)
    
    def generate(self, kind='gauss', nparts=1, eps=None, **kwargs):
        """Generate a distribution.
        
        Parameters
        ----------
        kind : {'kv', 'gauss', 'danilov'}
            The kind of distribution to generate.
        **kwargs
            Key word arguments passed to the generating function.
        
        Returns
        -------
        X : ndarray, shape (nparts, 4)
            The corodinate array
        """
        if type(eps) is float:
            eps = [eps, eps]
        if kind == 'kv':
            eps = [4 * e for e in eps]
        if eps is not None:
            self.set_eps(*eps)
        Xn = self._gen_funcs[kind](int(nparts), **kwargs)
        return self.unnormalize(Xn)
    
    def _kv(self, nparts, **kwargs):
        """Generate a KV distribution in normalized space.
        
        Particles uniformly populate the boundary of a 4D sphere. This is 
        achieved by normalizing the radii of all particles in 4D Gaussian
        distribution to unit length.
        """ 
        Xn = np.random.normal(size=(nparts, 4))
        return norm_rows(Xn)
        
    def _gauss(self, nparts, cut=None, **kwargs):
        """Gaussian distribution in normalized space.
        
        cut: float or None
            Cut off the distribution after this many standard devations."""
        if cut:
            Xn = truncnorm.rvs(a=4*[-cut], b=4*[cut], size=(nparts, 4))
        else:
            Xn = np.random.normal(size=(nparts, 4))
        return Xn
    
    def _danilov(self, nparts, phase_diff=90., **kwargs):
        """Danilov distribution in normalized space. 
        
        This is defined by the conditions {y' = ax + by, x' = cx + dy} for all
        particles. Note that it is best to use the 4D Twiss parameters instead.
        
        phase_diff : float
            Difference between x and y phases.
        """
        r = np.sqrt(np.random.random(nparts))
        theta = 2 * np.pi * np.random.random(nparts)
        x, y = r * np.cos(theta), r * np.sin(theta)
        xp, yp = -y, x
        Xn = np.vstack([x, xp, y, yp]).T
        P = np.identity(4)
        P[:2, :2] = rotation_matrix(np.radians(phase_diff - 90))
        return apply(P, Xn)
    
    def _waterbag(self, **kwargs):
        """Waterbag distribution in normalized space.
        
        Particles uniformly populate the interior of a 4D sphere. First, 
        a KV distribution is generated. Then, particle radii are scaled by
        the factor r^(1/4), where r is a uniform random variable in the range
        [0, 1].
        """
        Xn = norm_rows(np.random.normal(size=(nparts, 4)))
        r = np.random.random(nparts)**(1/4)
        r = r.reshape(nparts, 1)
        return r * Xn
    
    
def vec_to_mat(v):
    """Return covariance matrix from moment vector."""
    S11, S12, S13, S14, S22, S23, S24, S33, S34, S44 = v
    return np.array([[S11, S12, S13, S14],
                     [S12, S22, S23, S24],
                     [S13, S23, S33, S34],
                     [S14, S24, S34, S44]])