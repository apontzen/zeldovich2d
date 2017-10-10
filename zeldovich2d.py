"""Zeldovich2D by Andrew Pontzen. GPL licenced.

Usage:
 import zeldovich2d
 zeldovich2d.demo()

Prerequisites - py2.7/3.4+, numpy, pylab pynbody

Use pip install git+https://github.com/pynbody/pynbody.git to get pynbody
"""

import numpy as np
import pynbody
import pylab as p

def whitenoise(N):
    return np.random.normal(0,1,size=(N,N))

def sanitize(ar):
    """Remove infinities and NaNs from an array"""
    ar[ar!=ar]=0
    ar[ar==np.inf]=0

def apply_powerlaw_power_spectrum(f, n=-1.0,min_freq=2.0,max_freq=200.0):
    f_fourier = np.fft.fft2(f)
    freqs = np.fft.fftfreq(f.shape[0])
    freqs_2 = np.sqrt(freqs[:,np.newaxis]**2+freqs[np.newaxis,:]**2)
    f_fourier[freqs_2<min_freq/f.shape[0]]=0
    f_fourier[freqs_2>max_freq/f.shape[0]]=0
    freqs_2**=n
    sanitize(freqs_2)
    f_fourier*=freqs_2

    return np.fft.ifft2(f_fourier).real

def get_potential_gradients(den_real):
    """Starting from a density field in 2D, get the potential gradients i.e.
    returns the two components of grad (grad^-2 den_real)"""
    den = np.fft.fft2(den_real)

    freqs = np.fft.fftfreq(den.shape[0])

    del_sq_operator = -(freqs[:,np.newaxis]**2+freqs[np.newaxis,:]**2)

    grad_x_operator = -1.j*np.fft.fftfreq(den.shape[0])[:,np.newaxis]
    grad_y_operator = -1.j*np.fft.fftfreq(den.shape[0])[np.newaxis,:]

    phi = den/del_sq_operator
    sanitize(phi)

    grad_phi_x = grad_x_operator*phi
    grad_phi_y = grad_y_operator*phi

    grad_phi_x_real = np.fft.ifft2(grad_phi_x).real
    grad_phi_y_real = np.fft.ifft2(grad_phi_y).real

    return grad_phi_x_real, grad_phi_y_real

def get_evolved_particle_positions(den,Delta_t=0.025):
    """Generate a grid of particles, one for each cell of the density field,
    then displace those particles along gradient of potential implied by
    the density field."""
    N = len(den)
    x,y = np.mgrid[0.:N,0.:N]
    grad_x, grad_y = get_potential_gradients(den)
    x+=Delta_t*grad_x
    y+=Delta_t*grad_y
    x[x>N]-=N
    y[y>N]-=N
    x[x<0]+=N
    y[y<0]+=N
    return x.flatten(),y.flatten()

def get_final_density(input_linear_field, output_resolution=None, time=0.025):
    """Starting from a linear field, generate the equivalent non-linear field under the
    Zeldovich approximation at the specified time. If the output resolution is not
    specified, it is chosen to match in the input resolution. Output is both returned
    as a numpy array and displayed in a matplotlib window."""
    assert input_linear_field.shape[0]==input_linear_field.shape[1]
    N=len(input_linear_field)
    if not output_resolution:
        output_resolution = N
    x,y = get_evolved_particle_positions(input_linear_field,time)
    f = pynbody.new(len(x))
    f['x']=x-N/2
    f['y']=y-N/2
    f['mass']=1.0
    f['mass'].units="kg"
    f['x'].units="cm"
    return pynbody.plot.sph.image(f,width=N,resolution=output_resolution,units="kg cm^-2")

def create_linear_field(resolution=1024):
    return apply_powerlaw_power_spectrum(whitenoise(resolution))

def demo():
    linear_field = create_linear_field()
    get_final_density(linear_field)
    p.title("Zeldovich demo")
