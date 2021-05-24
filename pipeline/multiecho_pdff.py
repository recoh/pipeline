import copy
import json
import os
from typing import Dict

import SimpleITK as sitk
import numpy as np
import numpy.linalg as la
from dcmstack import NiftiWrapper
from numpy.fft import fftn, ifftn
from scipy.linalg import svdvals
from scipy.optimize import fmin
from scipy.signal import convolve
from unwrap import unwrap

from pipeline.constants import DirectoryStructure as ds, FileNames as fn
from pipeline.logger import log


def mdot(a: np.ndarray, b: np.ndarray, dim: int = 0) -> np.ndarray:
    """
    Python equivalent to the MATLAB dot function...

    dot  Vector dot product.
        C = dot(A,B) returns the scalar product of the vectors A and B.
        A and B must be vectors of the same length.  When A and B are both
        column vectors, dot(A,B) is the same as A'*B.

        dot(A,B), for N-D arrays A and B, returns the scalar product
        along the first non-singleton dimension of A and B. A and B must
        have the same size.

        dot(A,B,DIM) returns the scalar product of A and B in the
        dimension DIM.

        Class support for inputs A,B:
           float: double, single

        See also cross.
    """
    return np.sum(a.conj() * b, axis=dim)


def sorted_array(a: np.array) -> bool:
    n = a.shape
    if n[0] <= 1:
        return True

    return a[0] <= a[1] and sorted_array(a[1:])


def exp_fit_func(psif: np.ndarray, te: np.ndarray, data: np.ndarray) -> float:
    # exponential fitting function
    psif_tmp = np.complex(psif[0], psif[1])
    f = np.exp(1j * psif_tmp * te)  # function
    v = np.dot(f.conj().T @ data, la.pinv(f.conj().T @ f))  # varpro
    r = f @ v - data  # residual

    return la.norm(r)


def fat_basis(te: np.array, tesla: float, ndb: float = 2.5, h2o: float = 4.7,
              units: str = 'proton') -> (np.ndarray, np.float, np.array, np.array):
    """
    [A psif ampl freq] = fat_basis(te,Tesla,NDB,H2O,units)

     Function that produces the fat-water matrix. Units can be adjusted
     to reflect the relative masses of water and fat. Default is proton.
     For backward compatibility, set NDB=-1 to use old Hamilton values.

     Inputs:
      te (echo times in sec)
      Tesla (field strength)
      NDB (number of double bonds)
      H2O (water freq in ppm)
      units ('proton' or 'mass')

     Ouputs:
      A = matrix [water fat] of basis vectors
      psif = best fit of fat to exp(i*B0*te-R2*te)
             where B0=real(psif) and R2=imag(psif)
             note: B0 unit is rad/s (B0/2pi is Hz)
     ampl = vector of relative amplitudes of fat peaks
     freq = vector of frequencies of fat peaks (unit Hz)

     Ref: Bydder M, Girard O, Hamilton G. Magn Reson Imag. 2011(29):1041
    """

    """ argument checks """

    if np.max(te) < 1e-3 or np.max(te) > 1:
        log.error("'te' should be in seconds")

    """ triglyderide properties """

    if ndb == -1:
        # backward compatibility
        d = [1.300, 2.100, 0.900, 5.300, 4.200, 2.750]
        ampl = [0.700, 0.120, 0.088, 0.047, 0.039, 0.006]
        awater = 1
    else:
        # fat chemical shifts in ppm
        d = [5.29, 5.19, 4.20, 2.75, 2.24, 2.02, 1.60, 1.30, 0.90]

        # Mark's heuristic formulas
        cl = 16.8 + 0.25 * ndb
        nddb = 0.093 * ndb ** 2

        # Gavin's formulas (no. protons per molecule)
        awater = 2
        ampl = np.zeros(9)
        ampl[0] = ndb * 2
        ampl[1] = 1
        ampl[2] = 4
        ampl[3] = nddb * 2
        ampl[4] = 6
        ampl[5] = (ndb - nddb) * 4
        ampl[6] = 6
        ampl[7] = (cl - 4) * 6 - ndb * 8 + nddb * 2
        ampl[8] = 9

        # the above counts no. molecules so that 1 unit of water = 2
        # protons and 1 unit of fat = sum(a) = (2+6*CL-2*NDB) protons.
        if units == 'mass':
            # scale in terms of molecular mass so that 18 units of
            # water = 2 protons and (134+42*CL-2*NDB) units of fat =
            # (2+6*CL-2*NDB) protons. I.e. w and f are in mass units.
            awater = awater / 18
            ampl = ampl / (134 + 42 * cl - 2 * ndb)
        elif units == 'proton':
            # scale by no. protons (important for PDFF) so 1 unit
            # of water = 1 proton and 1 unit of fat = 1 proton.
            awater = awater / 2
            ampl = ampl / np.sum(ampl)
        else:
            log.error("Only units == 'mass' or units == 'proton' are valid options")
            raise RuntimeError

    """ fat water matrix """

    # put variables in the right format
    ne = len(te)
    te = np.reshape(te, (ne, 1))

    # time evolution matrix
    freq = np.zeros(len(d))
    fat = te * 0
    water = te * 0 + awater
    larmor = 42.57747892 * tesla  # larmor freq (MHz)
    for i in range(len(d)):
        freq[i] = larmor * (d[i] - h2o)  # relative to water
        fat = fat + ampl[i] * np.exp(2 * np.pi * 1j * freq[i] * te)

    a = np.concatenate((water, fat), axis=1)

    # nonlinear fit of fat to complex np.exp (gauss newton)
    psif = np.array([2.0 * np.pi * larmor * (1.3 - h2o), 50.0])  # initial estimates (rad/s)
    tmp_psif = fmin(exp_fit_func, psif, args=(te, fat), disp=False)

    return a, np.complex(tmp_psif[0], tmp_psif[1]), ampl, freq


def medfiltn(a: np.ndarray, p: np.array, mask: np.ndarray, debug: bool = False) -> np.ndarray:
    """
    Median filter of N-D array using kernel P.
    Performs circular wrapping at boundaries.

    A is an N-D array
    P is a kernel (default [3 3])
    mask (binary) excludes points
    """

    # argument checks
    if np.size(p) > np.ndim(a):
        log.error('P has more dims than A')
        raise RuntimeError
    elif np.size(p) == 2:
        p = p.reshape(np.size(p), 1, order='F')
    else:
        log.error('ndim(p) = {} does not make sense'.format(np.ndim(p)))
        raise RuntimeError

    if np.ndim(a) != np.ndim(mask) and np.sum(mask == 1):
        log.error('something wrong with mask')
        raise RuntimeError

    orig_a = a.copy()
    a[~mask] = np.NaN  # let's use use omitnan flag

    # generate shift indicies
    n_p = int(np.prod(p))
    t_p = np.size(p)
    s = np.zeros((t_p, n_p), dtype='int')

    for k in range(n_p):
        if t_p == 1:
            s[0, k] = np.unravel_index(k, shape=p[0, 0])
        elif t_p == 2:
            s[0, k], s[1, k] = np.unravel_index(k, shape=(p[0, 0], p[1, 0]))
        elif t_p == 3:
            s[0, k], s[1, k], s[2, k] = np.unravel_index(k, shape=(p[0, 0], p[1, 0], p[2, 0]))
        elif t_p == 4:
            s[0, k], s[1, k], s[2, 0], s[3, k] = np.unravel_index(k, shape=(p[0, 0], p[1, 0], p[2, 0], p[3, 0]))
        else:
            log.error('high dim not implemented - fix me')
            raise RuntimeError
    s += 1
    s = np.flipud(s)
    if debug:
        log.debug('S = {}'.format(s))

    # make data matrix
    b = np.zeros((np.size(a), n_p), dtype=a.dtype)

    for k in range(n_p):
        tmp = np.roll(a, s[:, k] - p.flatten() // 2 - 1, axis=(0, 1))
        b[:, k] = np.reshape(tmp, np.size(a), order='F')

    # median along shift dim - for complex maybe use medoid?
    b = np.nanmedian(b, axis=1, overwrite_input=True, keepdims=True)

    # use orig values when all masked
    k = np.isnan(b)
    b[k] = np.reshape(orig_a, (np.size(a), 1), order='F')[k]

    # return it as we found it
    return np.reshape(b, orig_a.shape, order='F')


def set_noise(data: np.ndarray, debug: bool = False) -> float:
    tmp = data.copy()
    nd = np.prod(tmp.shape)
    x = np.ndarray((nd, 25), dtype='complex')
    count = 0
    for j in range(-2, 3):
        for k in range(-2, 3):
            x[:, count] = np.reshape(np.roll(tmp, (j, k), axis=(0, 1)), nd, order='F')
            count += 1
    x = np.min(svdvals(x.conj().T @ x))  # smallest sval represents noise only
    noise = np.sqrt(x / np.sum(tmp != 0.0))
    if debug:
        log.debug('noise std = {:.4f}'.format(noise))

    return noise


def transform(psi: np.ndarray, options: Dict, debug: bool = False) -> (np.ndarray, float):
    # change of variable for nonnegative R2*
    r2 = psi.imag

    if options['nonnegR2']:
        if debug:
            log.debug("options['nonnegR2'] is True")
        dr2 = np.sign(r2) + (r2 == 0.0)  # fix deriv=0 at R2=0
        r2 = np.abs(r2)
    else:
        dr2 = 1.0

    # transformed psi
    return psi.real + 1j * r2, dr2


def unswap(psi: np.ndarray, te: np.array, options: Dict) -> np.ndarray:
    """"
    use phase unwrapping to remove 2pi boundaries

    this unwraps in 3 passes: first for aliasing and second for fat-water swaps. then again
    to fix unwraps made on the wrong pass. it would be better if we could do a single pass
    and choose the best option.
    """

    # swapping can occur at 2 frequences
    swap0 = 1 / np.amin(np.diff(te, axis=0))  # aliasing (Hz)
    swap1 = -np.real(options['psif']) / 2 / np.pi  # fat-water (Hz)
    swap2 = np.abs(np.diff([swap0, swap1]))  # both
    swap = np.array([swap0, swap1, float(swap2)])

    # frequencies too close, skip the difference
    if np.amin(swap[2] / swap[0: 1], axis=0) < 0.2:
        swap[2] = []

    # only unwrap B0
    b0 = psi.squeeze().real

    # unwrap is CPU only (mex)
    if np.ndim(options['mask']) == 4:
        ns, nx, ny, nz = options['mask'].shape
    elif np.ndim(options['mask']) == 3:
        nx, ny, nz = options['mask'].shape
    elif np.ndim(options['mask']) == 2:
        nx, ny = options['mask'].shape
    else:
        log.error('Mask dimensions != 2, 3, or 4 :(')
        raise RuntimeError
    mask = options['mask'].reshape(nx, ny)

    # unwrap all swaps
    for k in range(np.size(swap)):
        b0 = b0 / swap[k]
        if np.ndim(b0) in [2, 3]:
            b0 = unwrap(b0, False, False, False)
        b0 = b0 * swap[k]

    # remove gross aliasing (cosmetic)
    alias_freq = 2 * np.pi / np.min(np.diff(te, axis=0))  # aliasing freq (rad/s)
    center_freq = np.median(b0[mask > 0.0])  # center B0 (rad/s)
    nwraps = int(center_freq / alias_freq)  # no. of wraps
    b0[mask] = b0[mask] - (nwraps * alias_freq)

    # back to complex
    return b0[np.newaxis, ...] + 1j * psi.imag


def cost_func(arg: np.ndarray, te: np.ndarray, data: np.ndarray, psi_real: np.ndarray, options: Dict) -> np.ndarray:
    # cost function = sum of squares residual + penalty terms
    tmp_psi, _ = transform(arg, options)
    r, _, _, _, _ = pclsr(arg, te, data, options)

    return np.sum(np.abs(r.squeeze()) ** 2, axis=0) + \
           (options['muB'] ** 2) * (np.real(tmp_psi - psi_real) ** 2) + \
           (options['muR'] ** 2) * (np.imag(tmp_psi - psi_real) ** 2)


def pclsr(psi: np.ndarray, te: np.ndarray, b: np.ndarray, options: Dict,
          debug: bool = False) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """ phase constrained least squares residual r=W*A*x*exp(i*phi)-b """

    # note echos in dim 1
    if debug:
        log.debug('note echos in dim 1')
    if np.ndim(b) == 4:
        ne, nx, ny, nz = b.shape
        b = np.reshape(b, (ne, nx * ny * nz), order='F')
        psi = np.reshape(psi, (1, nx * ny * nz), order='F')
    else:
        log.error('Dimension of the data ({}) is not supported'.format(np.ndim(b)))
        return

    # change of variable
    if debug:
        log.debug('change of variable')
    tpsi, dr2 = transform(psi, options)

    # complex fieldmap
    if debug:
        log.debug('complex fieldmap')
    w = np.exp(1j * te * tpsi)

    # M = Re(A'*W'*W*A) is a 2x2 matrix [M1 M2;M2 M3] with inverse [M3 -M2;-M2 M1]/dtm (determinant)
    # log.debug("M = Re(A'*W'*W*A) is a 2x2 matrix [M1 M2;M2 M3] with inverse [M3 -M2;-M2 M1]/dtm (determinant)")
    ww = w.real ** 2 + w.imag ** 2
    m1 = np.real(options['A'][:, 0].conj() * options['A'][:, 0]).T @ ww
    m2 = np.real(options['A'][:, 0].conj() * options['A'][:, 1]).T @ ww
    m3 = np.real(options['A'][:, 1].conj() * options['A'][:, 1]).T @ ww
    dtm = m1 * m3 - m2 ** 2

    # z = inv(M)*A'*W'*b
    wb = w.conj() * b
    z = (options['A'].conj().T @ wb) / dtm
    z = np.array([m3 * z[0, :] - m2 * z[1, :], m1 * z[1, :] - m2 * z[0, :]])

    # p = z.'*M*z
    p = z[0, :] * m1 * z[0, :] + z[1, :] * m3 * z[1, :] + z[0, :] * m2 * z[1, :] + z[1, :] * m2 * z[0, :]

    # smooth initial phase (phi)
    if options['smooth_phase']:
        if debug:
            log.debug('smooth initial phase (phi)')
        p = np.reshape(p, (nx, ny, nz), order='F').squeeze()
        p = convolve(p, options['filter'], mode='same')
        p = np.reshape(p, (1, nx * ny * nz), order='F')

    phi = np.angle(p) / 2  # -pi/2<phi<pi/2
    x = np.real(z * np.exp(-1j * phi))

    # absorb sign of x into phi
    # log.debug('absorb sign of x into phi')
    x = x * np.exp(1j * phi)
    phi = np.angle(np.sum(x, axis=0))  # -pi<phi<pi
    x = np.real(z * np.exp(-1j * phi))

    # box constraint (0<=FF<=1)
    if options['nonnegFF']:
        log.warning('box constraint (0<=FF<=1)')
        x = (x < 0.0).choose(x, 0.0)
        wax = w * (options['A'] @ x)
        dotwaxwax = np.real(np.sum(wax.conj() * wax, axis=0))
        a = np.sum(wax.conj() * b, axis=0) / np.maximum(dotwaxwax, np.spacing(options['noise']))
        x = np.abs(a) * x
        phi = np.angle(a)
        if options['smooth_phase']:
            log.warning('smooth phase (phi) within box constraint')
            p = np.abs(p) * np.exp(1j * phi)
            p = np.reshape(p, (nx, ny, nz), order='F').squeeze()
            p = convolve(p, options['filter'], mode='same')
            p = np.reshape(p, (1, nx * ny * nz), order='F')
            phi = np.angle(p)

    # residual
    eiphi = np.exp(1j * phi)
    wax = w * (options['A'] @ x)
    r = wax * eiphi
    r = np.reshape(r - b, (ne, nx, ny, nz), order='F')

    """ derivatives w.r.t. real(psi) and imag(psi) """

    # y = inv(M)*A'*W'*T*b
    y = ((options['A'] * te).conj().T @ wb) / dtm
    y = np.array([m3 * y[0, :] - m2 * y[1, :], m1 * y[1, :] - m2 * y[0, :]])

    # q = y.'*M*z
    q = y[0, :] * m1 * z[0, :] + y[1, :] * m3 * z[1, :] + y[0, :] * m2 * z[1, :] + y[1, :] * m2 * z[0, :]

    # H is like M but with T in the middle
    ww = te * ww
    h1 = np.real(np.conj(options['A'][:, 0]) * options['A'][:, 0]).T @ ww
    h2 = np.real(np.conj(options['A'][:, 0]) * options['A'][:, 1]).T @ ww
    h3 = np.real(np.conj(options['A'][:, 1]) * options['A'][:, 1]).T @ ww

    # s = z.'*H*z
    s = z[0, :] * h1 * z[0, :] + z[1, :] * h3 * z[1, :] + z[0, :] * h2 * z[1, :] + z[1, :] * h2 * z[0, :]

    """ real part (B0): JB """

    # first term
    jb = 1j * te * wax

    # second term
    dphi = -np.real(q / p)
    dphi = (p == 0.0).choose(dphi, 0.0)
    jb = jb + wax * 1j * dphi

    # third term
    dx = y + z * dphi
    # eiphi = (eiphi == 0.0).choose(eiphi, np.finfo(dtype='float').resolution)
    dx = np.imag(dx / eiphi)
    dx = np.isnan(dx).choose(dx, 0.0)
    jb = jb + w * options['A'].dot(dx)

    # impart phase
    jb = jb * eiphi

    """ imag part (R2*): JR """

    # first term
    jr = -te * wax

    # second term
    dphi = -np.imag(q / p) + np.imag(s / p)
    dphi = (p == 0.0).choose(dphi, 0.0)
    jr = jr + wax * 1j * dphi

    # third term
    dx = y + z * 1j * dphi
    dx = np.real(dx * -1.0 / eiphi)
    # dx = np.isnan(dx).choose(dx, 0.0)

    hx = np.array([h1 * x[0, :] + h2 * x[1, :], h3 * x[1, :] + h2 * x[0, :]])
    hx = hx * 2.0 / dtm
    dx = np.array(dx + [m3 * hx[0, :] - m2 * hx[1, :], m1 * hx[1, :] - m2 * hx[0, :]])

    jr = jr + w * options['A'].dot(dx)

    # impart phase
    jr = jr * eiphi

    # change variable
    jr = jr * dr2

    """ return arguments """

    x = np.reshape(x, (2, nx, ny, nz), order='F')
    jb = np.reshape(jb, (ne, nx, ny, nz), order='F')
    jr = np.reshape(jr, (ne, nx, ny, nz), order='F')
    phi = np.reshape(phi, (1, nx, ny, nz), order='F')

    return r, phi, x, jb, jr


def nls_fit_presco(psi: np.ndarray, te: np.ndarray, data: np.ndarray,
                   options: Dict) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """ nonlinear least squares fitting """

    # regularizer (smooth B0 + zero R2)
    psi_real = psi.real

    for index in range(options['maxit'][1]):
        # log.info('Inner loop {}'.format(index))
        # residual and Jacobian J = [JB JR]
        # log.debug('residual and Jacobian J = [JB JR]')
        r, phi, x, jb, jr = pclsr(psi, te, data, options)

        # gradient G = J'*r = [gB gR]
        # log.debug("gradient G = J'*r = [gB gR]")
        gb = np.real(np.sum(jb.squeeze().conj() * r.squeeze(), axis=0)) + (options['muB'] ** 2) * np.real(psi - psi_real)
        gr = np.real(np.sum(jr.squeeze().conj() * r.squeeze(), axis=0)) + (options['muR'] ** 2) * np.imag(psi - psi_real)
        g = gb + 1j * gr

        # Hessian H ~ Re(J'*J) = [H1 H2; H2 H3]
        # log.debug("Hessian H ~ Re(J'*J) = [H1 H2; H2 H3]")
        h1 = np.real(np.sum(jb.squeeze().conj() * jb.squeeze(), axis=0)) + options['muB'] ** 2
        h2 = np.real(np.sum(jb.squeeze().conj() * jr.squeeze(), axis=0))
        h3 = np.real(np.sum(jr.squeeze().conj() * jr.squeeze(), axis=0)) + options['muR'] ** 2

        # Cauchy point = (G'G)/(G'HG)
        # log.debug("Cauchy point = (G'G)/(G'HG)")
        gg = gb ** 2 + gr ** 2
        ghg = (gb ** 2) * h1 + 2 * gb * gr * h2 + (gr ** 2) * h3

        # stabilize step size: small and nonzero
        # log.debug('stabilize step size: small and nonzero')
        damp = np.median(ghg.squeeze()[options['mask']]) / 1000
        step = gg / (ghg + damp) / options['maxit'][2]
        dpsi = -step * g

        # basic linesearch
        for k in range(options['maxit'][2]):
            # log.info('Line search {}'.format(k))
            ok = cost_func(psi + dpsi, te, data, psi_real, options) < cost_func(psi, te, data, psi_real, options)
            psi[ok] += dpsi[ok]
            dpsi[~ok] = dpsi[~ok] / 10

    # final phi and x
    # log.debug('final phi and x')
    r, phi, x, a, b = pclsr(psi, te, data, options)

    # catch numerical instabilities(due to negative R2*)
    if np.any(~np.isfinite(psi)):
        log.error('Problem is too ill-conditioned: use nonnegR2 or increase muR.')
        return

    return r, psi, phi, x


def presco(te: np.ndarray, data: np.ndarray, tesla: float, none: bool = False,
           debug: bool = False) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, float):
    """
    [params sse] = presco(te, data)
    [params sse] = presco(imDataParams)

    Phase regularized estimation using smoothing and constrained optimization for the proton density
    fat fraction (PDFF).

    Works best with phase unwrapping (e.g. unwrap2.m & unwrap3.m from https://github.com/marcsous/unwrap).

    Inputs:
     te is echo time in seconds (vector)
     data (1D, 2D or 3D with echos in last dimension)
     Tesla field strength (scalar)
     varargin in option/value pairs (e.g. 'ndb',2.5)

    Alternate input:
     imDataParams is a struct for fat-water toolbox

    Outputs:
     params.B0 is B0 (Hz)
     params.R2 is R2* (1/s)
     params.FF is PDFF (%)
     params.PH is PH (rad)
     sse is sum of squares error
    """

    """ options """

    options = {
        # constraints
        'muB': 0.00,  # regularization for B0
        'muR': 0.05,  # regularization for R2*
        'nonnegFF': False,  # nonnegative PDFF (1=on 0=off)
        'nonnegR2': True,  # nonnegative R2* (1=on 0=off)
        'smooth_phase': True,  # smooth phase (1=on 0=off)
        'smooth_field': True,  # smooth field (1=on 0=off)

        # constants
        'ndb': 2.5,  # no. double bonds
        'h2o': 4.7,  # water frequency ppm
        'filter': np.ones([3, 3]),  # low pass filter
        'maxit': [10, 10, 3],  # max. iterations (inner outer linesearch)
        'noise': None,  # noise std (if available)

        # debugging options
        'unwrap': True,  # use phase unwrapping of fieldmap (0=off)
        'psi': None,  # fixed initial psi estimate (for testing)
    }

    if none:
        log.warning('All constraints turned off')
        options['muB'] = 0
        options['muR'] = 0
        options['nonnegFF'] = False
        options['nonnegR2'] = False
        options['smooth_phase'] = False

    if np.amax(te, axis=None) >= 1.0:
        log.error("TE should be in seconds.")
        return

    if not sorted_array(te):
        log.warning("TE should be sorted. Sorting...")
        te = te.sort()

    if not data.dtype == 'complex':
        log.error('data should be complex floats.')
        return

    # Forwards because it's a numpy array taken from nibabel (at the moment)
    nx, ny, nz, ne = data.shape

    """ setup """

    # estimate noise std dev (smallest sing. value)
    if not options['noise']:
        # log.debug('Adding noise')
        options['noise'] = set_noise(data, debug=False)

    # signal mask (needs work)
    options['mask'] = np.sum(data.conj() * data, axis=3).squeeze() > 2 * (options['noise'] ** 2)

    # time evolution matrix
    options['A'], options['psif'], _, _ = fat_basis(te, tesla, options['ndb'], options['h2o'])

    # display
    log.info('Data size: {}'.format(data.shape))
    log.info('TE (ms): {}'.format(1000 * te.flatten()))
    log.info('Tesla (T): {}'.format(tesla))
    for k, v in options.items():
        log.info('{}: {}'.format(k, v.shape if isinstance(v, np.ndarray) else v))

    # standardize mu across datasets
    if options['noise']:
        options['muR'] = options['muR'] * options['noise']
        options['muB'] = options['muB'] * options['noise']

    """ center kspace (otherwise smoothing is risky) """

    data_fft = fftn(data)
    tmp = np.sum(data_fft.conj() * data_fft, axis=3)
    k = np.argmax(tmp)
    dx, dy, dz = np.unravel_index(k, (tmp.shape[0], tmp.shape[1], tmp.shape[2]))
    data_fft = np.roll(data_fft, (-dx, -dy, -dz), axis=(0, 1, 2))
    log.info('Shifting k-space center from [{}, {}, {}]'.format(dx, dy, dz))
    data = ifftn(data_fft)

    """ initial estimates """

    # crop = {'x': list(range(91 - 6, 91 + 6)), 'y': list(range(71 - 6, 71 + 6))}
    if not options['psi']:
        # dominant frequency (rad/s)
        tmp = np.sum(data[:, :, :, slice(0, ne - 1)].conj() * data[:, :, :, slice(1, ne)], axis=3).squeeze()
        psi = np.angle(tmp) / np.min(np.diff(te, axis=0)) + 1j * options['psif'].imag
        psi = psi[..., np.newaxis]
    else:
        # supplied psi (convert to rad/s)
        psi = 2 * np.pi * options['psif'].real + 1j * options['psif'].imag
        psi = np.tile(psi, (nx, ny, nz, 1))

    # echos in 1st dim
    psi = psi.transpose((2, 0, 1))
    data = data.transpose((3, 0, 1, 2))

    # allow for spatially varying regularization
    if np.size(options['muR']) > 1:
        options['muR'] = (np.repeat(options['muR'], np.size(psi))).reshape(psi.shape)
        log.error('PANIC, spatially varying muR')
        return

    if np.size(options['muB']) > 1:
        options['muB'] = (np.repeat(options['muB'], np.size(psi))).reshape(psi.shape)
        log.error('PANIC, spatial varying muB')
        return

    """ main algorithm """

    # the first opts.maxit-1 iterations are really only to get
    # rid of fat-water swaps. certain constraints don't make
    # sense if there are still fat-water swaps present so we can
    # take liberties and apply constraints only when they make
    # sense. e.g. this may mean setting muB and smooth_phase to
    # 0 on the first iterations.

    mu_b = options['muB'].copy()
    smooth_phase = copy.copy(options['smooth_phase'])

    for index in range(options['maxit'][0]):

        # fiddle with options, as discussed above
        if index == 0 and options['maxit'][1] > 1:
            if debug:
                log.debug('Fiddle with options, as discussed above')
            options['muB'] = 0.0
            options['smooth_phase'] = False
        else:
            options['muB'] = mu_b
            options['smooth_phase'] = smooth_phase

        # local optimization
        if debug:
            log.debug('Local optimization')
        r, psi, phi, x = nls_fit_presco(psi, te, data, options)
        log.debug('Iteration {}\t{:.3f}'.format(index, la.norm(r)))

        # resolve phase swaps
        if index < options['maxit'][1]:
            if options['unwrap']:
                if index == 0 and not options['psi']:
                    if debug:
                        log.debug('estimate based on least squares')
                    s, _, _, _, _ = pclsr(psi - options['psif'], te, data, options)
                    bad = mdot(r.squeeze(), r.squeeze()) > mdot(s.squeeze(), s.squeeze())
                    bad = bad[np.newaxis, ...]
                    psi[bad] = psi[bad] - options['psif']
                elif index > options['maxit'][1] / 2 - 1:
                    if debug:
                        log.debug('phase unwrapping')
                    psi = unswap(psi, te, options)

            if options['smooth_field']:
                if debug:
                    log.debug('low pass filtering')
                b0 = psi.squeeze().real  # only smooth B0
                b0 = medfiltn(b0, np.array(options['filter'].shape), options['mask'])
                psi = b0[np.newaxis, ...] + 1j * psi.imag

    """ return parameters in correct format """

    psi, _ = transform(psi, options)
    b0 = psi.squeeze().real / 2 / np.pi  # B0 (Hz)
    r2_star = psi.squeeze().imag  # R2* (1/s)
    ff = 100 * x[1, :, :, :].squeeze() / np.sum(x.squeeze(), axis=0)  # FF (%)
    ff[~options['mask']] = 0.0  # remove NaNs from 0/0 pixels
    phase = phi.squeeze()  # initial phase (rad)
    sse = np.sum(r.conj() * r, axis=None).real  # sum of squares error
    log.info('SSE: {:.3f}'.format(sse))

    return b0, ff, r2_star, phase, sse


def model_func(proton_density: np.ndarray, fat_fraction: np.ndarray, decay_rate: np.ndarray, te: np.array,
               data: np.ndarray, options: Dict) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """ model function """

    # no. echos
    ne = len(te)

    # nonnegative M and R2*
    proton_density = np.abs(proton_density)
    if options['nonnegR2']:
        decay_rate = np.abs(decay_rate)

    # components of the signal
    fat = options['A'][:, 1, np.newaxis] @ fat_fraction
    water = options['A'][:, 0, np.newaxis] @ (1 - fat_fraction)

    s1 = np.tile(proton_density, (ne, 1))
    s2 = np.abs(water + fat)
    s3 = np.exp(-te @ decay_rate)

    # signal
    f = np.hypot(s1 * s2 * s3, options['noise'])

    # derivatives
    jacob_m = s1 * (s2 ** 2) * (s3 ** 2) / f
    jacob_r = -te * jacob_m * s1

    if options['nonnegR2']:
        sign_r = np.sign(decay_rate) + (decay_rate == 0)
        jacob_r = jacob_r * sign_r

    # JF (analytical)
    tmp = np.abs(options['A'][:, 1]) ** 2 - np.real(options['A'][:, 0]) ** 2 - 2 * np.real(np.prod(options['A'], 1))
    tmp = np.diff(np.real(options['A']), 1, 1) + tmp[..., np.newaxis] @ fat_fraction
    jacob_f = (s1 ** 2) * (s3 ** 2) * tmp / f

    # JF (finite difference)
    # h = 1e-4
    # s4 = bsxfun(@times,diff(options['A'],1,2),h)
    # s2 = abs(water+fat+s4)
    # JF = (hypot(s1*s2*s3,options['noise'])-f)./h

    # residual (or display)
    if f.size == data.size:
        f = f - np.reshape(data, f.shape)

    return f, jacob_m, jacob_f, jacob_r


def nls_fit_lipoquant(proton_density: np.ndarray, fat_fraction: np.ndarray, decay_rate: np.ndarray, te: np.array,
                      data: np.ndarray, tesla: float, options: Dict) -> (float, np.ndarray, np.ndarray, np.ndarray):
    """ nonlinear least squares fitting """

    # water-fat matrix
    options['A'], _, _, _ = fat_basis(te, tesla, options['ndb'], options['h2o'])

    # residual and Jacobian
    resid, jacob_m, jacob_f, jacob_r = model_func(proton_density, fat_fraction, decay_rate, te, data, options)

    for i in range(options['maxit']):

        # Normal matrix for the Jacobian is 3x3, which has an exact inverse:
        #
        # | a11 a12 a13 |-1           |   a33.a22-a32.a23 -(a33.a12-a32.a13)  a23.a12-a22.a13  |
        # | a21 a22 a23 |   = 1/dtm * | -(a33.a21-a31.a23)  a33.a11-a31.a13 -(a23.a11-a21.a13) |
        # | a31 a32 a33 |             |   a32.a21-a31.a22 -(a32.a11-a31.a12)  a22.a11-a21.a12  |
        #
        # with dtm = a11.(a33.a22-a32.a23)-a21.(a33.a12-a32.a13)+a31.(a23.a12-a22.a13)

        a11 = np.sum(jacob_m.conj() * jacob_m, axis=0)  # dot(JM, JM) in MATLAB
        a12 = np.sum(jacob_m.conj() * jacob_f, axis=0)
        a13 = np.sum(jacob_m.conj() * jacob_r, axis=0)
        a21 = a12
        a22 = np.sum(jacob_f.conj() * jacob_f, axis=0)
        a23 = np.sum(jacob_f.conj() * jacob_r, axis=0)
        a31 = a13
        a32 = a23
        a33 = np.sum(jacob_r.conj() * jacob_r, axis=0)

        # Levenberg-Marquardt term
        if i == 0:
            lamb = (a11 + a22 + a33 + np.mean(data)) / 1000

        a11 = a11 + lamb
        a22 = a22 + lamb
        a33 = a33 + lamb

        dtm = a11 * (a33 * a22 - a32 * a23) - a21 * (a33 * a12 - a32 * a13) + a31 * (a23 * a12 - a22 * a13)

        # gradient (J'*r)
        grad_m = np.sum(jacob_m.conj() * resid, axis=0)
        grad_f = np.sum(jacob_f.conj() * resid, axis=0)
        grad_r = np.sum(jacob_r.conj() * resid, axis=0)

        # updates = -inv(J'J+lamb.I)*J'r
        delta_m = (-(a33 * a22 - a32 * a23) * grad_m + (a33 * a12 - a32 * a13) *
                   grad_f - (a23 * a12 - a22 * a13) * grad_r) / dtm
        delta_f = (+(a33 * a21 - a31 * a23) * grad_m - (a33 * a11 - a31 * a13) *
                   grad_f + (a23 * a11 - a21 * a13) * grad_r) / dtm
        delta_r = (-(a32 * a21 - a31 * a22) * grad_m + (a32 * a11 - a31 * a12) *
                   grad_f - (a22 * a11 - a21 * a12) * grad_r) / dtm

        # evaluate at new estimate
        resid_new, jacob_m_new, jacob_f_new, jacob_r_new = model_func(proton_density + delta_m, fat_fraction + delta_f,
                                                                      decay_rate + delta_r, te, data, options)

        # accept changes that gave an improvement
        ok = np.sum(resid_new.conj() * resid_new, axis=0) < np.sum(resid.conj() * resid, axis=0)
        resid[:, ok] = resid_new[:, ok]
        proton_density[:, ok] = proton_density[:, ok] + delta_m[ok]
        fat_fraction[:, ok] = fat_fraction[:, ok] + delta_f[ok]
        decay_rate[:, ok] = decay_rate[:, ok] + delta_r[ok]
        jacob_m[:, ok] = jacob_m_new[:, ok]
        jacob_f[:, ok] = jacob_f_new[:, ok]
        jacob_r[:, ok] = jacob_r_new[:, ok]

        # update damping term
        lamb[ok] = lamb[ok] / 10
        lamb[~ok] = lamb[~ok] * 9

    return sum(resid ** 2), proton_density, fat_fraction, decay_rate


def lipoquant(te: np.array, data: np.ndarray, tesla: float) -> (np.ndarray, np.ndarray, float):
    """
    [params sse] = lipoquant(te,data,Tesla,varargin)

     Proton density fat fraction estimation using magnitude.
     Returns struct of FF (%) and R2* (1/s).

     Inputs:
      te is echo time in seconds (vector)
      data (1D, 2D or 3D with echos in last dimension)
      Tesla field strength (scalar)
      varargin in option/value pairs (e.g. 'init',0.1)

     Alternate input:
      imDataParams is a struct from fat-water toolbox

     Outputs:
      params.R2 is R2* (1/s)
      params.FF is FF (%)
      sse is sum of squares error
    """

    if len(te) < 3:
        log.error('too few echos ({}) for FF, R2* estimation'.format(len(te)))
        raise RuntimeError
    if max(te) > 1:
        log.error('te should be in seconds')
        raise RuntimeError

    """ default options """

    options = {
        'ndb': 2.5,        # number of double bonds
        'h2o': 4.7,        # water ppm
        'maxit': 20,       # maximum number of iterations
        'nonnegR2': True,  # nonnegative R2*
        'noise': 0.0,      # noise standard deviation (if available)
        'init': 0.1,       # initial fat fraction estimate (0 - 1)
    }

    if isinstance(te, list):
        te = np.array(te)
    data = data.astype(np.float)

    # Backwards because it's a numpy array taken from SimpleITK
    ne, nz, ny, nx = data.shape

    # display
    log.info('Data size: {}'.format(data.shape))
    log.info('TE (ms): {}'.format(1000 * te.flatten()))
    log.info('Tesla (T): {}'.format(tesla))
    for k, v in options.items():
        log.info('{}: {}'.format(k, v.shape if isinstance(v, np.ndarray) else v))

    """ faster calculations """

    # echos in first dimension then (x, y, z)
    data = np.transpose(data, (0, 3, 2, 1))

    # consistent data types
    te = np.reshape(np.real(te.astype(data.dtype)), (ne, 1))

    """ nonlinear estimation (local optmization) """

    # initial estimates
    if not np.isreal(options['init']) or np.count_nonzero(options['init'] < 0 or options['init'] > 1) > 0:
        log.warning("options['init'] values should be [0 1]")

    options['init'] = min(np.max(np.real(options['init']), 0), 1)

    if np.isscalar(options['init']):
        fat_fraction = np.tile(options['init'].astype(data.dtype), (1, nx * ny * nz))
    else:
        fat_fraction = np.reshape(options['init'].astype(data.dtype), (1, nx * ny * nz))

    proton_density = np.amax(data, 0).flatten()[np.newaxis, ...]
    decay_rate = np.tile(np.float(0), (1, nx * ny * nz))

    # find the local minimum
    sse, _, fat_fraction, decay_rate = nls_fit_lipoquant(proton_density, fat_fraction, decay_rate, te, data, tesla,
                                                         options)

    """ return parameters in correct format """

    # nonegative M and R2*
    if options['nonnegR2']:
        decay_rate = np.abs(decay_rate)

    fat_fraction = np.reshape(fat_fraction, (nx, ny, nz))
    decay_rate = np.reshape(decay_rate, (nx, ny, nz))
    sse = np.reshape(sse, (nx, ny, nz))

    log.debug('sse = {:0.3f}'.format(np.sqrt(np.sum(sse))))

    return 100 * fat_fraction, decay_rate, sse


def fat_fraction_from_multiecho(organ: str, magnitude_only: bool = True, lut: str = 'cividis',
                                manufacturer: str = 'Siemens', debug: bool = False) -> None:
    if organ == 'multiecho_liver':
        if ds.nifti.exists(fn.liver_mag.value):
            file_magnitude = ds.nifti.path(fn.liver_mag.value)
            file_phase = ds.nifti.path(fn.liver_phase.value)
        else:
            log.warning('No {} magnitude acquisition has been detected'.format(organ))
            return
    elif organ == 'ideal_liver':
        if ds.nifti.exists(fn.ideal_liver_mag.value):
            file_magnitude = ds.nifti.path(fn.ideal_liver_mag.value)
            file_phase = ds.nifti.path(fn.ideal_liver_phase.value)
        else:
            log.warning('No {} magnitude acquisition has been detected'.format(organ))
            return
    elif organ == 'multiecho_pancreas':
        if ds.nifti.exists(fn.pancreas_mag.value):
            file_magnitude = ds.nifti.path(fn.pancreas_mag.value)
            file_phase = ds.nifti.path(fn.pancreas_phase.value)
        else:
            log.warning('No {} magnitude acquisition has been detected'.format(organ))
            return
    else:
        log.error("Only 'multiecho_liver', 'multiecho_pancreas' and 'ideal_liver' are valid strings")
        return

    # It is *CRUCIAL* that the echo times are in milliseconds!
    if organ in ['multiecho_liver', 'multiecho_pancreas', 'ideal_liver']:
        magnitude_nifti = NiftiWrapper.from_filename(file_magnitude)
        magnitude_meta = magnitude_nifti.meta_ext
        echo_times = np.asarray(magnitude_meta.get_values('EchoTime')).astype(float)
    else:
        with open('{}.json'.format(organ), 'r') as jsonfile:
            metadata = json.load(jsonfile)
        echo_times = 1000 * np.asarray(metadata['TE']).astype(float)
    if debug:
        log.info('Echo times from metadata (milliseconds):\n  {}'.format(echo_times))

    magnitude = sitk.ReadImage(file_magnitude)
    np_magnitude = sitk.GetArrayFromImage(magnitude)
    x, y, z, t = magnitude.GetSize()
    if x == 1 or y == 1:
        log.error('Only axial (transverse) slices are allowed, PDFF not estimated')
        return
    if magnitude_only:
        algorithm = 'lipoquant'
        np_fat_fraction, np_r2star, _ = lipoquant(echo_times / 1000, np_magnitude, tesla=1.5)
    else:
        algorithm = 'presco'
        phase = sitk.ReadImage(file_phase)
        np_phase = sitk.GetArrayFromImage(phase)
        if manufacturer == 'Siemens':
            data = np_magnitude * np.exp(1j * np.pi * np_phase / 4096)  # Siemens saves phase as complex
        elif manufacturer == 'GE':
            data = np_magnitude * np.exp(-1j * np.pi * np_phase / 4096)  # GE saves phase as complex conjugate
        else:
            log.error("Scanner manufacturer '{}' not recognized!".format(manufacturer))
            return
        data = np.transpose(data, (3, 2, 1, 0))  # because SimpleITK -> numpy has reversed indices
        echo_times = np.ndarray((len(echo_times), 1), buffer=np.array(echo_times))
        np_b0, np_fat_fraction, np_r2star, np_ph, _ = presco(echo_times / 1000, data, tesla=1.5)
        np_fat_fraction = np_fat_fraction[..., np.newaxis]
        np_r2star = np_r2star[..., np.newaxis]
        b0 = sitk.GetImageFromArray(np.transpose(np_b0[..., np.newaxis], (2, 1, 0)))
        b0.CopyInformation(sitk.Extract(magnitude, (x, y, z, 0), (0, 0, 0, 0)))
        if not ds.analysis.exists():
            os.makedirs(ds.analysis.value)
        sitk.WriteImage(b0, ds.analysis.path('{}_{}_b0.nii.gz'.format(organ.replace('_', '.'), algorithm)))
        ph = sitk.GetImageFromArray(np.transpose(np_ph[..., np.newaxis], (2, 1, 0)))
        ph.CopyInformation(sitk.Extract(magnitude, (x, y, z, 0), (0, 0, 0, 0)))
        sitk.WriteImage(ph, ds.analysis.path('{}_{}_phase.nii.gz'.format(organ.replace('_', '.'), algorithm)))

    fat_fraction = sitk.GetImageFromArray(np.transpose(np_fat_fraction, (2, 1, 0)))
    fat_fraction.CopyInformation(sitk.Extract(magnitude, (x, y, z, 0), (0, 0, 0, 0)))
    if not ds.analysis.exists():
        os.makedirs(ds.analysis.value)
    sitk.WriteImage(fat_fraction, ds.analysis.path('{}_{}_pdff.nii.gz'.format(organ.replace('_', '.'), algorithm)))
    r2star = sitk.GetImageFromArray(np.transpose(np_r2star, (2, 1, 0)))
    r2star.CopyInformation(sitk.Extract(magnitude, (x, y, z, 0), (0, 0, 0, 0)))
    sitk.WriteImage(r2star, ds.analysis.path('{}_{}_r2star.nii.gz'.format(organ.replace('_', '.'), algorithm)))


def iron_concentration_from_r2star(organ: str, magnitude_only: bool = False) -> None:
    """
    From:
    Measurement of liver iron by magnetic resonance imaging in the UK Biobank population
    A. McKay, H. R. Wilman, A. Dennis, M. Kelly, M. L. Gyngell, S. Neubauer, J. D. Bell, R. Banerjee, E. L. Thomas
    PLoS ONE 13(12): e0209340. https://doi.org/10.1371/journal.pone.0209340

    Using:
    Wood JC, Enriquez C, Ghugre N, Tyzka JM, Carson S, Nelson MD, et al. MRI R2 and R2* mapping accurately estimates
    hepatic iron concentration in transfusion-dependent thalassemia and sickle cell disease patients.
    Blood 2005; 106: 1460–5. https://doi.org/10.1182/blood-2004-10-3982 PMID: 15860670
    """

    if organ in ['multiecho_liver', 'ideal_liver', 'multiecho_pancreas']:
        if magnitude_only:
            file_r2star = ds.analysis.path('{}_lipoquant_r2star.nii.gz'.format(organ.replace('_', '.')))
        else:
            file_r2star = ds.analysis.path('{}_presco_r2star.nii.gz'.format(organ.replace('_', '.')))
        file_iron = file_r2star.replace('r2star', 'iron')
    else:
        log.error("Only 'multiecho_liver', 'multiecho_pancreas' and 'ideal_liver' are valid strings")
        raise RuntimeError

    if not os.path.exists(file_r2star):
        log.warning("No '{}' r2star file has been detected".format(organ))
        return

    log.info('Computing iron concentration for {}'.format(organ))
    # units are (mg iron) / (g dry tissue)
    sitk.WriteImage(0.0254 * sitk.ReadImage(file_r2star) + 0.202, file_iron)
