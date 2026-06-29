"""Generate a smooth bottle STL via surface of revolution."""
import numpy as np
import struct

def write_stl(path, triangles):
    with open(path, 'wb') as f:
        f.write(b'\x00' * 80)
        f.write(struct.pack('<I', len(triangles)))
        for tri in triangles:
            n = np.cross(tri[1]-tri[0], tri[2]-tri[0])
            nn = np.linalg.norm(n)
            n = n / nn if nn > 1e-10 else n
            f.write(struct.pack('<fff', *n))
            for v in tri:
                f.write(struct.pack('<fff', *v))
            f.write(b'\x00\x00')

def revolve(profile_r, profile_z, n_phi=64):
    """Revolve a 2D profile (r, z) around Z axis."""
    triangles = []
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    cos_p = np.cos(phis)
    sin_p = np.sin(phis)

    for i in range(len(profile_r) - 1):
        r0, z0 = profile_r[i],   profile_z[i]
        r1, z1 = profile_r[i+1], profile_z[i+1]
        for j in range(n_phi):
            j2 = (j + 1) % n_phi
            v00 = np.array([r0*cos_p[j],  r0*sin_p[j],  z0])
            v01 = np.array([r0*cos_p[j2], r0*sin_p[j2], z0])
            v10 = np.array([r1*cos_p[j],  r1*sin_p[j],  z1])
            v11 = np.array([r1*cos_p[j2], r1*sin_p[j2], z1])
            triangles.append([v00, v10, v11])
            triangles.append([v00, v11, v01])
    return triangles

def cap(r, z, n_phi=64, flip=False):
    """Flat circular cap."""
    triangles = []
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    center = np.array([0.0, 0.0, z])
    for j in range(n_phi):
        j2 = (j + 1) % n_phi
        v0 = np.array([r*np.cos(phis[j]),  r*np.sin(phis[j]),  z])
        v1 = np.array([r*np.cos(phis[j2]), r*np.sin(phis[j2]), z])
        if flip:
            triangles.append([center, v1, v0])
        else:
            triangles.append([center, v0, v1])
    return triangles

# --- bottle profile (r, z), z=0 at bottom center ---
# Smooth wine-bottle-like shape
t = np.linspace(0, 1, 120)

def bottle_profile(t):
    z = t * 0.20  # total height 20cm
    # piecewise smooth radius profile
    r = np.where(
        t < 0.08,   # flat bottom edge
        0.035 + 0.0*t,
        np.where(
            t < 0.15,  # bottom rounding
            0.035 + 0.008 * np.sin((t-0.08)/0.07 * np.pi/2),
            np.where(
                t < 0.55,  # body
                0.033 + 0.005 * np.sin((t-0.15)/0.40 * np.pi),
                np.where(
                    t < 0.68,  # shoulder taper
                    0.033 * (1 - ((t-0.55)/0.13)**1.5 * 0.7),
                    np.where(
                        t < 0.78,  # neck
                        0.012 + 0.002*(1-((t-0.68)/0.10)),
                        np.where(
                            t < 0.90,  # slight lip flare
                            0.011 + 0.003*((t-0.78)/0.12),
                            0.014  # lip
                        )
                    )
                )
            )
        )
    )
    return r, z

r_prof, z_prof = bottle_profile(t)
# center so bottom is at z = -0.10, top at +0.10
z_prof = z_prof - z_prof[-1]/2

tris = revolve(r_prof, z_prof, n_phi=72)
tris += cap(r_prof[0],  z_prof[0],  n_phi=72, flip=True)   # bottom cap
tris += cap(r_prof[-1], z_prof[-1], n_phi=72, flip=False)  # top cap

out = '/home/qsq/qqsq12321/vr_teleop/example/scene/meshes/bottle_smooth.stl'
write_stl(out, tris)
print(f'写入 {out}，面数: {len(tris)}')
