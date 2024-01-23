"""
(c) 2024, Lorenzo Baldinelli, Filippo De Angelis, Giovanni Bistoni

License: MIT License

Credits: If you use this code, please cite the work ###

Description: This script computes the London Dispersion density difference function, which is described in Ref. ###

As input file only the "{basename}.atomwise.txt" is needed.  This contains the spatial coordinates of a chemical system in XYZ format and, as a 5th column, the difference between the atomic LD energy of each atom minus that obtained for a different molecular structure.  
For example, the atomic LD energies can be obtained using the lddensity.py tool. However, this tool can be used in principle in conjuction with any arbitrary atom-wise decomposition scheme.

At the end of the calculation, one output file will be generated:
    - ".omega.cube": The London dispersion density difference function in cube format.

Run: python3 lddensitydifference.py basename npoints nprocs (e.g., if water.atomwise.txt is provided as input, python3 lddensitydifference.py water 80 1).

Arguments:
    basename -> file name without the extension ".atomwise.txt".
    npoint -> the number of grid points for each dimension (e.g., 80).
    nprocs -> the number of processors for parallel calculations.
"""

def box_gen(xmin, ymin, zmin, xstep, ystep, zstep, npoints):
    xyzbox = []
    for p in range(npoints):
        xpt = xmin + p * xstep
        for q in range(npoints):
            ypt = ymin + q * ystep
            for r in range(npoints):
                zpt = zmin + r * zstep
                xyzbox.append((xpt,ypt,zpt))
    return xyzbox

def omega_comp_wrapper(args):
    ang_to_au, boxpt, xyzcoord_au, atwdisp = args
    omegaval, ith = 0.0, 0
    a = 0.5
    norm = (1.0 / (math.pi / a)) ** 1.5
    for syspt in xyzcoord_au:
        rbox_rith = (boxpt[0] - syspt[0]) ** 2 + (boxpt[1] - syspt[1]) ** 2 + (boxpt[2] - syspt[2]) ** 2
        exp_norm = norm * math.exp(- a * rbox_rith)
        ld_rbox_rith = atwdisp[ith] * exp_norm
        omegaval += ld_rbox_rith
        ith += 1
    return omegaval

def omega_comp(nproc, ang_to_au, xyzbox, x, y, z, atwdisp):
    xyzcoord_au = [[x[i] * ang_to_au, y[i] * ang_to_au, z[i] * ang_to_au]  for i in range(len(x))]

    with Pool(nproc) as pool:
        args = [(ang_to_au, boxpt, xyzcoord_au, atwdisp) for boxpt in xyzbox]
        results = pool.map(omega_comp_wrapper, args)
    omega = results
    return omega

if __name__ == "__main__":
    import os
    from multiprocessing import Pool
    import sys
    import time
    import math

    start_time = time.time()

    ang_to_au = 1.0 / 0.5291772083
    au_to_ang = 0.5291772083

    elements = [None,
     "H", "He",
     "Li", "Be",
     "B", "C", "N", "O", "F", "Ne",
     "Na", "Mg",
     "Al", "Si", "P", "S", "Cl", "Ar",
     "K", "Ca",
     "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
     "Ga", "Ge", "As", "Se", "Br", "Kr",
     "Rb", "Sr",
     "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
     "In", "Sn", "Sb", "Te", "I", "Xe",
     "Cs", "Ba",
     "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
     "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
     "Tl", "Pb", "Bi", "Po", "At", "Rn",
     "Fr", "Ra",
     "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No",
     "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Uub"]

    basename = sys.argv[1]
    npoints = int(sys.argv[2])
    nproc = int(sys.argv[3])
    atwinp = f"{basename}.atomwise.txt"
    omegaout = f"{basename}.{npoints}.omega.cube"

    if not os.path.isfile(atwinp):
        sys.exit(f"Could not find the *.atomwise.txt file")

    # start
    with open(atwinp, "r") as atwread:
        atw = atwread.readlines()
        nlines = len(atw)
        at, xat, yat, zat, atwdisp = [], [], [], [], []
        for line in range(2, nlines):
            atom, x, y, z, atwd3 = atw[line].strip().split()
            at.append(str(atom))
            xat.append(float(x))
            yat.append(float(y))
            zat.append(float(z))
            atwdisp.append(float(atwd3))

    extent = 7.0
    xmin = min(xat) * ang_to_au - extent
    xmax = max(xat) * ang_to_au + extent
    ymin = min(yat) * ang_to_au - extent
    ymax = max(yat) * ang_to_au + extent
    zmin = min(zat) * ang_to_au - extent
    zmax = max(zat) * ang_to_au + extent


    with open(f"{omegaout}", "w") as fp:
        fp.write(f"LD difference density ({npoints} grid points)\n")
        fp.write(f"input file: {basename}.atomwise.txt\n")
        fp.write(f"{len(at):5d}{xmin:12.6f}{ymin:12.6f}{zmin:12.6f}\n")

        xstep = (xmax - xmin) / float(npoints - 1)
        fp.write(f"{npoints:5d}{xstep:12.6f}{0:12.6f}{0:12.6f}\n")

        ystep = (ymax - ymin) / float(npoints - 1)
        fp.write(f"{npoints:5d}{0:12.6f}{ystep:12.6f}{0:12.6f}\n")

        zstep = (zmax - zmin) / float(npoints - 1)
        fp.write(f"{npoints:5d}{0:12.6f}{0:12.6f}{zstep:12.6f}\n")

        for i, atom in enumerate(at):
            index = elements.index(atom)
            xi, yi, zi = xat[i] * ang_to_au, yat[i] * ang_to_au, zat[i] * ang_to_au
            fp.write(f"{index:5d}{index:12.6f}{xi:12.6f}{yi:12.6f}{zi:12.6f}\n")

        xyzbox = box_gen(xmin, ymin, zmin, xstep, ystep, zstep, npoints)

        omega = omega_comp(nproc, ang_to_au, xyzbox, xat, yat, zat, atwdisp)
        m = 0
        n = 0
        num = 0
        for ix in range(npoints):
            for iy in range(npoints):
                for iz in range(npoints):
                    fp.write(f"{omega[num]:14.5e}")
                    m += 1
                    n += 1
                    num += 1
                    if (n > 5):
                        fp.write("\n")
                        n = 0
                if n != 0:
                    fp.write("\n")
                    n = 0
        fp.write(" ")

        omegaintegral = sum(omega) * xstep * ystep * zstep

        print(f"total dispersion energy: {sum(atwdisp):10.3f}")
        print("")
        print(f"integral of omega function: {omegaintegral:10.3f}")
        print("")

exetime = (time.time() - start_time)
print(f"execution time: {exetime:10.5f} seconds")
