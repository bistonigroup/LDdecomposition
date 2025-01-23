"""
(c) 2024, Lorenzo Baldinelli, Filippo De Angelis, Giovanni Bistoni

License: MIT License

Credits: If you use this code, please cite the work "J. Chem. Theory Comput. 2024, 20, 5, 1923–1931; DOI: 10.1021/acs.jctc.3c00977"

Description: This script computes the atomic contributions to the London dispersion energy as well as the London dispersion density function based on the D3 dispersion correction, as described in Ref. ####

Before running the script, ensure the DFT-D3 executable is located at the specified path (currently, pathd3 = "./dftd3").
The DFT-D3 excecutable can be downloaded from https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3 
Please also cite the related papers of Prof. Grimme and coworkers.

As input file only a "{basename}.xyz" file is needed (in Angstroem).
At the end of the calculation, three output files will be generated:
    - ".atomwise.txt": a file where coordinates are accompanied by a column indicating the contribution to the London dispersion energy for each atom.
    - ".d3out.txt": the original DFT-D3 output (J. Chem. Phys. 132, 154104 (2010); DOI: 10.1063/1.3382344).
    - ".omega.cube": a function of spatial coordinates, facilitating visualization and analysis of dispersion energy contributions.

Run: python3 lddensity.py basename npoints func damp nprocs (e.g., if water.xyz is provided as input, python3 lddensity.py water 40 b3-lyp bj 2).

Arguments:
    basename -> file name without the extension ".xyz".
    npoints -> the number of grid points for each dimension (e.g., 80).
    func -> the functional parameters used in the D3 dispersion calculation (e.g., b3-lyp).
    damp -> the damping scheme used in the D3 dispersion calculation (e.g., bj).
    nprocs -> the number of processors for parallel calculations.
"""

def read_xyz(xyz):
    atoms = []
    x, y, z = [], [], []

    with open(xyz) as fp:
        # Skip the first two lines.
        next(fp)
        next(fp)
        for line in fp:
            data = line.split()
            atoms.append(data[0])
            x.append(float(data[1]))
            y.append(float(data[2]))
            z.append(float(data[3]))
            nat = len(atoms)

    return atoms, x, y, z, nat

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
    import subprocess
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
    func = sys.argv[3]
    damp = sys.argv[4]
    nproc = int(sys.argv[5])
    xyz = f"{basename}.xyz"
    atw = f"{basename}.{npoints}.{func}.{damp}.atomwise.txt"
    d3out = f"{basename}.{npoints}.{func}.{damp}.d3out.txt"
    omegaout = f"{basename}.{npoints}.{func}.{damp}.omega.cube"

    if not os.path.isfile(xyz):
        sys.exit("Could not find the .xyz file")

    atoms, x, y, z, natoms = read_xyz(xyz)

    extent = 7.0
    xmin = min(x) * ang_to_au - extent
    xmax = max(x) * ang_to_au + extent
    ymin = min(y) * ang_to_au - extent
    ymax = max(y) * ang_to_au + extent
    zmin = min(z) * ang_to_au - extent
    zmax = max(z) * ang_to_au + extent

    pathd3 = "./dftd3"

    subprocess.run([f"{pathd3} "
                    f"{xyz} -func {func} -{damp} -anal > "
                    f"{d3out}"], shell=True)

    with open(f"{d3out}", "r") as d3realsys:
        d3rs = d3realsys.readlines()
        # generate list to store pairwise terms:
        atwdisp = [0.000000] * natoms
        combo = int(natoms * (natoms - 1) / 2)

        for j in range(len(d3rs)):
            if ('Edisp /kcal,au:' in d3rs[j]):
                erealsys = d3rs[j].strip().split()
                Esyskcal = float(erealsys[2])
                Esysau = float(erealsys[3])

            if ('analysis of pair-wise terms (in kcal/mol)' in d3rs[j]):
                for l in range(j + 2, j + 2 + combo):
                    epair = d3rs[l].strip().split()
                    if len(epair) < 9:
                        break
                    iat, jat, eij = int(epair[0]), int(epair[1]), float(epair[8])
                    atwdisp[iat - 1] = float((atwdisp[iat - 1] + eij / 2.00000))
                    atwdisp[jat - 1] = float((atwdisp[jat - 1] + eij / 2.00000))

    # generate the atomwise txt file where the atomic d3 contributions are listed
    with open(f"{atw}", "w") as pwout:
        pwout.write("analysis of atom-wise contributions (in kcal/mol)\n")
        pwout.write("          X            Y           Z         Edisp\n")
        for el in range(len(atwdisp)):
            at = int(el + 1)
            pwout.write(f"{atoms[el]} {x[el]:12.6f} {y[el]:12.6f}"
                        f"{z[el]:12.6f}{atwdisp[el]:12.6f}\n")

    with open(f"{omegaout}", "w") as fp:
        fp.write(f"LD density ({npoints} grid points)\n")
        fp.write(f"input file: {basename}.xyz ({func} {damp})\n")
        fp.write(f"{len(atoms):5d}{xmin:12.6f}{ymin:12.6f}{zmin:12.6f}\n")

        xstep = (xmax - xmin) / float(npoints - 1)
        fp.write(f"{npoints:5d}{xstep:12.6f}{0:12.6f}{0:12.6f}\n")

        ystep = (ymax - ymin) / float(npoints - 1)
        fp.write(f"{npoints:5d}{0:12.6f}{ystep:12.6f}{0:12.6f}\n")

        zstep = (zmax - zmin) / float(npoints - 1)
        fp.write(f"{npoints:5d}{0:12.6f}{0:12.6f}{zstep:12.6f}\n")

        for i, atom in enumerate(atoms):
            index = elements.index(atom)
            xi, yi, zi = x[i] * ang_to_au, y[i] * ang_to_au, z[i] * ang_to_au
            fp.write(f"{index:5d}{index:12.6f}{xi:12.6f}{yi:12.6f}{zi:12.6f}\n")

        xyzbox = box_gen(xmin, ymin, zmin, xstep, ystep, zstep, npoints)

        omega = omega_comp(nproc, ang_to_au, xyzbox, x, y, z, atwdisp)
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

        print(f"d3 atomwise analysis info for {basename}.xyz ({func} {damp})")
        print("")
        print(f"total dispersion energy: {Esyskcal:10.3f}")
        print("")
        print(f"integral of omega function: {omegaintegral:10.3f}")
        print("")

exetime = (time.time() - start_time)
print(f"execution time: {exetime:10.5f} seconds")
