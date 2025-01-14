import os
from ase import Atoms
from ase.build import stack
from ase.io import read, write
from ase.md.npt import NPT
from ase import units
from ase.neighborlist import NeighborList
from mattersim.forcefield.potential import MatterSimCalculator
from mattersim.utils.supercell_utils import get_supercell_parameters

#Build solid-liquid coexistence structure
def build_solid_liquid_coexistence(
    solid_file: str,
    liquid_file: str,
    max_atoms_solid: int = 500,
    max_atoms_liquid: int = 500,
    direction='z',
    vacuum_gap=0.0
):
    print(f"[INFO] Reading and building solid-liquid coexistence structure: {solid_file} + {liquid_file}")

    #Read structures and automatically determine supercell
    atoms_solid = read(solid_file)
    atoms_liquid = read(liquid_file)
    nrep_solid, _ = get_supercell_parameters(atoms_solid, max_atoms=max_atoms_solid)
    nrep_liquid, _ = get_supercell_parameters(atoms_liquid, max_atoms=max_atoms_liquid)
    atoms_solid = atoms_solid.repeat(nrep_solid)
    atoms_liquid = atoms_liquid.repeat(nrep_liquid)

    #Stack the structures
    axis = {'x': 0, 'y': 1, 'z': 2}[direction.lower()]
    coexist = stack(atoms_solid, atoms_liquid, axis=axis, distance=vacuum_gap)
    return coexist

#Estimate liquid fraction
def estimate_liquid_fraction(atoms: Atoms, cutoff=1.2):
    #Use the number of neighbors to estimate the liquid fraction.
    #Define liquid atoms as those with fewer neighbors than the solid coordination number.

    # Define the cutoff for each atom
    cutoffs = [cutoff / 2] * len(atoms)
    
    #Create a neighbor list
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    
    liquid_like_count = 0
    coordination_threshold = 4  #Assume the coordination number in solid B2 structure is 4
    
    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        num_neighbors = len(indices)
        if num_neighbors < coordination_threshold:
            liquid_like_count += 1
    
    fraction_liquid = liquid_like_count / len(atoms)
    return fraction_liquid

#Main function: Two-phase coexistence method simulation
def run_coexistence_md():
    #1) File check
    solid_file = r"D:\\Huang\\学习资料\\科研\\Princeton\\FeO_B2.cif"     #Change to your file path
    liquid_file = r"D:\\Huang\\学习资料\\科研\\Princeton\\FeO_liquid.xyz"#Change to your file path
    if not os.path.exists(solid_file):
        print(f"[ERROR] Solid file not found: {solid_file}")
        return
    if not os.path.exists(liquid_file):
        print(f"[ERROR] Liquid file not found: {liquid_file}")
        return

    #2) Build the solid-liquid coexistence structure
    atoms_coexist = build_solid_liquid_coexistence(solid_file, liquid_file)
    write("FeO_coexist_initial.xyz", atoms_coexist)

    #3) Initialize the calculator
    try:
        calc = MatterSimCalculator()
    except Exception as e:
        print(f"[ERROR] 计算器初始化失败: {e}")
        return
    atoms_coexist.calc = calc

    #4) Set initial temperature, pressure, etc.
    guess_temperature = 6000.0  # Initial guess for temperature，MgO 6000，FeO 9500
    target_pressure = 1000.0 * units.GPa

    #Create NPT dynamics
    npt_dyn = NPT(
        atoms_coexist,
        timestep=1.0 * units.fs,
        temperature_K=guess_temperature,
        externalstress=target_pressure
    )

    total_steps = 20000
    report_interval = 100
    found_Tm = None

    #5) MD main loop
    for step in range(1, total_steps + 1):
        npt_dyn.run(1)  #Run 1 step at a time

        #Estimate the liquid fraction and adjust the temperature at intervals
        if step % report_interval == 0:
            fraction_liq = estimate_liquid_fraction(atoms_coexist, cutoff=1.2)
            print(f"[STEP {step}] T = {guess_temperature}, Liquid fraction ~ {fraction_liq:.2f}")

            if fraction_liq > 0.8:
                guess_temperature -= 100
                print(f"  --> Liquid fraction too high, reducing temperature to {guess_temperature}")
            elif fraction_liq < 0.2:
                guess_temperature += 100
                print(f"  --> Liquid fraction too low, increasing temperature to {guess_temperature}")
            else:
                found_Tm = guess_temperature
                print(f"  --> Liquid fraction is moderate, estimated melting point is {found_Tm} K, stopping MD.")
                break

            #Adjust the target temperature of the dynamics object
            npt_dyn.temperature_K = guess_temperature

    write("FeO_coexist_afterMD.xyz", atoms_coexist)
    if found_Tm is not None:
        print(f"[INFO] MD completed, melting point Tm = {found_Tm} K")
    else:
        print("[INFO] MD completed, no suitable liquid fraction found, consider modifying the search strategy and retry.")

if __name__ == "__main__":
    run_coexistence_md()
