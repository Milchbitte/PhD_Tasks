from ase.md.langevin import Langevin
from ase import units
from mattersim.forcefield.potential import MatterSimCalculator
from ase.io import read, write

def run_feo_liquid_generation():

    #Step 1: Load the FeO B2 structure (similar for MgO_B2.cif)
    print("Step 1: Loading FeO B2 structure...")
    b2_structure_file = r"D:\\Huang\\学习资料\\科研\\Princeton\\FeO_B2.cif"  #Change to your file path
    feo_b2 = read(b2_structure_file)

    #Step 2: Initialize the MatterSim calculator
    print("Step 2: Initializing the MatterSim calculator...")
    calc = MatterSimCalculator.from_checkpoint("mattersim-v1.0.0-1M.pth")
    feo_b2.calc = calc  # Bind the calculator to the FeO B2 structure

    #Step 3: Generate liquid FeO configuration via MD simulation
    print("Step 3: Generating liquid FeO configuration through MD simulation...")
    dyn = Langevin(feo_b2, timestep=1.0 * units.fs, temperature_K=3000, friction=0.01)  #Adjust timestep and temperature
    dyn.run(steps=20000)  #Increase steps to ensure complete melting

    #Save the liquid structure
    atoms_liquid = dyn.atoms
    liquid_file_path = r"D:\\Huang\\学习资料\\科研\\Princeton\\FeO_liquid.xyz"  #Change to your file path
    write(liquid_file_path, atoms_liquid)
    print(f"Liquid FeO configuration saved as {liquid_file_path}")

if __name__ == "__main__":
    run_feo_liquid_generation()
