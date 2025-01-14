import numpy as np
from ase import units
from ase.io import read
from ase.md.npt import NPT

#MatterSim & Phonon
from mattersim.forcefield.potential import MatterSimCalculator
from mattersim.applications.phonon import PhononWorkflow
from mattersim.utils.supercell_utils import get_supercell_parameters

#使用 PhononWorkflow 计算固态 B2 结构的声子熵
def compute_solid_entropy_phonon(cif_file: str, T_target: float):

    print(f"[INFO] 计算固态熵: {cif_file}, T={T_target}K")

    #1. 读取固态 B2 结构
    atoms_solid = read(cif_file)
    #2. 附上 MatterSim 计算器
    calc = MatterSimCalculator()
    atoms_solid.calc = calc

    #3. 生成超胞 & q点
    nrep, qpoints = get_supercell_parameters(atoms_solid, max_atoms=400)

    #4. 创建并运行 PhononWorkflow
    phonon_workflow = PhononWorkflow(
        atoms=atoms_solid,
        qpoints_mesh=qpoints,
        amplitude=0.01
    )
    has_imag, phonon = phonon_workflow.run()
    if has_imag:
        print("[WARNING] 声子中出现虚频, 可能在该压力下不稳定!")

    #5. 计算热力学量
    phonon.run_thermal_properties(t_min=0, t_max=T_target, t_step=50)
    tp_dict = phonon.get_thermal_properties_dict()

    #6. 找到最接近 T_target 处的熵
    temps = np.array(tp_dict["temperature"])
    entropies = np.array(tp_dict["entropy"])  # 通常 eV/K per cell 或 kB/atom，需确认
    idx = np.argmin(np.abs(temps - T_target))
    s_solid = entropies[idx]

    print(f"[INFO] 在 T={T_target} K 下的固态熵 s_solid = {s_solid:.6f} (注意单位!)")
    return s_solid

#简化公式 S_liq ~ -G_liq / T 近似液态熵
def compute_liquid_entropy_simple(liquid_file: str, T_target: float, pressure_pa=1e11):

    print(f"[INFO] 计算液态熵(简化): {liquid_file}, T={T_target}K")

    #1. 读取液态构型
    atoms_liquid = read(liquid_file)
    #2. 设置 MatterSim 计算器
    calc = MatterSimCalculator()
    atoms_liquid.calc = calc

    #3. 在 T_target, pressure_pa 下跑NPT，让体系充分平衡
    dyn = NPT(
        atoms_liquid,
        timestep=1.0 * units.fs,
        temperature_K=T_target,
        externalstress=pressure_pa
    )

    #先让它平衡 2000 步
    dyn.run(2000)

    #4. 读取当前体系能量(E_pot)做近似的 G_liq
    g_liquid = atoms_liquid.get_potential_energy()  # eV (?)
    print(f"  [INFO] 近似 G_liquid = {g_liquid:.6f} eV (?)")

    #5. 简化近似: S_liq = - G_liquid / T
    s_liquid = - g_liquid / T_target
    print(f"  [INFO] 近似液态熵 s_liquid = {s_liquid:.6f} (注意单位!)")

    return s_liquid


def main_calculate_melting_entropy():

    T_m = 6000.0  # 假设已经找到的熔点 (K)
    pressure_pa = 1.0e14  # 1000 GPa = 1e14 Pa
    #Solid / Liquid 文件路径 (MgO 或 FeO)
    cif_file_solid = r"D:\Huang\学习资料\科研\Princeton\FeO_B2.cif"
    liquid_file = r"D:\Huang\学习资料\科研\Princeton\FeO_liquid.xyz"

    #1) 计算固态熵
    s_solid = compute_solid_entropy_phonon(cif_file_solid, T_m)

    #2) 计算液态熵 (简化方法)
    s_liquid = compute_liquid_entropy_simple(liquid_file, T_m, pressure_pa=pressure_pa)

    #3) 熔化熵 & 熔化焓
    delta_s_m = s_liquid - s_solid
    delta_h_m = T_m * delta_s_m

    print("\n[RESULT] =======================")
    print(f"  熔化熵 (Delta S_m) = {delta_s_m:.6f}")
    print(f"  熔化焓 (Delta H_m) = {delta_h_m:.6f}")


if __name__ == "__main__":
    main_calculate_melting_entropy()
