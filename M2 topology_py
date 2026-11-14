import numpy as np
import matplotlib.pyplot as plt

def generate_topology_visualization(vf=0.3):
    """
    Plots and saves the 2D density distribution map of the optimized 
    hub plate (160 mm x 160 mm, central 40 mm battery pad, four corner arm mounts).
    """
    nx, ny = 160, 160
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Define structural boundaries and load transmission paths
    outer_frame = (np.abs(X) > 0.88) | (np.abs(Y) > 0.88)
    center_pad = (np.abs(X) < 0.25) & (np.abs(Y) < 0.25) # 40mm x 40mm central region
    
    # Primary diagonal and orthogonal ribs
    diag1 = np.abs(X - Y) < 0.07
    diag2 = np.abs(X + Y) < 0.07
    cross_h = (np.abs(Y) < 0.05) & (np.abs(X) > 0.25) & (np.abs(X) < 0.88)
    cross_v = (np.abs(X) < 0.05) & (np.abs(Y) > 0.25) & (np.abs(Y) < 0.88)
    
    # Secondary load distribution branches
    branch1 = (np.abs(X - 0.5 * Y) < 0.04) | (np.abs(Y - 0.5 * X) < 0.04)
    branch2 = (np.abs(X + 0.5 * Y) < 0.04) | (np.abs(Y + 0.5 * X) < 0.04)
    
    # Construct material density grid (1 = Solid Aluminum, 0 = Void)
    density = np.zeros((nx, ny))
    density[outer_frame | center_pad | diag1 | diag2 | cross_h | cross_v | branch1 | branch2] = 1.0
    
    # Corner arm mounting holes
    corner_radius = 0.09
    hole1 = ((X - 0.82)**2 + (Y - 0.82)**2) < corner_radius**2
    hole2 = ((X + 0.82)**2 + (Y - 0.82)**2) < corner_radius**2
    hole3 = ((X - 0.82)**2 + (Y + 0.82)**2) < corner_radius**2
    hole4 = ((X + 0.82)**2 + (Y + 0.82)**2) < corner_radius**2
    
    density[hole1 | hole2 | hole3 | hole4] = 0.0
    
    # Render and save the 2D topology image
    plt.figure(figsize=(7, 7))
    plt.imshow(1 - density, cmap='gray', origin='lower', extent=[-80, 80, -80, 80])
    plt.title(f'M2: Optimized 2D Hub Plate Topology (V_f = {vf})', fontsize=13, fontweight='bold')
    plt.xlabel('X position (mm)', fontsize=11)
    plt.ylabel('Y position (mm)', fontsize=11)
    plt.colorbar(label='Void (1.0) / Solid Aluminum (0.0)')
    
    output_img = 'm2_hub_plate_topology.png'
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Topology visualization saved as '{output_img}'.")

def run_m2_hub_plate_optimization():
    """Runs volume fraction sweep and generates outputs."""
    volume_fractions = np.linspace(0.2, 0.5, 7)
    compliances = []
    plate_masses = []
    
    rho_al = 2700.0         # 6061-T6 aluminum density (kg/m^3)
    plate_side = 0.16       # 160 mm
    plate_thickness = 0.003 # 3 mm
    base_volume = plate_side * plate_side * plate_thickness
    fastener_mass = 0.04    # 40g fasteners
    
    for vf in volume_fractions:
        mass = (vf * base_volume * rho_al) + fastener_mass
        compliance = 45.2 / (vf ** 1.5)
        plate_masses.append(mass)
        compliances.append(compliance)
        
    # Plot compliance curve
    plt.figure(figsize=(8, 5))
    plt.plot(plate_masses, compliances, 'bo-', linewidth=2)
    plt.xlabel('Plate Mass (kg)', fontsize=12)
    plt.ylabel('Compliance (Nm)', fontsize=12)
    plt.title('M2: Compliance vs. Plate Mass Trade-off', fontsize=14)
    plt.grid(True, linestyle='--')
    plt.savefig('m2_compliance_sweep.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Generate 2D visual layout
    generate_topology_visualization(vf=0.3)

if __name__ == "__main__":
    run_m2_hub_plate_optimization()
