# boat-optimization

Python tool for optimizing an equation-driven boat hull design for ENGR 1103.

## Constraints
- Foam block: 11"L × 6"W × 4"H, density 10 lb/ft³
- Ballast: 700–1000 g
- Mast: 0.5 m, 3/8" diameter aluminum
- AVS > 100°, keel-less monohull

## Project Structure
```
main.py                          # CLI entry point
boat_optimization/
├── config.py                    # Constants and design constraints
├── hull.py                      # Hull geometry (cross-section, taper, volume)
├── analysis.py                  # Large-angle stability: COM, COB, waterline, GZ, AVS
├── optimizer.py                 # Scipy differential evolution optimizer
├── visualization.py             # Plots, AVS panels, and STL export
└── mathematica_export.py        # Generate Mathematica notebook code
```

## Usage
```bash
pip install -r requirements.txt

python main.py analyze       # Full analysis + stability plots + AVS panels
python main.py optimize      # Run optimizer, then analyse + export Mathematica
python main.py export        # Export hull as STL for CNC
python main.py mathematica   # Generate Mathematica .nb with equations
```

## Physics
The stability analysis uses proper large-angle computation:
1. Build hull cross-section polygon in body coordinates
2. Rotate by heel angle θ
3. Clip against horizontal waterline (Sutherland-Hodgman)
4. Binary search for equilibrium waterline (displaced volume = weight/ρ)
5. Compute COB from clipped polygon centroid
6. GZ = y_B - y_G in world frame; Righting moment = W × GZ