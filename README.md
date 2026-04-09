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
├── analysis.py                  # Physics: COM, COB, waterline, stability
├── optimizer.py                 # Scipy differential evolution optimizer
└── visualization.py             # Plots and STL export
```

## Usage
```bash
pip install -r requirements.txt

python main.py analyze     # Analyze default hull
python main.py optimize    # Run optimizer
python main.py export      # Export hull as STL
```