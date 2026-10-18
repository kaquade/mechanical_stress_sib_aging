# Dilatometry

Thickness change of the full electrode stack, the anode and the cathode (full
stack minus anode) against the full-cell voltage, and the cumulative thickness
change traversed by every aged cell.

```bash
python dilatometry.py --data <dataset> --output results
```

## Output

- `thickness_change.csv`: smoothed thickness change of full stack, anode and
  cathode and its derivative dΔL/dV during discharge
- `cumulative_thickness_change.csv`: per cell the EFCs, the cumulative thickness
  change of full stack, anode and cathode, and the full-stack value per EFC
