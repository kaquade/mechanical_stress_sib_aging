# Operando CT

Inward (toward the core) and outward (toward the can) radial expansion of the
jelly roll in the 77 operando CT cross-sections, and its dependence on the
cell voltage.

```bash
python operando_ct.py --input <dataset>/operando_ct --output results
```

## Output

- `expansion_per_scan.csv`: per scan the cycle, time, cell voltage at start and
  end of the scan and its mean, mean current, core and jelly-roll radius, and
  inward, outward and total expansion relative to the first scan
- `expansion_vs_voltage.csv`: mean and standard deviation of the expansion
  over the cycles against cell voltage, for charge and discharge
