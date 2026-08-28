# AeroTwin-4 — Representative Aero Engine Digital Twin

A prototype Digital Twin for a representative 4-cylinder aero piston engine.

## Structure

- `phase1/`: Core engine mathematical modeling & rotational dynamics.
  - `engine/parameters.py`: Centralized parameter definitions.
  - `engine/dynamics.py`: Rotational dynamics equation solver ($J \frac{d\omega}{dt} = T_{engine} - T_{load} - T_{friction}$).
  - `tests/test_dynamics.py`: Unit tests for dynamics stability and unit conversions.
  - `main.py`: Simulation runner script.

## Execution

To run the Phase 1 Step 1 simulation:
```bash
python3 phase1/main.py
```

To run unit tests:
```bash
pytest phase1/tests
```
