# SiliconForge

Project repository: https://github.com/Waleed99i/chipathon-2026-systolic-ip-core/tree/a34-rtl-fix-pd

SiliconForge is a parameterized systolic AI accelerator IP-core design developed for the IEEE SSCS PICO Open-Source Chipathon and presented as part of the Code-a-Chip at ISSCC 2027. The project targets a 4×4 processing-element array in GF180MCU and focuses on efficient matrix-multiply acceleration using a structured dataflow architecture, approximate arithmetic, and reusable RTL blocks.

## Project overview

This design organizes multiply-accumulate operations into a regular two-dimensional array, allowing operands to propagate between neighboring processing elements while partial sums remain local. The notebook documents the architecture, design goals, pin assignment, RTL organization, simulation context, and functional-check summaries for the team’s submission.

The implementation is intended as an open-source reference design for digital accelerator exploration and can be used to understand:

- dataflow-oriented accelerator organization,
- PE-level arithmetic and accumulation,
- input/output packetization,
- functional simulation and design review documentation.

## Team members

| Team member | Affiliation | Contribution |
|---|---|---|
| Muhammad Waleed Akram | University of Engineering and Technology, Lahore | Team lead; RTL, verification, interfaces and chip wrapper |
| Abdul Muiz | University of Engineering and Technology, Lahore | Systolic top RTL, verification and optimization |
| Rumali Siddiqua | University of Virginia, USA | Physical design and designated travel representative |
| Nur Cahyo Ihsan Prastyawan | Universitas Gadjah Mada, Indonesia | MAC RTL, verification and FPGA synthesis |

## Designated travel representative

Rumali Siddiqua  
University of Virginia, USA

## Project contact

Muhammad Waleed Akram  
Email: waleedakram059@gmail.com

or Alternatively:

rumalisiddiqua@gmail.com


## Notebook

Notebook file: `SiliconForge_Code_a_Chip.ipynb`

The notebook contains the main design narrative, architecture summary, pin assignment, block-level organization, and simulation/verification figures used for review.

## Dependencies

To view and run the notebook locally, install the following:

- Python 3.10+
- Jupyter Notebook or JupyterLab
- IPython / ipykernel
- NumPy
- Matplotlib
- Pandas

Typical installation:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install jupyterlab notebook numpy matplotlib pandas ipykernel
```

## How to view the notebook

From the repository root:

```bash
jupyter notebook ISSCC27/submitted_notebooks/SiliconForge/SiliconForge_Code_a_Chip.ipynb
```

Alternatively, open the notebook directly in VS Code with the Jupyter extension enabled.

## How to reproduce the notebook

1. Open the notebook in Jupyter or VS Code.
2. Ensure the required Python packages are installed.
3. Run the notebook cells in order from top to bottom.
4. Review the architecture summary, pin assignment, and generated figures.
5. If needed, modify the design parameters and rerun the relevant cells to explore the parameterized accelerator behavior.

## Notes

This README is intended to help reviewers and collaborators understand the project quickly. The notebook is the primary source of technical detail, while the repository context provides the supporting design artifacts and documentation.
