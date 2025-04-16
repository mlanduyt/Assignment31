# Download FEniCSx

Create a conda environment -- 

See: https://fenicsproject.org/download/

On the SCC:

```bash
module load miniconda
mamba create create -n fenicsx-env
mamba activate fenicsx-env
mamba install -c conda-forge fenics-dolfinx mpich pyvista
pip install imageio
pip install gmsh
```

Note: this might take a couple of minutes. 

Then, you can launch a VSCode server and choose fenicsx-env a your conda environment.

Some features intended for the example do not work as intended such as the visualization animation. Currently, the results are compared to an analytical solution and the error is reported. 