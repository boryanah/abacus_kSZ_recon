# Installation

My recommendation is simply to load the DESI environment:

`source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main`

Alternatively, you can install `pyrecon` from here, but there are quite a few dependencies.

`pip install git+https://github.com/cosmodesi/pyrecon`

# Scripts

- `analyze_reconstruction.py`: Output the _r_ coefficient between true and reconstructed velocity
- `correlate_catalog.py`: Compute the correlation function or power spectrum
- `join_lc_catalog.py`: Combine light cone mocks across different redshifts
- `prepare_lc_catalog.py`: Read light cone mocks and output the necessary fields
- `reconstruct_box_catalog.py`: Apply reconstruction to a cubic box mock
- `reconstruct_lc_catalog.py`: Apply reconstruction to a light cone mock
- `visualize_correlation.py`: Plot the correlation function
- `visualize_power.py`: Plot the power spectrum


# Running on an interactive node

You can start an interactive job on Perlmutter with:

`salloc --nodes 1 --qos interactive --time 04:00:00 --constrain cpu`

On Cori, `cpu` needs to be replaced by `haswell`.

After loading the environment, you can then just run the `python` command:

`python reconstruct_box_catalog.py`

When running MPI code (e.g., `correlate_catalog.py`), you also need to include `srun -np N python ...`, where `N` is some reasonable number given the architecture of the cluster (Note that Perlmutter has 128 physical threads, whereas Cori has 32).
