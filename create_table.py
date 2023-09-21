from pathlib import Path
import sys
import os
import glob

import numpy as np

# old labels
#par_labs = ["Tracer(s)", r"$N(z)$", "Galaxy number", r"$N_{\rm mesh}$", r"$R_{\rm sm}$", r"$\sigma_\perp$", r"$\sigma_{||}$", r"$\sigma^{\rm rec}_\perp$", r"$\sigma^{\rm rec}_{||}$", r"$\sigma^{\rm rec, RSD}_\perp$", r"$\sigma^{\rm rec, RSD}_{||}$", r"$r_\perp$", r"$r_{||}$", r"$r^{\rm RSD}_\perp$", r"$r^{\rm RSD}_{||}$"]

# new labels
par_labs = ["Test name", "Tracer(s)", r"$N(z)$", "Area", r"$\frac{\sigma_z}{1+z}$", r"$N_{\rm mesh}$", r"$R_{\rm sm}$", r"$r_\perp$", r"$r_{||}$", r"$r^{\rm RSD}_\perp$", r"$r^{\rm RSD}_{||}$"]

# find files with glob (maybe sort in order of them being written out?)
fns_save = sorted(glob.glob("table/*npz"))

lines = []
for i in range(len(fns_save)):
    fn_save = fns_save[i]
    data = np.load(fn_save)

    test_name = data['test_name']
    tracer_tab = data['tracer_tab']
    nz_tab = data['nz_tab']
    area_tab = data['area_tab']
    photoz_error = data['photoz_error']
    nmesh = data['nmesh']
    sr = data['sr']
    sigma_perp = data['sigma_perp']
    sigma_perp_err = data['sigma_perp_err']
    sigma_para = data['sigma_para']
    sigma_para_err = data['sigma_para_err']
    sigma_rec_perp = data['sigma_rec_perp']
    sigma_rec_perp_err = data['sigma_rec_perp_err']
    sigma_rec_para = data['sigma_rec_para']
    sigma_rec_para_err = data['sigma_rec_para_err']
    sigma_rec_perp_rsd = data['sigma_rec_perp_rsd']
    sigma_rec_perp_rsd_err = data['sigma_rec_perp_rsd_err']
    sigma_rec_para_rsd = data['sigma_rec_para_rsd']
    sigma_rec_para_rsd_err = data['sigma_rec_para_rsd_err']
    r_perp = data['r_perp']
    r_perp_err = data['r_perp_err']
    r_para = data['r_para']
    r_para_err = data['r_para_err']
    r_perp_rsd = data['r_perp_rsd']
    r_perp_rsd_err = data['r_perp_rsd_err']
    r_para_rsd = data['r_para_rsd']
    r_para_rsd_err = data['r_para_rsd_err']

    print(test_name)
    if "Small" in str(test_name) and "area" in str(test_name) and "20" in str(test_name):
        test_name = "Small area" # drop 20
        area_tab = "392 deg$^2$"
    if "High $n$" in str(test_name):
        test_name = r"High $\bar n$"
        
    par_vals = []
    par_vals.append(f"{test_name}")
    par_vals.append(f"{tracer_tab}")
    par_vals.append(f"{nz_tab}")
    par_vals.append(f"{area_tab}")
    par_vals.append(rf"${photoz_error:.1f}$")
    par_vals.append(rf"${nmesh:d}$")
    par_vals.append(rf"${sr:.1f}$")
    #par_vals.append(rf"${sigma_perp:.1f} \pm {sigma_perp_err:.1f}$")
    #par_vals.append(rf"${sigma_para:.1f} \pm {sigma_para_err:.1f}$")
    #par_vals.append(rf"${sigma_rec_perp:.1f} \pm {sigma_rec_perp_err:.1f}$")
    #par_vals.append(rf"${sigma_rec_para:.1f} \pm {sigma_rec_para_err:.1f}$")
    #par_vals.append(rf"${sigma_rec_perp_rsd:.1f} \pm {sigma_rec_perp_rsd_err:.1f}$")
    #par_vals.append(rf"${sigma_rec_para_rsd:.1f} \pm {sigma_rec_para_rsd_err:.1f}$")
    #par_vals.append(rf"${r_perp:.2f} \pm {r_perp_err:.3f}$")
    #par_vals.append(rf"${r_para:.2f} \pm {np.std(r_3d[:, 2]):.3f}$")
    #par_vals.append(rf"${r_perp_rsd:.2f} \pm {r_perp_rsd:.3f}$")
    #par_vals.append(rf"${r_para_rsd:.2f} \pm {r_para_rsd_err:.3f}$")
    
    par_vals.append(rf"\cellcolor{{red!{.5*r_perp*100.:.2f}}} ${r_perp:.2f} \pm {r_perp_err:.3f}$")
    par_vals.append(rf"\cellcolor{{blue!{.5*r_para*100.:.2f}}} ${r_para:.2f} \pm {r_para_err:.3f}$")
    par_vals.append(rf"\cellcolor{{red!{.5*r_perp_rsd*100.:.2f}}} ${r_perp_rsd:.2f} \pm {r_perp_rsd_err:.3f}$")
    par_vals.append(rf"\cellcolor{{blue!{.5*r_para_rsd*100.:.2f}}} ${r_para_rsd:.2f} \pm {r_para_rsd_err:.3f}$")

    col_str = ' '.join(np.repeat("c", len(par_labs)))
    par_str = ' & '.join(par_labs)
    line = ' & '.join(par_vals)
    line += " \\\\ [0.5ex] \n"
    if "r01_" in fn_save:
        lines.append(" \hline \n")
    lines.append(line)

fn = "table/table.tex"
f = open(fn, "w")
f.write(f"\\begin{{table*}} \n")
f.write(f"\\begin{{center}} \n")
#f.write("\footnotesize \n")
f.write(f"\\begin{{tabular}}{{ {col_str} }} \n")
f.write(" \hline\hline \n")
f.write(f" {par_str} \\\\ [0.5ex] \n")
f.write(" \hline \n")
for line in lines:
    f.write(line)
f.write(" \hline \n")
f.write(" \hline \n")
f.write("\end{tabular} \n")
f.write("\end{center} \n")
f.write("\label{tab:r_coeff} \n")
f.write("\caption{Values of the correlation coefficient between the true and reconstructed velocities in the parallel ($r_{||}$) and perpendicular ($r_\perp$) directions for idealistic (Section \\ref{sec:ideal}) and realistic scenarios (Section \\ref{sec:real}). We separate these two groups of tests by a horizontal line. Additionally, we consider the effect of switching on and off RSD effects. Of most relevance to kSZ analyses using observations is the $r_{||}^{\\rm rsd}$ column, as the signal-to-noise ratio is proportional to it. We discuss the results in the relevant sections.} \n")
f.write("\end{table*} \n")
f.close()
