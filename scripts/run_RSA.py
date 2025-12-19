# %%
from lstnn.generate_fMRI_rdms import generate_group_rdms
from lstnn.compare_rdms import compare_models_to_fmri
from run_stats import run_statistics

method_fmri = "euclidean"  #"crossnobis"
method_ann = "euclidean"  # "euclidean"
compare_method = "corr"
epoch = 4000
atlas = "Glasser"
group = "group"
overwrite = True

# Generate the fMRI rdms
run = True
if run:
    out = (
        f"/home/lukeh/projects/LSTNN/data/fMRI/rdms/{group}_atlas-{atlas}"
        f"/method_{method_fmri}"
    )
    generate_group_rdms(method_fmri, out, atlas, overwrite)

# Main model comparison
run = True
pe_desc = "2dpe"
n_perms = 10
if run:
    out = (
        f"/home/lukeh/projects/LSTNN/results/model_comparison/{group}_"
        f"atlas-{atlas}/pe-{pe_desc}_fmethod-{method_fmri}_"
        f"amethod-{method_ann}_cmethod-{compare_method}_epoch-{epoch}_"
        f"nperms-{n_perms}.csv"
    )
    compare_models_to_fmri(pe_desc, method_fmri, method_ann,
                           compare_method, epoch, n_perms, group,
                           out, atlas, overwrite)
# statistics
run = True
if run:
    print("Running stats...")
    out = (
        f"/home/lukeh/projects/LSTNN/results/model_comparison/{group}_"
        f"atlas-{atlas}/pe-{pe_desc}_fmethod-{method_fmri}_"
        f"amethod-{method_ann}_cmethod-{compare_method}_epoch-{epoch}_"
        f"nperms-{n_perms}_stats.csv"
    )
    run_statistics(pe_desc, group, method_fmri, method_ann,
                   compare_method, epoch, n_perms, out,
                   atlas, overwrite)

# PE models
run = False
n_perms = 10
pe_descs = [
    "learn-0.01", "learn-0.05", "learn-0.1", "learn-0.2", "learn-0.3",
    "learn-0.4", "learn-0.5", "learn-0.6", "learn-0.7", "learn-0.8",
    "learn-0.9", "learn-1.0", "learn-1.1", "learn-1.2", "learn-1.3",
    "learn-1.4", "learn-1.5", "learn-1.6", "learn-1.7", "learn-1.8",
    "learn-1.9", "learn-2.0", "learn-3.0"
]

if run:
    for pe_desc in pe_descs:
        out = (
            f"/home/lukeh/projects/LSTNN/results/model_comparison/{group}_"
            f"atlas-{atlas}/pe-{pe_desc}_fmethod-{method_fmri}_"
            f"amethod-{method_ann}_cmethod-{compare_method}_epoch-{epoch}_"
            f"nperms-{n_perms}.csv"
        )
        print(out)
        compare_models_to_fmri(pe_desc, method_fmri, method_ann,
                               compare_method, epoch, n_perms, group,
                               out, atlas, overwrite)

# statistics
run = False
if run:
    print("Running stats...")
    for pe_desc in pe_descs:
        out = (
            f"/home/lukeh/projects/LSTNN/results/model_comparison/{group}_"
            f"atlas-{atlas}/pe-{pe_desc}_fmethod-{method_fmri}_"
            f"amethod-{method_ann}_cmethod-{compare_method}_epoch-{epoch}_"
            f"nperms-{n_perms}_stats.csv"
        )
        run_statistics(pe_desc, group, method_fmri, method_ann,
                       compare_method, epoch, n_perms, out,
                       atlas, overwrite)

# Big perms for largest and smallest PE
run = False
n_perms = 10000
pe_descs = ["learn-0.1", "learn-0.2", "learn-2.0", "learn-3.0"]

if run:
    for pe_desc in pe_descs:
        out = (
            f"/home/lukeh/projects/LSTNN/results/model_comparison/{group}_"
            f"atlas-{atlas}/pe-{pe_desc}_fmethod-{method_fmri}_"
            f"amethod-{method_ann}_cmethod-{compare_method}_epoch-{epoch}_"
            f"nperms-{n_perms}.csv"
        )
        print(out)
        compare_models_to_fmri(pe_desc, method_fmri, method_ann,
                               compare_method, epoch, n_perms, group,
                               out, atlas, overwrite)

# statistics
run = False
if run:
    print("Running stats...")
    for pe_desc in pe_descs:
        out = (
            f"/home/lukeh/projects/LSTNN/results/model_comparison/{group}_"
            f"atlas-{atlas}/pe-{pe_desc}_fmethod-{method_fmri}_"
            f"amethod-{method_ann}_cmethod-{compare_method}_epoch-{epoch}_"
            f"nperms-{n_perms}_stats.csv"
        )
        run_statistics(pe_desc, group, method_fmri, method_ann,
                       compare_method, epoch, n_perms, out,
                       atlas, overwrite)
