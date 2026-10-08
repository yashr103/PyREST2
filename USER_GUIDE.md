# User Guide

A complete walkthrough of the PyREST2 application: what each page does,
what every setting means, how to choose the simulation parameters, and how to
read the results.

For installation see [README.md](README.md). For the correctness tests see
[benchmarks/README.md](benchmarks/README.md).

**Contents**

1. [What this software does](#1-what-this-software-does)
2. [Workflow overview](#2-workflow-overview)
3. [Step 1 - PDB Preprocessing](#3-step-1---pdb-preprocessing)
4. [Step 2 - Ligand Prep](#4-step-2---ligand-prep)
5. [Step 3 - System Generation](#5-step-3---system-generation)
6. [Step 4 - Output format](#6-step-4---output-format)
7. [Step 5 - Simulation run](#7-step-5---simulation-run)
8. [Choosing the replica ladder](#8-choosing-the-replica-ladder)
9. [Step 6 - Analysis](#9-step-6---analysis)
10. [Reading the results](#10-reading-the-results)
11. [Resuming and extending a run](#11-resuming-and-extending-a-run)
12. [Command-line use](#12-command-line-use)
13. [Validating your installation](#13-validating-your-installation)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. What this software does

The app runs **REST2** (Replica Exchange with Solute Tempering, Wang et al.,
*J. Phys. Chem. B* 2011) on a protein or a protein-ligand complex, using
OpenMM and openmmtools.

In REST2 **every replica runs at the same real temperature** (T min, normally
300 K). What differs between replicas is the *Hamiltonian*: the solute's own
interactions are progressively weakened, which makes the solute behave as if
it were hotter while the water stays at 300 K. Each replica is labelled by the
effective temperature T_eff it mimics, and the scaling factor is

```
lambda = sqrt(T_min / T_eff)
```

| Interaction | Scaled by |
|---|---|
| solute-solute (bonded, charges, LJ) | lambda² |
| solute-water | lambda |
| water-water | 1 (unchanged) |

**State 0 is the unmodified force field.** It is the ensemble you analyse and
report; the hotter replicas exist only to help it cross energy barriers. This
is why REST2 results are unbiased even though the upper replicas are distorted.

The solute (the "hot region") is the **whole protein, plus the ligand** if one
is present. Water and ions are never scaled.

---

## 2. Workflow overview

The left sidebar is a 6-step pipeline. Steps are numbered, show a tick when
complete, and the ligand step is marked *(skipped)* in protein-only mode.

| Step | Produces | Needs AmberTools |
|---|---|---|
| 1. PDB Preprocessing | cleaned `*_fixed.pdb` | no |
| 2. Ligand Prep | `ligand.mol2`, `ligand.frcmod` | **yes** (antechamber, parmchk2) |
| 3. System Generation | `complex.prmtop`, `.inpcrd`, `.pdb` | **yes** (tleap) |
| 4. Output format | OpenMM system settings | no |
| 5. Simulation run | `rest2_remd.nc`, checkpoint, log, metadata | no |
| 6. Analysis | plots + CSVs | no |

**Two routes.** On step 1 set **System type**:

- **Protein–ligand complex** - all six steps.
- **Protein only** - Ligand Prep is disabled and skipped, every ligand field
  disappears, and the ligand is omitted from the tleap script. REST2 still
  heats the whole protein, so this is a valid apo solute-tempering run.

**Entry shortcuts.** Each of steps 1-3 has an *"Already have…"* panel at the
bottom, so you can join the pipeline at any point with files you already have
(a cleaned PDB, prepared ligand files, or a finished `prmtop`/`inpcrd`).

**Saving settings.** `File → Save Settings` (Ctrl+S) writes every field to a
JSON file; `Load Settings` (Ctrl+O) restores them. Use this to keep a record of
exactly how a published run was configured.

---

## 3. Step 1 - PDB Preprocessing

Cleans a crystal structure with PDBFixer.

| Field | Default | Meaning |
|---|---|---|
| **System type** | Protein–ligand complex | Switches the whole app between complex and apo mode |
| **Input PDB** | - | Your starting structure (RCSB file or your own) |
| Remove crystallographic waters | on | Drops `HOH`; tleap adds fresh solvent later |
| Remove heteroatoms / non-standard residues | on | Drops ions, buffer molecules, sugars. **Turn this off if your ligand is inside this PDB** |
| Alternate locations | Highest occupancy | Which altloc to keep |
| Chain selection | all | Comma-separated chain IDs, e.g. `A` or `A,B` |
| Model missing residues/loops | off | Builds unresolved loops. Modelled loops are guesses - check them |
| Add missing hydrogens | on | Adds H at the chosen pH |
| Protonation pH | 7.0 | Used for titratable side chains |
| Auto-detect disulfide bonds | on | Reports candidate S-S pairs (SG-SG within 2.5 Å) |

Press **Run PDB Preprocessing**. The report lists what was changed and any
disulfide candidates found.

> **Note on disulfides:** detection here is informational. The actual bonding
> is done automatically in step 3, which renames the paired cysteines to `CYX`,
> removes their `HG` hydrogens and writes explicit `bond` commands for tleap.

---

## 4. Step 2 - Ligand Prep

Parameterises a small molecule with antechamber and parmchk2. **Skipped in
protein-only mode.** Requires AmberTools.

| Field | Default | Meaning |
|---|---|---|
| **Ligand file** | - | `.mol2`, `.sdf` or `.pdb` of the ligand alone, **with hydrogens and correct bond orders** |
| Ligand force field | gaff2 | GAFF2 is recommended for new work |
| Partial charge method | AM1-BCC | Standard for GAFF. RESP needs external QM; Gasteiger is fast but crude |
| **Net charge** | 0 | Formal charge of the ligand. **A wrong value is the most common failure** |
| Spin multiplicity | 1 | 1 for closed-shell molecules |
| Generate frcmod (parmchk2) | on | Fills in any missing GAFF parameters |

**Execution** fields point at AmberTools. If the app was not launched from an
activated AmberTools environment, set **Conda environment path** (recommended,
e.g. `/home/you/miniforge3/envs/amber-env`); the environment *name* and an
explicit conda executable are also accepted.

After a successful run the generated `.mol2`/`.frcmod` are handed to step 3
automatically. Check the `.frcmod` for lines marked `ATTN, need revision` -
those are guessed parameters.

---

## 5. Step 3 - System Generation

Builds the solvated, neutralised system with tleap, then creates the OpenMM
system.

### Force fields and solvation

| Field | Default | Notes |
|---|---|---|
| Protein force field | leaprc.protein.ff19SB | ff19SB is current; ff14SB and ff99SBildn are offered for compatibility |
| Ligand force field | leaprc.gaff2 | Disabled in protein-only mode |
| Water model | leaprc.water.opc | **OPC is the matched partner for ff19SB.** Use tip3p with ff14SB |
| Box shape | Truncated octahedron | Fewer waters than a cube for the same padding |
| Box padding | 10.0 Å | Minimum solute-to-wall distance |

### Ions

| Field | Default | Notes |
|---|---|---|
| Neutralize system | on | Adds counterions to reach zero net charge |
| Cation / Anion | Na+ / Cl- | |
| Additional ionic strength | 0.15 M | Physiological salt **on top of** neutralisation |

The salt concentration is converted to a **number of ions** automatically
(tleap has no concentration option): the system is solvated once to count the
waters, then `n_ions = conc × n_waters / 55.56`. The status line reports the
result, e.g. *"Salt: 7 ion pairs for 0.15 M among 2469 waters"*.

### Output and execution

`Output file prefix` (default `complex`) names the `.prmtop`, `.inpcrd` and
`.pdb`. The **Protein PDB** field defaults to the fixed PDB from step 1.
`Working directory` is where tleap runs and the files are written.

Press **Generate System (tleap)**. The status line reports atom and residue
counts, any PDB corrections applied, and the salt added.

**Automatic PDB corrections.** Before tleap runs, the app writes a corrected
copy (`<name>_tleap.pdb`) fixing two things Amber's templates reject:

- the N-terminal backbone hydrogen named `H` (Amber wants `H1`/`H2`/`H3`);
- disulfide cysteines, renamed to `CYX` with their `HG` removed and explicit
  `bond` commands added.

Your original PDB is never modified.

---

## 6. Step 4 - Output format

OpenMM system settings. These apply to **equilibration and production alike**.

| Field | Default | Notes |
|---|---|---|
| Nonbonded method | PME | Production needs a periodic method: PME, Ewald or CutoffPeriodic |
| Nonbonded cutoff | 1.0 nm | |
| Ewald error tolerance | 0.0005 | |
| Constraints | HBonds | Required for a 2 fs or larger timestep |
| Rigid water | on | |
| Enable hydrogen mass repartitioning | on | Moves mass from heavy atoms onto hydrogens so longer timesteps stay stable |
| Hydrogen mass | 1.5 amu | **3.0 amu is the standard choice** and allows 3-4 fs |

**Timestep and hydrogen mass go together:**

| Hydrogen mass | Safe timestep | Notes |
|---|---|---|
| 1.0 amu (HMR off) | 1-2 fs | |
| 1.5 amu | 2 fs | Small margin; a source of NaN crashes |
| **3.0 amu** | **2-4 fs** | Recommended; 3 fs is a good balance |
| 4.0 amu | 3-4 fs | Non-standard; methyl carbons become very light |

Heavier hydrogens do not change equilibrium results (distributions, free
energies) - only the speed of hydrogen-related motion.

---

## 7. Step 5 - Simulation run

Runs EM → NPT → NVT equilibration, then REST2 production, as a separate
process whose log is streamed into the window.

### Energy Minimization

| Field | Default |
|---|---|
| Max iterations | 5000 (0 = until converged) |
| Energy tolerance | 10 kJ/mol/nm |

### NVT Equilibration

| Field | Default | Notes |
|---|---|---|
| Number of steps | 500000 | 1 ns at 2 fs |
| Target temperature | 300 K | Should equal T min; a mismatch is warned about |
| **Timestep** | 2.0 fs | **This timestep is used for production too** |
| Thermostat | Langevin middle integrator | Nose-Hoover and Andersen also available; NVT stage only |
| Apply positional restraints on heavy atoms | on | Holds the solute while solvent relaxes |

### NPT Equilibration

| Field | Default | Notes |
|---|---|---|
| Number of steps | 500000 | Sets the box density |
| Target pressure | 1.0 atm | |
| Barostat | Monte Carlo barostat | The membrane option is not meaningful for these systems |
| Barostat update interval | 25 steps | |

Production itself is **NVT** - the box is fixed after equilibration.

### REMD Production

| Field | Default | Meaning |
|---|---|---|
| **T min (real)** | 300 K | The real temperature of every replica |
| **T max (effective)** | 400 K | Effective temperature of the hottest replica. Not a physical temperature - see §8 |
| Number of replicas | 16 | |
| Temperature distribution | Exponential spacing | Exponential, Linear, or Custom list |
| Custom temperatures | - | Shown only for "Custom list": comma-separated, ascending, first value = T min |
| **Exchange attempt interval** | 1000 steps | MD steps between exchange attempts (2 ps at 2 fs) |
| **Total production steps** | 50000000 | Per replica. Iterations = total ÷ exchange interval |
| Exchange scheme | Neighbor swap (Metropolis) | Gibbs sampling attempts `n_replicas³` swaps per iteration - better mixing, more overhead |
| Ligand residue name(s) | MOL | Hidden in protein-only mode. **Detect…** reads it from the topology |
| Friction coefficient | 1.0 /ps | Langevin collision rate |
| **Checkpoint interval** | 200 cycles | How often coordinates are saved - see the frame-spacing formula below |
| Platform | CUDA | CUDA, OpenCL, HIP or CPU |
| Output directory | ./rest2_output | |
| Resume from previous run | off | See §11 |

### Two formulas you will need

**Simulated time**

```
total time per replica = total production steps × timestep
```
e.g. 50,000,000 × 2 fs = 100 ns.

**Frame spacing in the saved trajectory**

```
frame spacing = checkpoint interval × exchange attempt interval × timestep
```
e.g. 200 × 1000 × 2 fs = **0.4 ns per frame**.

Checkpoints store the coordinates of *every replica*, so they dominate disk
use (roughly 4-8 MB per checkpoint for a 50,000-atom system with 16 replicas).

| Wanted frame spacing | Checkpoint interval (1000 steps × 2 fs) |
|---|---|
| 0.4 ns | 200 |
| 0.2 ns | 100 |
| 0.1 ns | 50 |
| 0.05 ns | 25 |

The default of 200 gives few frames and is the most common cause of sparse,
unusable structural plots. **For analysis-quality output use 0.1-0.2 ns.**

### While it runs

The log streams into the window and is written to `rest2_remd.log`. The
progress bar follows lines of the form:

```
Progress: 8000/50000 iterations (16.0%) - elapsed 825.7 min, ETA 4334.8 min
```

If openmmtools hits a NaN it restarts that step automatically (up to 6 times)
and the count is appended to the progress line as `NaN restarts so far: N`.

---

## 8. Choosing the replica ladder

This is the part that decides whether REST2 actually helps.

### T max is an effective temperature

Nothing is physically heated above T min. T max only sets how far the solute's
interactions are weakened:

| T max | lambda | Solute-solute scaling | Verdict |
|---|---|---|---|
| 330 K | 0.953 | 91% | Negligible - barely differs from plain MD |
| 400 K | 0.866 | 75% | Weak |
| **500-600 K** | 0.775-0.707 | 60-50% | **Typical working range** |
| 700 K | 0.655 | 43% | Strong; needs many replicas or a small solute |

For reference, the OpenMM Cookbook REST tutorial uses **300-600 K with 12
replicas** for villin, a 35-residue protein.

### Acceptance tells you if the spacing is right

After a short test run, open `01_acceptance_rates.png`:

| Acceptance | Meaning | Action |
|---|---|---|
| **20-40%** | Ideal | Keep |
| > 50% | Replicas too close together | Raise T max, or use fewer replicas |
| < 15% | Replicas too far apart | Add replicas, or lower T max |
| ~100% | States are not actually different | Something is wrong - see §13 |

Acceptance should be **roughly even across all pairs**. A steady decline from
one end to the other (e.g. 52% → 16%) means the spacing is uneven: with
**Exponential spacing** the gaps widen towards the hot end, so switch to
**Linear spacing** or add replicas at the top.

### How many replicas

The number needed grows with the size of the hot region and the width of the
ladder. A practical recipe:

1. Run 1-2 ns with your intended ladder.
2. Read the acceptance rates.
3. Scale: to roughly halve acceptance, double the spacing in lambda.

### How long to run

REST2 needs **many round trips** across the ladder before the hot replicas
influence state 0.

| Run length | What it can tell you |
|---|---|
| 1-3 ns | The pipeline works. **Nothing scientific** |
| 20-50 ns | Side-chain and ligand-pose sampling |
| 100 ns+ | Backbone and loop rearrangements |

**Energies separate within picoseconds; structures take nanoseconds to
microseconds.** Seeing well-separated energy distributions but identical
RMSD/Rg across states after a short run is expected, not a bug.

---

## 9. Step 6 - Analysis

When you start a run, this page is filled in automatically from the production
settings. The run also writes `run_metadata.json`, and the analysis reads it so
the ladder and timing always match the actual run - values here that disagree
are overridden, with a warning in the log.

| Field | Default | Notes |
|---|---|---|
| Production directory | rest2_output | Folder containing the `.nc` files |
| Topology (prmtop) | from step 3 | |
| Reference structure (PDB) | tleap's `complex.pdb` | RMSD is measured against this |
| NetCDF file / checkpoint | auto-detect | Override only for unusual layouts |
| Number of replicas, T min, T max | from the run | Overridden by `run_metadata.json` |
| Ligand selection | `resname MOL` | MDTraj syntax; hidden in protein-only mode |
| Receptor selection | `protein` | MDTraj syntax |
| Timestep, steps per iteration, checkpoint interval | from the run | **Must match the run** or frames map to the wrong replicas |
| Frame stride | 1 | Use every Nth saved frame |
| Skip structural analysis | off | Diagnostics only |
| Apply PBC image_molecules() | off | Enable if molecules look split across the box |
| Skip energy decomposition | off | This is the slowest step |
| Energy decomposition stride | 10 | Extra thinning for the energy step; use 1-2 for short runs |
| FES outlier cutoff (robust z) | 5.0 | Removes isolated stray frames from the free-energy surface; 0 disables |

---

## 10. Reading the results

### Exchange diagnostics

| File | What to look for |
|---|---|
| `01_acceptance_rates.png` | 20-40%, roughly even across pairs (§8) |
| `02_replica_walk.png` | Each replica should travel the full ladder repeatedly. Replicas stuck at one end mean poor mixing |
| `03_energy_distributions.png` | Neighbouring states must overlap. Gaps mean exchanges cannot happen |

### Structure (state 0 unless stated)

| File | Contents |
|---|---|
| `04_rmsd_time.png` | Backbone RMSD, plus ligand-pose and site-aligned protein RMSD |
| `05_rg_time.png` | Radius of gyration |
| `06_ligand_com_distance.png` | Ligand-receptor centre-of-mass distance (protein-ligand only) |
| `07_fes_rmsd_rg.png` | Free-energy surface over RMSD and Rg |
| `08_rmsf.png` | Per-residue Cα RMSF |
| **`09_state_comparison.png`** | **Mean RMSD / Rg / ligand RMSD vs effective temperature, all states.** Flat curves mean the ladder is doing nothing |
| **`10_rmsf_by_state.png`** | Per-residue RMSF for every state |

Plots 09 and 10 are the ones that answer *"are the hot replicas actually
sampling differently?"* - the others show state 0 only.

### Energy decomposition

| File | Contents |
|---|---|
| `11_e_solute_distributions.png` | Solute intramolecular energy per state. **Should separate clearly across the ladder** |
| `12_e_solute_water_distributions.png` | Solute-water interaction energy |
| `13_energy_component_scatter.png` | Overlap of the two components |
| `14_scaled_e_solute_water_distributions.png` | The scaled solute-water term used in the REST acceptance test |
| `15_rest_combination_distributions.png` | The full combination appearing in the exchange criterion |
| `16_protein_ligand_energy_split.png` | Protein-only and ligand-only energies (protein-ligand runs) |

### Data files

| File | Contents |
|---|---|
| `rmsd_state_NN.csv` | Per-frame metrics for each state |
| `all_replica_rmsd.csv` | All states, long format |
| `summary_statistics.csv` | Per-state means - the quickest check of whether states differ |
| `energy_decomposition.csv` | Per-frame energy components |
| `energy_decomposition_outliers.csv` | Frames flagged as outliers (written only if any) |

### A quick sanity routine

1. `01` - acceptance in range and even?
2. `02` - replicas traversing the ladder?
3. `11` - do the solute energies separate across states? If yes, REST2 is
   working, whatever the structural plots show.
4. `09` - do structural metrics change with effective temperature? If flat,
   the ladder is too narrow or the run too short (§8).
5. `summary_statistics.csv` - confirm the numbers behind 09.

---

## 11. Resuming and extending a run

Tick **Resume from previous run (restart)** and press Start, using the **same
output directory**.

- Resume continues from the **last checkpoint**, so up to one checkpoint
  interval of work is repeated.
- **The stored run wins.** The ladder, replica count, timestep, exchange
  interval and checkpoint interval all come from the saved file; values on the
  page that differ are reported in the log and ignored. To change any of
  them you must start a **new run in a new directory**.
- **To extend a run**, keep everything else identical and only increase
  *Total production steps*.
- After restarting the app, reload the system first (step 3 → *"Use These
  Files Directly"*) so the prmtop/inpcrd paths are known again.

---

## 12. Command-line use

The production and analysis stages run headless, which is what you want on a
cluster.

```bash
python simulation_run.py --prmtop complex.prmtop --inpcrd complex.inpcrd \
    --protein-only --n-replicas 16 --t-min 300 --t-max 600 \
    --temp-distribution "Linear spacing" --timestep 2.0 --hydrogen-mass 3.0 \
    --n-iterations 50000 --n-steps-per-iter 1000 --checkpoint-interval 50 \
    --output-dir rest2_output --platform CUDA
```

For a complex, replace `--protein-only` with `--ligand-resnames MOL`.

```bash
python analysis.py --dir rest2_output --top complex.prmtop --ref complex.pdb \
    --out analysis_output --energy-decomposition-stride 2
```

Both accept `--help` for the full list. `analysis.py` reads
`run_metadata.json` from the production directory, so the ladder and timing
flags are usually unnecessary for runs made with this version.

Two standalone utilities:

- `dcd_extraction.py` - writes DCD trajectories (per replica and a demuxed
  state-0 trajectory) from the `.nc` storage. Edit the CONFIG block at the top.
- `fes_with_structures.py` - free-energy surface annotated with representative
  structures.

---

## 13. Validating your installation

Before committing GPU time, check that REST2 is built correctly **for your own
system**:

```bash
python benchmarks/validate_rest2.py --prmtop complex.prmtop --inpcrd complex.inpcrd \
    --ligand-resnames MOL --run-dir rest2_output
```

It verifies, among other things, that the heated region contains the atoms you
expect, that every energy term is scaled, that state 0 reproduces the
unmodified force field exactly, that the lambda-scaling law holds, that forces
match the energies, and - given a finished or running production - that the
stored exchange energies can be reproduced.

Checks 1-7 need only the topology and coordinates, so run them **before**
starting a long job. See [benchmarks/README.md](benchmarks/README.md).

---

## 14. Troubleshooting

### tleap fails in System Generation

The error dialog shows the headline; press **Show Details** for tleap's own
output, the error lines from `leap.log`, and the paths to `leap.log` and the
generated `tleap.in`.

| Message | Cause | Fix |
|---|---|---|
| `Atom .R<NMET 1>.A<H 20> does not have a type` | N-terminal hydrogen naming | Fixed automatically; if it persists, check for other non-standard atoms |
| `Unknown residue: XXX` | A residue tleap has no parameters for | Remove it in step 1, or parameterise it as a ligand |
| `... does not have a type` on your ligand | Ligand parameters missing | Make sure Ligand Prep finished and produced `.mol2` + `.frcmod` |
| `conda run ... failed` only | The real error is further up | Open **Show Details** |

### Ligand Prep fails

Most often the **Net charge** is wrong. The details pane includes `sqm.out`,
where the AM1-BCC charge calculation reports failures.

### "Particle coordinate is NaN"

openmmtools restarts the affected step automatically; occasional restarts over
a long run are tolerable. If they are frequent:

1. Set **Hydrogen mass** to 3.0 amu (keep the timestep at 2 fs).
2. Or lower the timestep.
3. Check the ligand parameters for `ATTN, need revision`.
4. Lengthen equilibration.

### Exchange acceptance is ~100%

The replicas are not actually different. Check that T max > T min, and run the
validation script (§13).

### Hot replicas look identical to state 0

Expected for short runs. Work through the checklist in §10, then §8: a λ of
0.95 (T max 330 K) changes almost nothing, and 1-3 ns is far too short for
structural change. Confirm REST2 is working by checking that
`11_e_solute_distributions.png` separates across states.

### Too few frames in the structural plots

The checkpoint interval was too large. Frame spacing is
`checkpoint interval × exchange interval × timestep`; frames that were never
written cannot be recovered. Use 0.1-0.2 ns spacing next time (§7).

### Qt warnings on Linux

Messages like `qt.qpa.wayland.textinput ... Got leave event` are harmless and
are suppressed by default. If the interface misbehaves under Wayland, run with
`QT_QPA_PLATFORM=xcb python main.py`.
