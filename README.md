# MASCOT-DS materials

This repository contains scripts to reproduce the figures and analyses related to the MASCOT-DS preprint:

> Weidemüller PH, Esquivel Gomez LR, Rodriguez-Barraquer I, Müller NF.
> MASCOT-DS improves transmission dynamics inference by integrating multiple
> epidemiological data streams with phylodynamic inference. *medRxiv* 2026.
> doi: [10.64898/2026.08.21.26361056](https://doi.org/10.64898/2026.08.21.26361056)

## Overview

No single surveillance signal tells the full story of an outbreak. Case counts, wastewater viral concentrations, seroprevalence surveys, and pathogen genomes each offer a different, incomplete view on how a pathogen is actually spreading. Existing methods generally analyze them one at a time, or sometimes using a subset of them.

**MASCOT-DS (MASCOT-DataStreams)** is a [BEAST2](https://www.beast2.org/) package that closes this gap: it extends the MASCOT structured-coalescent model so that prevalence and between-location transmission rates can be inferred jointly from any combination of genomic and epidemiological data streams, such as case counts, wastewater concentrations, seroprevalence data. The MASCOT-DS BEAST2 package itself is available at [github.com/Pweidemueller/Mascot_datastreams](https://github.com/Pweidemueller/Mascot_datastreams).

## Usage
This repository contains the scripts and instructions on how to run the simulation study and real-world application on the SARS-CoV-2 winter 2020-21 wave in California.

1. `bayarea_application` contains:

- instructions on how to access and download input data necessary for the analyses
- run the analyses
- produce final figures

2. `simulation_study` contains:

- nextflow pipeline to run the full simulation study from simulation to producing MCMC inference results using BEAST2 and MASCOT-DS
- nextflow pipeline to analyse the BEAST2 outputs and produce final figures

## Citation

If you use this repository, please cite:

> Weidemüller PH, Esquivel Gomez LR, Rodriguez-Barraquer I, Müller NF.
> MASCOT-DS improves transmission dynamics inference by integrating multiple
> epidemiological data streams with phylodynamic inference. *medRxiv* 2026.
> doi: [10.64898/2026.08.21.26361056](https://doi.org/10.64898/2026.08.21.26361056)

