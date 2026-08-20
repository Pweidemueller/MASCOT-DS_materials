# Downloaded Data

This document describes the datasets that should be stored in the `data/` folder. **Note**: Some datasets require approval before being used in research/published.

---
## Case counts
### Statewide COVID-19 Cases Deaths Tests - ARCHIVED
| Field | Value |
|---|---|
| **Source URL** | https://data.chhs.ca.gov/dataset/covid-19-time-series-metrics-by-county-and-state/resource/046cdd2b-31e5-4d34-9ed3-b48cdbc4be7a |
| **Local file** | `data/covid19cases_test.csv` |
| **Last accessed** | Mar 2, 2026 |

**Description**
Regional, daily data (test positivity, flu typing, emergency department visits) and severity (hospital admissions, deaths, and pediatric deaths). Updated until December 2023.

**Attribution:** CDPH, Division of Communicable Disease Control

## Wastewater
### Wastewater Surveillance, California

| Field | Value |
|---|---|
| **Source URL** | https://data.chhs.ca.gov/dataset/wastewater-surveillance-data-california/resource/2742b824-3736-4292-90a9-7fad98e94c06?inner_span=True |
| **Local file** | `data/wastewatersurveillancecalifornia.csv` |
| **Last accessed** | Mar 2, 2026 |

**Description**
California Department of Public Health (CDPH) is coordinating with wastewater utilities, local health departments, academic researchers, and laboratories in California on wastewater surveillance. Data collected from this network of participants, called the California Surveillance of Wastewaters (Cal-SuWers) Network, are submitted to the U.S. Centers for Disease Control and Prevention (CDC) National Wastewater Surveillance System (NWSS). Methodologies for producing wastewater data are not currently standardized, and analyses, comparisons, and aggregations should be done with caution. 

*Anyone seeking to use the database for other purposes or for research is required to contact the WastewaterSCAN/SCAN team at wwscan_stanford_emory@lists.stanford.edu*

There are measurements for SARS-CoV-2 with `pcr_gene_target` being `n`, `s`, `n1` and `n2`. I used `n` measurements.

There's were no WastewaterSCAN entries from before 2022-01-09. Before that the data source was `Sewage Coronavirus Alert Network (SCAN)`. 

**Attribution:** CDPH, Wastewater Surveillance, Surveillance Section | Coronavirus Control Branch and WastewaterSCAN

## Seroprevalence
### Nationwide Commercial Laboratory Seroprevalence Survey

| Field | Value |
|---|---|
| **Source URL** | https://data.cdc.gov/Laboratory-Surveillance/Nationwide-Commercial-Laboratory-Seroprevalence-Su/d2tw-32xv/data_preview |
| **Local file** | `data/Nationwide_Commercial_Laboratory_Seroprevalence_Survey_20260302.csv` |
| **Last accessed** | Mar 2, 2026 |

**Description**
CDC works with commercial laboratories on state, local, territorial, academic, and commercial level. Clinical blood samples are tested for antibodies. For California rounds 31 and later (Mar 14 - Apr 30, 2022 and onwards) seem to only have data for children (under 18) not for the whole population. Anti-S antibody is present both after vaccination and infection, Anti-N only after infection

**Attribution:** National Center for Immunization and Respiratory Diseases (NCIRD), Division of Viral Diseases 

## Variants
### COVID 19 Variant Data, California

| Field | Value |
|---|---|
| **Source URL** | https://data.chhs.ca.gov/dataset/covid-19-variant-data/resource/d7f9acfa-b113-4cbc-9abc-91e707efc08a |
| **Local file** | `data/covid19_variants.csv` |
| **Last accessed** | Mar 2, 2026 |

**Description**
CDPH sequencing of a subset of positive samples.

**Attribution:** CDPH, COVID-19 Response Data, Informatics, Surveillance, Clinical and Outbreaks (DISCO) Team

### CoVariants
| Field | Value |
|---|---|
| **Source URL** | https://github.com/hodcroftlab/covariants/blob/master/cluster_tables/USAClusters_data.json |
| **Local file** | `data/USAClusters_data.json` |
| **Last accessed** | Aug 6, 2026 |

**Description**
SARS-CoV-2 variant frequencies based on GISAID entries world-wide and also for US states and Switzerland.  

**Attribution:** Emma B. Hodcroft. 2021. "CoVariants: SARS-CoV-2 Mutations and Variants of Interest." https://covariants.org/

## Genomic data - sequences
### Winter 2020-2021 Epsilon wave in the US
Epsilon wave (PANGO lineage B.1.427 and B.1.429). In GISAID EpiCoV, I used following filters:

- Location: North America / USA
- Boxes ticked: Complete, Low coverage excluded, Collection date complete
- Variant: Lineage B.1.427 and B.1.429 (downloaded separately)
- Host: Human
- Sampling date on or before April 30, 2021.

Put the fasta and also metadata (.tsv) files into lineage specific subfolders:

- `data/GISAID_sequences/B1427`, e.g. `gisaid_hcov-19_2026_03_05_00.tsv` and `gisaid_hcov-19_2026_03_05_00.fasta`
- `data/GISAID_sequences/B1429`

It is possible to add multiple pairs of tsv and fasta files into the lineage folders, since GISAID restricts the number of sequences that can be downloaded in one download.

**Description**
All genome sequences used in this study and associated metadata can be accessed through the EPI_SET_260725qm identifier (\url{https://doi.org/10.55876/gis8.260725qm}). Also see the uploaded PDF relating to the [GISAID EPI SET identifier](data/gisaid_supplemental_table_epi_set_260725qm.pdf).

**Attribution:**
Khare, S., et al (2021) GISAID’s Role in Pandemic Response. China CDC Weekly, 3(49): 1049-1051. doi: 10.46234/ccdcw2021.255 

We gratefully acknowledge all data contributors, i.e., the Authors and their Originating laboratories responsible for obtaining the specimens, and their Submitting laboratories for generating the genetic sequence and metadata and sharing via the GISAID Initiative, on which this research is based.
