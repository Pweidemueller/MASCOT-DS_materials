// Nextflow pipeline for structured SIR simulations: generate BEAST2 XMLs from contact matrices
//
// Improvements applied from IMPROVEMENT_PLAN.md section 1:
//   1.1  parseVariant() helper replaces 5× duplicated regex
//   1.2  Canonical empty files replace per-base/variant unique files
//   1.3  Consolidated COMBINE_LOGS: one job per xmlname (loop over log types)
//   1.4  Simplified base_inputs channel
//   1.5  base_name/variant_type carried through channels from MAKE_MASCOT_XML
//   1.6  storeDir removed; publishDir used for RUN_MASCOT; ANALYSE_FROM_PUBLISHED entrypoint added
//   1.7  Clip variants removed (only noclip was active; _noclip suffix dropped)
//   1.8  Hardcoded conda path replaced with params.conda_env
//   1.9  ANALYSE_POSTERIORS uses val(meta) map pattern
//
// Improvements applied from IMPROVEMENT_PLAN.md section 2:
//   2.   Integrated standalone scripts as Nextflow processes:
//        MAKE_INDIVIDUAL_SIM_FIGURES, COMBINE_HPD_NE_BY_MODEL,
//        QUANTIFY_INFORMATION_CONTENT, MAKE_FIGURE_TRUE_VS_ESTIMATE

nextflow.enable.dsl=2

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

// [1.1] Parse base_name and variant_type from an XML/log basename.
// Fallback for edge cases where metadata is not available through channels.
def parseVariant(String name) {
    if (name.endsWith('_original')) {
        return [name.replaceAll(/_original$/, ''), 'original']
    }
    def matcher = name =~ /^(.*)_(datastreams.*)$/
    if (matcher) {
        return [matcher[0][1], matcher[0][2]]
    }
    return [name, 'unknown']
}

// [1.2] Empty-file machinery eliminated for datastream files.
// All variants receive the real datastream files; the variant_type argument
// tells create_mascot_xml_fixedtree.py which datastreams to actually use.
// For analysis, real files serve as ground truth regardless of MASCOT variant.
//
// The emptyFile() helper is retained ONLY as a safety-net fallback for combined
// log files that may not exist (e.g. original variant has no NeDynamics logs).
// Each call uses a per-(base,variant) name to avoid Nextflow staging collisions.
def emptyFile(String name) {
    def target = new File("${projectDir}/empty_files/${name}")
    target.parentFile.mkdirs()
    if (!target.exists()) target.text = ""
    file(target.toString())
}

// ---------------------------------------------------------------------------
// Process definitions
// ---------------------------------------------------------------------------

process SAMPLE_SIMPARAMS {
    tag "${ndemes}demes_seed${seed}"                                             // [1.8]
    publishDir "${params.outdir}/0_sample_simparameters", mode: 'copy'

    input:
    tuple val(simNb), val(ndemes), val(pop_sizes), val(seed)

    output:
    tuple val(simNb), val(ndemes), path("${simNb}_${ndemes}_matrix.csv"), path("${simNb}_${ndemes}_sampled_parameters.csv")

    shell:
    """
    sample_simparameters.py \
        --n_demes ${ndemes} \
        --population_sizes "${pop_sizes}" \
        --seed ${seed} \
        --matrix_csv ${simNb}_${ndemes}_matrix.csv \
        --params_csv ${simNb}_${ndemes}_sampled_parameters.csv
    """
}

process MAKE_SIM_XML {
    tag "${contact.baseName}"                                           // [1.8]
    publishDir "${params.outdir}/1_remaster_sim", mode: 'copy'

    input:
    tuple val(simNb), val(ndemes), path(contact), path(sampled_params)

    output:
    tuple val(simNb), val(ndemes), path("${simNb}_${ndemes}_simulation.xml"), path("${simNb}_${ndemes}_simulation_parameters.csv")

    shell:
    """
    create_birthdeath_simXML.py \
        --sampled_params ${sampled_params} \
        --ndemes ${ndemes} \
        --maxTime ${params.maxTime} \
        --endsWhen ${params.endsWhen} \
        --output_file ${simNb}_${ndemes}_simulation.xml
    """
}

process RUN_REMASTER {
    tag "${xml.baseName}"
    publishDir "${params.outdir}/1_remaster_sim", mode: 'copy'

    input:
    tuple val(simNb), val(ndemes), path(xml), path(params_csv)

    output:
    tuple val(simNb), val(ndemes), path(params_csv), path("${xml.baseName}.trees"), path("${xml.baseName}.traj"), path("${xml.baseName}.nexus"), emit: remaster_outputs
    path("${xml.baseName}.log"), emit: remaster_log

    shell:
    """
    "${params.beast_path}" -overwrite ${xml} > ${xml.baseName}.log 2>&1
    """
}

process PLOT_TREES_GROUNDTRUTH {
    tag "${trees.baseName}"
    publishDir "${params.outdir}/1_remaster_sim", mode: 'copy'

    input:
    tuple val(simNb), val(ndemes), path(params_csv), path(trees), path(traj), path(nexus)

    output:
    tuple val("${trees.baseName}"), path("${trees.baseName}_deme_switches_groundtruth.csv"), emit: deme_csv
    path("${trees.baseName}_tree_groundtruth.png"), emit: tree_plot

    shell:
    """
    plot_trees.R ${trees} groundtruth
    """
}

process SIMULATE_DATASTREAMS {
    tag "${traj.baseName}"                                               // [1.8]
    publishDir "${params.outdir}/1_remaster_sim", mode: 'copy'

    input:
    tuple val(simNb), val(ndemes), path(params_csv), path(trees), path(traj), path(nexus)

    output:
    tuple val(simNb), val(ndemes), path(params_csv), path(trees), path(traj), path(nexus), path("${traj.baseName}_casecounts.csv"), path("${traj.baseName}_seroprevalence.csv"), path("${traj.baseName}_wastewater.csv"), emit: datastreams
    path "${traj.baseName}_trajectories.png", emit: output
    path "${traj.baseName}_sim_metadata.csv", emit: sim_metadata

    shell:
    """
    simulate_datastreams.py --traj_file ${traj} --params_csv ${params_csv} --out_prefix ${traj.baseName}
    """
}

process CONCATENATE_SIM_METADATA {
    tag "sim metadata"
    publishDir "${params.outdir}/3_analysis", mode: 'copy'

    input:
    path csvs

    output:
    path "all_sim_metadata.csv", emit: sim_metadata_csv

    script:
    def csvList = csvs instanceof List ? csvs : [csvs]
    def firstFile = csvList[0]
    def otherFiles = csvList.size() > 1 ? csvList[1..-1] : []
    """
    head -n 1 "${firstFile}" > all_sim_metadata.csv
    tail -n +2 "${firstFile}" >> all_sim_metadata.csv
    ${otherFiles.collect { f -> "tail -n +2 \"${f}\" >> all_sim_metadata.csv" }.join('\n    ')}
    """
}

// [1.5] MAKE_MASCOT_XML now emits base_name and variant_type as val() outputs
// [1.7] Clip dimension removed — always passes --clip-trans-rate false
process MAKE_MASCOT_XML {
    tag "${nexus.baseName}_${variant_type}"
    publishDir "${params.outdir}/2_mascot/${nexus.baseName}/${variant_type}", mode: 'copy', pattern: '*.xml'
    publishDir "${params.outdir}/2_mascot/${nexus.baseName}", mode: 'copy', pattern: '*_state_time.csv'

    input:
    tuple val(simNb), val(ndemes), path(params_csv), path(trees), path(traj), path(nexus), path(casecounts), path(seroprevalence), path(wastewater), val(variant_type)

    output:
    tuple val(simNb), val(ndemes), path(trees), path(traj), path(nexus), path("${nexus.baseName}_${variant_type}.xml"), val("${nexus.baseName}"), val(variant_type), emit: mascot_xmls
    path "${nexus.baseName}_state_time.csv", emit: output

    script:
    """
    create_mascot_xml_fixedtree.py \
        --standard_template "${projectDir}/data/Mascot_template_fixedtree.xml" \
        --datastream_template "${projectDir}/data/Mascot_datastreams_template_fixedtree.xml" \
        --tree ${trees} \
        --case_counts ${casecounts} \
        --seroprevalence ${seroprevalence} \
        --wastewater ${wastewater} \
        --parameters ${params_csv} \
        --xml_name ${nexus.baseName} \
        --variant_type ${variant_type} \
        --clip-trans-rate false \
        --chain_length ${params.chainlength}
    """
}

// [1.5] RUN_MASCOT carries base_name and variant_type through all output emits
// [1.6] publishDir enabled for MASCOT outputs
process RUN_MASCOT {
    tag "${xmlfile.baseName} (seed=${seed})"
    publishDir "${params.outdir}/2_mascot/${base_name}/${variant_type}", mode: 'copy', pattern: "${seed}_${xmlfile.baseName}.*"

    input:
    tuple val(simNb), val(ndemes), path(trees), path(traj), path(nexus), path(xmlfile), val(seed), val(base_name), val(variant_type)

    output:
    tuple val(base_name), val(variant_type), path("${seed}_${xmlfile.baseName}.*"), emit: outputs
    tuple val(base_name), val(variant_type), val(xmlfile.baseName), path("${seed}_${xmlfile.baseName}.log"), emit: mascot_logs
    tuple val(base_name), val(variant_type), val(xmlfile.baseName), path("${seed}_${xmlfile.baseName}.SimDataset.trees"), emit: mascot_trees

    shell:
    """
    "${params.beast_path}" -overwrite -seed ${seed} -threads ${task.cpus} -prefix "${seed}_" ${xmlfile}
    cp .command.out "${seed}_${xmlfile.baseName}.out"
    cp .command.err "${seed}_${xmlfile.baseName}.err"
    """
}

// [1.3] Consolidated: one invocation per xmlname runs logcombiner for all log types in a loop
process COMBINE_LOGS {
    tag "${xmlname}"
    publishDir "${params.outdir}/2_mascot/${base_name}/${variant_type}", mode: 'copy'

    input:
    tuple val(xmlname), val(base_name), val(variant_type), path(logs)

    output:
    tuple val(xmlname), val(base_name), val(variant_type), path("${xmlname}.*.combined.log"), emit: combined_logs

    script:
    """
    #!/bin/bash
    set -euo pipefail

    # Main MASCOT log: files ending in .log but NOT NeDynamics or cumulativeIncidence
    main_logs=\$(ls *.log 2>/dev/null | grep -v '\\.NeDynamics\\.' | grep -v '\\.cumulativeIncidence\\.' || true)
    if [ -n "\$main_logs" ]; then
        log_args=""
        for f in \$main_logs; do log_args="\$log_args -log \$f"; done
        ${params.logcombiner_path} -b ${params.log_burnin} -resample ${params.logcombiner_resample} \$log_args -o "${xmlname}.mascot_logs.combined.log"
    fi

    # Per-deme log types (NeDynamics and cumulativeIncidence)
    for logtype in NeDynamics.Deme1 NeDynamics.Deme2 cumulativeIncidence.Deme1 cumulativeIncidence.Deme2; do
        type_logs=\$(ls *.\${logtype}.log 2>/dev/null || true)
        if [ -n "\$type_logs" ]; then
            log_args=""
            for f in \$type_logs; do log_args="\$log_args -log \$f"; done
            ${params.logcombiner_path} -b ${params.log_burnin} -resample ${params.logcombiner_resample} \$log_args -o "${xmlname}.\${logtype}.combined.log"
        fi
    done
    """
}

process COMBINE_TREES {
    tag "${xmlname}"
    publishDir "${params.outdir}/2_mascot/${base_name}/${variant_type}", mode: 'copy'

    input:
    tuple val(xmlname), path(trees), val(base_name), val(variant_type)

    output:
    tuple val(base_name), val(variant_type), val(xmlname), path("${xmlname}.combined.trees"), emit: combined_trees

    script:
    def treeArgs = trees.collect { f -> "-log \"${f}\"" }.join(' ')
    """
    ${params.logcombiner_path} -b ${params.tree_burnin} -resample ${params.logcombiner_resample} ${treeArgs} -o "${xmlname}.combined.trees"
    """
}

process ANNOTATE_TREES {
    tag "${xmlname}"
    publishDir "${params.outdir}/2_mascot/${base_name}/${variant_type}", mode: 'copy'

    input:
    tuple val(base_name), val(variant_type), val(xmlname), path(combined_trees)

    output:
    tuple val(base_name), val(variant_type), val(xmlname), path("${xmlname}.combined.trees"), path("${xmlname}.mcc.trees"), emit: annotated_trees

    script:
    """
    ${params.treeannotator_path} -burnin ${params.tree_burnin} -height mean -lowMem true ${combined_trees} "${xmlname}.mcc.trees"
    """
}

process PLOT_TREES_MASCOT {
    tag "${mcc_tree.baseName}"
    publishDir "${params.outdir}/2_mascot/${base_name}/${variant_type}", mode: 'copy'

    input:
    tuple val(base_name), val(variant_type), val(xmlname), path(combined_trees), path(mcc_tree)

    output:
    path("${mcc_tree.baseName}_tree_*.png"), emit: tree_plot
    tuple val("${mcc_tree.baseName}"), path("${mcc_tree.baseName}_deme_switches_*.csv"), emit: deme_csv

    script:
    def tree_type = variant_type == 'original' ? 'original' : 'datastreams'
    """
    plot_trees.R ${mcc_tree} ${tree_type}
    """
}

// [1.9] Uses val(meta) map pattern instead of unwieldy 15-element tuple
process ANALYSE_POSTERIORS {
    tag "${meta.base}_${meta.variant}"
    publishDir "${params.outdir}/3_analysis/${meta.base}/${meta.variant}", mode: 'copy'

    input:
    tuple val(meta), path(log_original), path(log_datastream), path(log_nedyn_deme1), path(log_nedyn_deme2), path(log_cuminc_deme1), path(log_cuminc_deme2), path(params_csv), path(traj), path(case_counts), path(seroprevalence), path(wastewater), path(deme_switches_csv)

    output:
    path "${meta.base}_${meta.variant}_*", emit: analysis_outputs
    tuple val(meta.base), val(meta.variant), path("*_datastreams_hpd_validation_cumulative_incidence.csv"), path("*_datastreams_hpd_validation_ne.csv"), path("*_datastreams_hpd_validation_params.csv"), path("*_datastreams_hpd_validation_prevalence.csv"), path("*_hpd_validation_migration_rates.csv"), path("*_original_hpd_validation_ne.csv"), emit: hpd_validation_outputs

    shell:
    """
    analyse_posteriors.py \
        --log_file_original ${log_original} \
        --log_file_datastream ${log_datastream} \
        --case_counts_file ${case_counts} \
        --seroprevalence_file ${seroprevalence} \
        --wastewater_file ${wastewater} \
        --cumulative_incidence_deme1 ${log_cuminc_deme1} \
        --cumulative_incidence_deme2 ${log_cuminc_deme2} \
        --nedynamics_deme1 ${log_nedyn_deme1} \
        --nedynamics_deme2 ${log_nedyn_deme2} \
        --trajectory_file ${traj} \
        --params_csv ${params_csv} \
        --deme_switches_csv ${deme_switches_csv} \
        --burnin 0.0 \
        --out_prefix ${meta.base}_${meta.variant}
    if [ ! -s "${meta.base}_${meta.variant}_datastreams_hpd_validation_cumulative_incidence.csv" ]; then
        : > "${meta.base}_${meta.variant}_datastreams_hpd_validation_cumulative_incidence.csv"
    fi
    """
}

process CONCATENATE_HPD_VALIDATION {
    tag "${type_name}"
    publishDir "${params.outdir}/3_analysis", mode: 'copy'

    input:
    tuple val(type_name), val(variant), path(csvs)

    output:
    tuple val(type_name), val(variant), path("all_${type_name}_${variant}_hpd_validation.csv"), emit: concatenated_csv

    script:
    def csvList = csvs instanceof List ? csvs : [csvs]
    def firstFile = csvList[0]
    def otherFiles = csvList.size() > 1 ? csvList[1..-1] : []
    """
    # Take header from the first file
    head -n 1 "${firstFile}" > "all_${type_name}_${variant}_hpd_validation.csv"

    # Append without header from first file
    tail -n +2 "${firstFile}" >> "all_${type_name}_${variant}_hpd_validation.csv"

    # Append without header from remaining files
    ${otherFiles.collect { f -> "tail -n +2 \"${f}\" >> \"all_${type_name}_${variant}_hpd_validation.csv\"" }.join('\n    ')}
    """
}

process PLOT_HPD_VALIDATION {
    tag "HPD validation plots"
    publishDir "${params.outdir}/3_analysis", mode: 'copy'

    input:
    tuple val(variant), path(prevalence_csv), path(ne_csv), path(cumulative_incidence_csv), path(params_csv), path(migration_rates_csv)

    output:
    tuple val(variant), path("all_hpd_validation_${variant}*.png"), path("all_hpd_validation_${variant}*.pdf"), emit: plots

    script:
    """
    plot_hpd_validation.py \
        --cumulative_incidence ${cumulative_incidence_csv} \
        --ne ${ne_csv} \
        --prevalence ${prevalence_csv} \
        --parameters ${params_csv} \
        --migration_rates ${migration_rates_csv} \
        --variant ${variant} \
        --output_dir .
    """
}

process CALCULATE_ESS {
    tag "${run_name}"
    // publishDir "${params.outdir}/4_ess", mode: 'copy'

    input:
    tuple val(run_name), val(variant), path(log_file)

    output:
    tuple val(variant), path("${run_name}_ess.csv"), emit: ess_csv
    path("${run_name}_ess.out"), emit: output

    script:
    """
    calculate_ess.py \
        --burnin 0.0 \
        --output "${run_name}_ess.csv" \
        --run-name "${run_name}" \
        ${log_file} > ${run_name}_ess.out
    """
}

process AGGREGATE_ESS {
    tag "ESS summary"
    publishDir "${params.outdir}/4_ess", mode: 'copy'

    input:
    tuple val(type_name), val(variant), path(csvs)

    output:
    tuple val(variant), path("ess_summary_${variant}.csv"), emit: ess_summary

    script:
    def csvList = csvs instanceof List ? csvs : [csvs]
    def firstFile = csvList[0]
    def otherFiles = csvList.size() > 1 ? csvList[1..-1] : []
    """
    # Take header from the first file
    head -n 1 "${firstFile}" > "ess_summary_${variant}.csv"

    # Append without header from first file
    tail -n +2 "${firstFile}" >> "ess_summary_${variant}.csv"

    # Append without header from remaining files
    ${otherFiles.collect { f -> "tail -n +2 \"${f}\" >> \"ess_summary_${variant}.csv\"" }.join('\n    ')}
    """
}

process PLOT_ESS_HEATMAP {
    tag "ESS heatmap"
    publishDir "${params.outdir}/4_ess", mode: 'copy'

    input:
    tuple val(variant), path(original_ess_summary), path(variant_ess_summary)

    output:
    tuple val(variant), path("ess_summary_${variant}_heatmap.png"), path("ess_summary_${variant}_heatmap.pdf")

    script:
    """
    # Concatenate original and variant ESS summaries into a single CSV
    head -n 1 ${original_ess_summary} > ess_summary_${variant}_heatmap.csv
    tail -n +2 ${original_ess_summary} >> ess_summary_${variant}_heatmap.csv
    tail -n +2 ${variant_ess_summary} >> ess_summary_${variant}_heatmap.csv

    # Plot heatmap comparing original vs current variant
    plot_ess_heatmap.py --ess_summary ess_summary_${variant}_heatmap.csv --out_prefix ess_summary_${variant}_heatmap
    """
}

// [2] Per-simulation publication figures (prevalence, Ne, cumIncidence per deme)
// Same inputs as ANALYSE_POSTERIORS — reuses parse_arguments() from analyse_posteriors.py
process MAKE_INDIVIDUAL_SIM_FIGURES {
    tag "${meta.base}_${meta.variant}"
    publishDir "${params.outdir}/3_analysis/${meta.base}/${meta.variant}", mode: 'copy'

    input:
    tuple val(meta), path(log_original), path(log_datastream), path(log_nedyn_deme1), path(log_nedyn_deme2), path(log_cuminc_deme1), path(log_cuminc_deme2), path(params_csv), path(traj), path(case_counts), path(seroprevalence), path(wastewater), path(deme_switches_csv)

    output:
    path "${meta.base}_${meta.variant}_*", emit: figure_outputs

    shell:
    """
    make_figure_individualsim.py \
        --log_file_original ${log_original} \
        --log_file_datastream ${log_datastream} \
        --case_counts_file ${case_counts} \
        --seroprevalence_file ${seroprevalence} \
        --wastewater_file ${wastewater} \
        --cumulative_incidence_deme1 ${log_cuminc_deme1} \
        --cumulative_incidence_deme2 ${log_cuminc_deme2} \
        --nedynamics_deme1 ${log_nedyn_deme1} \
        --nedynamics_deme2 ${log_nedyn_deme2} \
        --trajectory_file ${traj} \
        --params_csv ${params_csv} \
        --deme_switches_csv ${deme_switches_csv} \
        --burnin 0.0 \
        --out_prefix ${meta.base}_${meta.variant}
    """
}

// [2] Combine Ne HPD validation CSVs from original + datastreams into one CSV with Model column
process COMBINE_HPD_NE_BY_MODEL {
    tag "Ne by model"
    publishDir "${params.outdir}/3_analysis", mode: 'copy'

    input:
    tuple path(original_ne_csv), path(datastreams_ne_csv)

    output:
    path "combined_ne_hpd_validation_by_model.csv", emit: combined_ne_csv

    script:
    """
    combine_hpd_validation_ne_by_model.py \
        --original-ne-csv ${original_ne_csv} \
        --datastreams-ne-csv ${datastreams_ne_csv} \
        --output combined_ne_hpd_validation_by_model.csv
    """
}

// [2] Quantify information content by comparing HPD widths across leave-one-out variants
// Uses a single flat path input to avoid combine() flattening separate collected lists.
// Files are sorted into params/prevalence/migration_rates by filename prefix.
process QUANTIFY_INFORMATION_CONTENT {
    tag "information content"
    publishDir "${params.outdir}/6_informationcontent", mode: 'copy'

    input:
    path all_csvs

    output:
    path "information_content_*", emit: plots

    script:
    def fileList = all_csvs instanceof List ? all_csvs : [all_csvs]
    def paramArgs = fileList.findAll { it.name.startsWith('all_params_') }.collect { it.name }.join(' ')
    def prevArgs  = fileList.findAll { it.name.startsWith('all_prevalence_') }.collect { it.name }.join(' ')
    def migArgs   = fileList.findAll { it.name.startsWith('all_migration_rates_') }.collect { it.name }.join(' ')
    def prevFlag  = prevArgs  ? "--prevalence-files ${prevArgs}" : ''
    def migFlag   = migArgs   ? "--migration-rates-files ${migArgs}" : ''
    """
    quantify_informationcontent.py \
        ${paramArgs} \
        ${prevFlag} \
        ${migFlag} \
        --output information_content
    """
}

// [2] True vs estimated scatter plots, migration bias/uncertainty, prevalence/Ne coverage over time
process MAKE_FIGURE_TRUE_VS_ESTIMATE {
    tag "true vs estimate"
    publishDir "${params.outdir}/5_simulation_study", mode: 'copy'

    input:
    tuple path(params_csv), path(migration_rates_csv), path(prevalence_csv), path(combined_ne_csv), path(sim_metadata_csv)

    output:
    path "*.png", emit: png_plots
    path "*.pdf", emit: pdf_plots
    path "*.csv", optional: true, emit: enriched_csvs

    script:
    """
    make_figure_param_truevsestimate.py \
        --csv ${params_csv} \
        --migration_rates_csv ${migration_rates_csv} \
        --prevalence_csv ${prevalence_csv} \
        --combined_ne_csv ${combined_ne_csv} \
        --sim_metadata_csv ${sim_metadata_csv} \
        --output_dir .
    """
}

// ---------------------------------------------------------------------------
// Main workflow
// ---------------------------------------------------------------------------
workflow {
    // Build tuples for sampling (index 1..50, seed 42+index)
    tuples_to_sample = Channel.from(1..5).map { n ->
        tuple(n, params.ndemes, params.population_sizes, 41 + n)
    }

    sampled = SAMPLE_SIMPARAMS(tuples_to_sample)
    xmls = MAKE_SIM_XML(sampled)
    beast_outputs = RUN_REMASTER(xmls)
    groundtruth_tree = PLOT_TREES_GROUNDTRUTH(beast_outputs.remaster_outputs)
    ds_outputs = SIMULATE_DATASTREAMS(beast_outputs.remaster_outputs)

    // ── Create variants ─────────────────────────────────────────────────
    // [1.7] Clip dimension removed: variant_type is used directly (no _noclip suffix)
    // [1.2] All variants receive the real datastream files; variant_type controls
    //       which ones the script actually uses. No empty files needed.
    datastream_variants = ds_outputs.datastreams.flatMap { t ->
        def (simNb, nd, params_csv, tr, tj, nx, cc, sp, ww) = t
        [
            tuple(simNb, nd, params_csv, tr, tj, nx, cc, sp, ww, "datastreams"),
            tuple(simNb, nd, params_csv, tr, tj, nx, cc, sp, ww, "datastreams_nocasecounts"),
            tuple(simNb, nd, params_csv, tr, tj, nx, cc, sp, ww, "datastreams_noseroprevalence"),
            tuple(simNb, nd, params_csv, tr, tj, nx, cc, sp, ww, "datastreams_nowastewater"),
            tuple(simNb, nd, params_csv, tr, tj, nx, cc, sp, ww, "datastreams_nomascotll"),
            tuple(simNb, nd, params_csv, tr, tj, nx, cc, sp, ww, "datastreams_onlytree"),
        ]
    }

    original_variants = ds_outputs.datastreams.map { t ->
        def (simNb, nd, params_csv, tr, tj, nx, cc, sp, ww) = t
        tuple(simNb, nd, params_csv, tr, tj, nx, cc, sp, ww, "original")
    }

    all_variants = original_variants.concat(datastream_variants)
    mascot = MAKE_MASCOT_XML(all_variants)

    // ── Seed expansion ──────────────────────────────────────────────────
    // [1.5] base_name and variant_type carried from MAKE_MASCOT_XML — no re-parsing
    mascot_all_seeded = mascot.mascot_xmls.flatMap { t ->
        def (simNb, ndemes, trees, traj, nexus, xmlfile, base_name, variant_type) = t
        [410, 430, 450].collect { seed ->
            tuple(simNb, ndemes, trees, traj, nexus, xmlfile, seed, base_name, variant_type)
        }
    }

    mascot_runs = RUN_MASCOT(mascot_all_seeded)

    // ── Log combining ───────────────────────────────────────────────────
    // [1.3] Consolidated: collect ALL log files per xmlname, run one COMBINE_LOGS job
    // [1.1/1.5] No regex parsing — base_name/variant_type from channel metadata
    all_combinable_logs = mascot_runs.outputs
        .flatMap { base_name, variant_type, files ->
            def xmlname = base_name + "_" + variant_type
            def fileList = files instanceof List ? files : [files]
            fileList.findAll { f -> f.name.endsWith('.log') }
                .collect { f -> tuple(xmlname, base_name, variant_type, f) }
        }
        .groupTuple(by: [0, 1, 2])

    combined = COMBINE_LOGS(all_combinable_logs)

    // ── Tree combining ──────────────────────────────────────────────────
    // [1.5] Metadata from channel, no filename re-parsing needed
    trees_grouped = mascot_runs.mascot_trees
        .map { base_name, variant_type, xmlname, file ->
            tuple(xmlname, file, base_name, variant_type)
        }
        .groupTuple(by: [0, 2, 3])

    combined_trees = COMBINE_TREES(trees_grouped)
    mcc_trees = ANNOTATE_TREES(combined_trees.combined_trees)
    PLOT_TREES_MASCOT(mcc_trees.annotated_trees)

    // ── ESS calculation ─────────────────────────────────────────────────
    // Extract mascot_logs combined log from each consolidated COMBINE_LOGS output
    ess_inputs = combined.combined_logs
        .map { xmlname, base_name, variant_type, logs ->
            def logList = logs instanceof List ? logs : [logs]
            def mascot_log = logList.find { it.name.contains('mascot_logs') }
            mascot_log != null
                ? tuple(xmlname + ".mascot_logs", variant_type, mascot_log)
                : null
        }
        .filter { it != null }

    ess_results = CALCULATE_ESS(ess_inputs)

    ess_csvs_collected = ess_results.ess_csv
        .groupTuple(by: 0)
        .map { variant, csvs -> tuple("ess", variant, csvs) }

    ess_summary = AGGREGATE_ESS(ess_csvs_collected)

    // ESS heatmap: pair each non-original variant with the original summary
    ess_summary_original = ess_summary.ess_summary
        .filter { variant, csv -> variant == 'original' }
        .map { variant, csv -> tuple('all', csv) }

    ess_summary_other = ess_summary.ess_summary
        .filter { variant, csv -> variant != 'original' }
        .map { variant, csv -> tuple('all', variant, csv) }

    ess_heatmap_inputs = ess_summary_original
        .combine(ess_summary_other, by: 0)
        .map { key, orig_csv, variant, var_csv ->
            tuple(variant, orig_csv, var_csv)
        }

    PLOT_ESS_HEATMAP(ess_heatmap_inputs)

    // ── Analysis inputs ─────────────────────────────────────────────────
    // [1.4] Simplified base_inputs: real datastream files for all variants.
    // They serve as ground truth for analysis regardless of MASCOT variant.
    base_inputs = ds_outputs.datastreams.flatMap { t ->
        def (simNb, nd, params_csv, tr, tj, nx, cc, sp, ww) = t
        def base_name = nx.baseName
        [
            tuple(base_name, "original", params_csv, simNb, tj, cc, sp, ww),
            tuple(base_name, "datastreams", params_csv, simNb, tj, cc, sp, ww),
            tuple(base_name, "datastreams_nocasecounts", params_csv, simNb, tj, cc, sp, ww),
            tuple(base_name, "datastreams_noseroprevalence", params_csv, simNb, tj, cc, sp, ww),
            tuple(base_name, "datastreams_nowastewater", params_csv, simNb, tj, cc, sp, ww),
            tuple(base_name, "datastreams_nomascotll", params_csv, simNb, tj, cc, sp, ww),
            tuple(base_name, "datastreams_onlytree", params_csv, simNb, tj, cc, sp, ww),
        ]
    }

    // Extract original MASCOT log per base_name
    original_logs_by_base = combined.combined_logs
        .filter { xmlname, base_name, variant_type, logs -> variant_type == 'original' }
        .map { xmlname, base_name, variant_type, logs ->
            def logList = logs instanceof List ? logs : [logs]
            def mascot_log = logList.find { it.name.contains('mascot_logs') }
            tuple(base_name, mascot_log ?: emptyFile("${base_name}_original_empty_mascot.log"))
        }

    // Extract datastream variant logs and pair with the original log
    logs_by_variant = combined.combined_logs
        .filter { xmlname, base_name, variant_type, logs -> variant_type.startsWith('datastreams') }
        .map { xmlname, base_name, variant_type, logs ->
            def logList = logs instanceof List ? logs : [logs]
            tuple(
                base_name,
                variant_type,
                logList.find { it.name.contains('mascot_logs') }              ?: emptyFile("${base_name}_${variant_type}_empty_mascot.log"),
                logList.find { it.name.contains('NeDynamics.Deme1') }         ?: emptyFile("${base_name}_${variant_type}_empty_nedyn1.log"),
                logList.find { it.name.contains('NeDynamics.Deme2') }         ?: emptyFile("${base_name}_${variant_type}_empty_nedyn2.log"),
                logList.find { it.name.contains('cumulativeIncidence.Deme1') } ?: emptyFile("${base_name}_${variant_type}_empty_cuminc1.log"),
                logList.find { it.name.contains('cumulativeIncidence.Deme2') } ?: emptyFile("${base_name}_${variant_type}_empty_cuminc2.log")
            )
        }
        .combine(original_logs_by_base, by: 0)
        .map { base_name, variant_type, ds_log, nedyn1, nedyn2, cuminc1, cuminc2, orig_log ->
            tuple(base_name, variant_type, orig_log, ds_log, nedyn1, nedyn2, cuminc1, cuminc2)
        }

    // [1.9] Join with base_inputs and ground truth, convert to meta map
    analysis_inputs = logs_by_variant
        .join(base_inputs, by: [0, 1])
        .combine(groundtruth_tree.deme_csv, by: 0)
        .map { base_name, variant_type, orig_log, ds_log, nedyn1, nedyn2, cuminc1, cuminc2,
               params_csv, simNb, traj, cc, sp, ww, deme_csv ->
            def meta = [base: base_name, variant: variant_type, simNb: simNb]
            tuple(meta, orig_log, ds_log, nedyn1, nedyn2, cuminc1, cuminc2,
                  params_csv, traj, cc, sp, ww, deme_csv)
        }

    analysis_results = ANALYSE_POSTERIORS(analysis_inputs)

    // ── HPD validation ──────────────────────────────────────────────────
    grouped_by_type = analysis_results.hpd_validation_outputs
        .flatMap { base, variant, f_ci, f_ne, f_params, f_prev, f_migration_rates, f_original_ne ->
            def tuples = [
                tuple("cumulative_incidence", variant, f_ci),
                tuple("ne", variant, f_ne),
                tuple("params", variant, f_params),
                tuple("prevalence", variant, f_prev),
                tuple("migration_rates", variant, f_migration_rates)
            ]
            if (variant == 'datastreams') {
                tuples << tuple("ne", "original", f_original_ne)
            }
            tuples
        }
        .groupTuple(by: [0, 1])

    concatenated_csvs = CONCATENATE_HPD_VALIDATION(grouped_by_type)

    // Build per-variant inputs for HPD validation plots
    ch_prev   = concatenated_csvs.concatenated_csv.filter { it[0] == 'prevalence' }          .map { it -> tuple(it[1], it[2]) }
    ch_ne     = concatenated_csvs.concatenated_csv.filter { it[0] == 'ne' }                  .map { it -> tuple(it[1], it[2]) }
    ch_ci     = concatenated_csvs.concatenated_csv.filter { it[0] == 'cumulative_incidence' } .map { it -> tuple(it[1], it[2]) }
    ch_params = concatenated_csvs.concatenated_csv.filter { it[0] == 'params' }              .map { it -> tuple(it[1], it[2]) }
    ch_mig    = concatenated_csvs.concatenated_csv.filter { it[0] == 'migration_rates' }     .map { it -> tuple(it[1], it[2]) }

    // plot_inputs = ch_prev
    //     .combine(ch_ne, by: 0)
    //     .combine(ch_ci, by: 0)
    //     .combine(ch_params, by: 0)
    //     .combine(ch_mig, by: 0)

    // PLOT_HPD_VALIDATION(plot_inputs)

    // ── [2] Per-simulation publication figures ──────────────────────────
    MAKE_INDIVIDUAL_SIM_FIGURES(analysis_inputs)

    // ── [2] Combine Ne HPD validation: original + datastreams → Model column
    ne_original = concatenated_csvs.concatenated_csv
        .filter { it[0] == 'ne' && it[1] == 'original' }
        .map { it[2] }

    ne_datastreams = concatenated_csvs.concatenated_csv
        .filter { it[0] == 'ne' && it[1] == 'datastreams' }
        .map { it[2] }

    combine_ne_input = ne_original.combine(ne_datastreams)
    combined_ne = COMBINE_HPD_NE_BY_MODEL(combine_ne_input)

    // ── [2] Quantify information content across leave-one-out variants ──
    // Collect all relevant CSVs (params, prevalence, migration_rates) for
    // datastream variants into a single flat list. The process sorts them
    // by filename prefix, avoiding combine() list-flattening issues.
    info_all_files = concatenated_csvs.concatenated_csv
        .filter { it[0] in ['params', 'prevalence', 'migration_rates'] && it[1].startsWith('datastreams') }
        .map { it[2] }
        .collect()

    QUANTIFY_INFORMATION_CONTENT(info_all_files)

    // ── [2] True vs estimated parameter figures ─────────────────────────
    // Needs: params CSV (datastreams), migration CSV (datastreams), prevalence CSV (datastreams),
    //        combined Ne CSV (from COMBINE_HPD_NE_BY_MODEL), concatenated sim metadata CSV
    fig_params_csv = concatenated_csvs.concatenated_csv
        .filter { it[0] == 'params' && it[1] == 'datastreams' }
        .map { it[2] }

    fig_mig_csv = concatenated_csvs.concatenated_csv
        .filter { it[0] == 'migration_rates' && it[1] == 'datastreams' }
        .map { it[2] }

    fig_prev_csv = concatenated_csvs.concatenated_csv
        .filter { it[0] == 'prevalence' && it[1] == 'datastreams' }
        .map { it[2] }

    // Concatenate per-simulation metadata CSVs (start deme + first-I times)
    all_sim_metadata = ds_outputs.sim_metadata.collect()
    concatenated_sim_metadata = CONCATENATE_SIM_METADATA(all_sim_metadata)

    fig_truevsest_input = fig_params_csv
        .combine(fig_mig_csv)
        .combine(fig_prev_csv)
        .combine(combined_ne.combined_ne_csv)
        .combine(concatenated_sim_metadata.sim_metadata_csv)

    MAKE_FIGURE_TRUE_VS_ESTIMATE(fig_truevsest_input)
}

// ---------------------------------------------------------------------------
// [1.6] Re-analysis workflow: reads from published results directory
// Usage:
//   nextflow run main.nf -entry ANALYSE_FROM_PUBLISHED \
//       --outdir /path/to/published/results
//
// This workflow skips simulation and inference — it only runs post-processing
// and figure generation from previously published outputs.
// ---------------------------------------------------------------------------
workflow ANALYSE_FROM_PUBLISHED {
    // ── Read combined log files from 2_mascot/{base_name}/{variant}/ ────
    combined_logs_ch = Channel
        .fromPath("${params.outdir}/2_mascot/*/*/*.combined.log")
        .map { log ->
            def variant_type = log.parent.name
            def base_name = log.parent.parent.name
            def xmlname = base_name + "_" + variant_type
            tuple(xmlname, base_name, variant_type, log)
        }
        .groupTuple(by: [0, 1, 2])

    // ── Read simulation metadata from 1_remaster_sim/ ───────────────────
    sim_params = Channel
        .fromPath("${params.outdir}/1_remaster_sim/*_simulation_parameters.csv")
        .map { csv ->
            def base_name = csv.baseName.replaceAll(/_parameters$/, '')
            tuple(base_name, csv)
        }

    trajectories = Channel
        .fromPath("${params.outdir}/1_remaster_sim/*.traj")
        .map { traj -> tuple(traj.baseName, traj) }

    casecounts_ch = Channel
        .fromPath("${params.outdir}/1_remaster_sim/*_casecounts.csv")
        .map { cc ->
            def base_name = cc.baseName.replaceAll(/_casecounts$/, '')
            tuple(base_name, cc)
        }

    seroprevalence_ch = Channel
        .fromPath("${params.outdir}/1_remaster_sim/*_seroprevalence.csv")
        .map { sp ->
            def base_name = sp.baseName.replaceAll(/_seroprevalence$/, '')
            tuple(base_name, sp)
        }

    wastewater_ch = Channel
        .fromPath("${params.outdir}/1_remaster_sim/*_wastewater.csv")
        .map { ww ->
            def base_name = ww.baseName.replaceAll(/_wastewater$/, '')
            tuple(base_name, ww)
        }

    groundtruth = Channel
        .fromPath("${params.outdir}/1_remaster_sim/*_deme_switches_groundtruth.csv")
        .map { csv ->
            def base_name = csv.baseName.replaceAll(/_deme_switches_groundtruth$/, '')
            tuple(base_name, csv)
        }

    // ── Join simulation metadata per base_name ──────────────────────────
    sim_meta = sim_params
        .join(trajectories, by: 0)
        .join(casecounts_ch, by: 0)
        .join(seroprevalence_ch, by: 0)
        .join(wastewater_ch, by: 0)

    // ── Build base_inputs for each variant ──────────────────────────────
    // Real files for all variants — they serve as ground truth for analysis
    base_inputs = sim_meta.flatMap { base_name, params_csv, traj, cc, sp, ww ->
        def simNb = base_name.split('_')[0]
        [
            tuple(base_name, "original", params_csv, simNb, traj, cc, sp, ww),
            tuple(base_name, "datastreams", params_csv, simNb, traj, cc, sp, ww),
            tuple(base_name, "datastreams_nocasecounts", params_csv, simNb, traj, cc, sp, ww),
            tuple(base_name, "datastreams_noseroprevalence", params_csv, simNb, traj, cc, sp, ww),
            tuple(base_name, "datastreams_nowastewater", params_csv, simNb, traj, cc, sp, ww),
            tuple(base_name, "datastreams_nomascotll", params_csv, simNb, traj, cc, sp, ww),
            tuple(base_name, "datastreams_onlytree", params_csv, simNb, traj, cc, sp, ww),
        ]
    }

    // ── Extract and pair logs ───────────────────────────────────────────
    original_logs_by_base = combined_logs_ch
        .filter { xmlname, base_name, variant_type, logs -> variant_type == 'original' }
        .map { xmlname, base_name, variant_type, logs ->
            def logList = logs instanceof List ? logs : [logs]
            def mascot_log = logList.find { it.name.contains('mascot_logs') }
            tuple(base_name, mascot_log ?: emptyFile("${base_name}_original_empty_mascot.log"))
        }

    logs_by_variant = combined_logs_ch
        .filter { xmlname, base_name, variant_type, logs -> variant_type.startsWith('datastreams') }
        .map { xmlname, base_name, variant_type, logs ->
            def logList = logs instanceof List ? logs : [logs]
            tuple(
                base_name, variant_type,
                logList.find { it.name.contains('mascot_logs') }              ?: emptyFile("${base_name}_${variant_type}_empty_mascot.log"),
                logList.find { it.name.contains('NeDynamics.Deme1') }         ?: emptyFile("${base_name}_${variant_type}_empty_nedyn1.log"),
                logList.find { it.name.contains('NeDynamics.Deme2') }         ?: emptyFile("${base_name}_${variant_type}_empty_nedyn2.log"),
                logList.find { it.name.contains('cumulativeIncidence.Deme1') } ?: emptyFile("${base_name}_${variant_type}_empty_cuminc1.log"),
                logList.find { it.name.contains('cumulativeIncidence.Deme2') } ?: emptyFile("${base_name}_${variant_type}_empty_cuminc2.log")
            )
        }
        .combine(original_logs_by_base, by: 0)
        .map { base_name, variant_type, ds_log, nedyn1, nedyn2, cuminc1, cuminc2, orig_log ->
            tuple(base_name, variant_type, orig_log, ds_log, nedyn1, nedyn2, cuminc1, cuminc2)
        }

    // ── Build analysis inputs with meta map ─────────────────────────────
    analysis_inputs = logs_by_variant
        .join(base_inputs, by: [0, 1])
        .combine(groundtruth, by: 0)
        .map { base_name, variant_type, orig_log, ds_log, nedyn1, nedyn2, cuminc1, cuminc2,
               params_csv, simNb, traj, cc, sp, ww, deme_csv ->
            def meta = [base: base_name, variant: variant_type, simNb: simNb]
            tuple(meta, orig_log, ds_log, nedyn1, nedyn2, cuminc1, cuminc2,
                  params_csv, traj, cc, sp, ww, deme_csv)
        }

    analysis_inputs.count().view { n -> "ANALYSE_FROM_PUBLISHED: ${n} analysis inputs" }

    // ── Run analysis ────────────────────────────────────────────────────
    analysis_results = ANALYSE_POSTERIORS(analysis_inputs)

    grouped_by_type = analysis_results.hpd_validation_outputs
        .flatMap { base, variant, f_ci, f_ne, f_params, f_prev, f_migration_rates, f_original_ne ->
            def tuples = [
                tuple("cumulative_incidence", variant, f_ci),
                tuple("ne", variant, f_ne),
                tuple("params", variant, f_params),
                tuple("prevalence", variant, f_prev),
                tuple("migration_rates", variant, f_migration_rates)
            ]
            if (variant == 'datastreams') {
                tuples << tuple("ne", "original", f_original_ne)
            }
            tuples
        }
        .groupTuple(by: [0, 1])

    concatenated_csvs = CONCATENATE_HPD_VALIDATION(grouped_by_type)

    ch_prev   = concatenated_csvs.concatenated_csv.filter { it[0] == 'prevalence' }          .map { it -> tuple(it[1], it[2]) }
    ch_ne     = concatenated_csvs.concatenated_csv.filter { it[0] == 'ne' }                  .map { it -> tuple(it[1], it[2]) }
    ch_ci     = concatenated_csvs.concatenated_csv.filter { it[0] == 'cumulative_incidence' } .map { it -> tuple(it[1], it[2]) }
    ch_params = concatenated_csvs.concatenated_csv.filter { it[0] == 'params' }              .map { it -> tuple(it[1], it[2]) }
    ch_mig    = concatenated_csvs.concatenated_csv.filter { it[0] == 'migration_rates' }     .map { it -> tuple(it[1], it[2]) }

    plot_inputs = ch_prev
        .combine(ch_ne, by: 0)
        .combine(ch_ci, by: 0)
        .combine(ch_params, by: 0)
        .combine(ch_mig, by: 0)

    PLOT_HPD_VALIDATION(plot_inputs)

    // ── [2] Per-simulation publication figures ──────────────────────────
    MAKE_INDIVIDUAL_SIM_FIGURES(analysis_inputs)

    // ── [2] Combine Ne HPD validation: original + datastreams → Model column
    ne_original = concatenated_csvs.concatenated_csv
        .filter { it[0] == 'ne' && it[1] == 'original' }
        .map { it[2] }

    ne_datastreams = concatenated_csvs.concatenated_csv
        .filter { it[0] == 'ne' && it[1] == 'datastreams' }
        .map { it[2] }

    combine_ne_input = ne_original.combine(ne_datastreams)
    combined_ne = COMBINE_HPD_NE_BY_MODEL(combine_ne_input)

    // ── [2] Quantify information content across leave-one-out variants ──
    info_all_files = concatenated_csvs.concatenated_csv
        .filter { it[0] in ['params', 'prevalence', 'migration_rates'] && it[1].startsWith('datastreams') }
        .map { it[2] }
        .collect()

    QUANTIFY_INFORMATION_CONTENT(info_all_files)

    // ── [2] True vs estimated parameter figures ─────────────────────────
    fig_params_csv = concatenated_csvs.concatenated_csv
        .filter { it[0] == 'params' && it[1] == 'datastreams' }
        .map { it[2] }

    fig_mig_csv = concatenated_csvs.concatenated_csv
        .filter { it[0] == 'migration_rates' && it[1] == 'datastreams' }
        .map { it[2] }

    fig_prev_csv = concatenated_csvs.concatenated_csv
        .filter { it[0] == 'prevalence' && it[1] == 'datastreams' }
        .map { it[2] }

    // Read concatenated sim metadata from published results
    published_sim_metadata = Channel
        .fromPath("${params.outdir}/3_analysis/all_sim_metadata.csv")

    fig_truevsest_input = fig_params_csv
        .combine(fig_mig_csv)
        .combine(fig_prev_csv)
        .combine(combined_ne.combined_ne_csv)
        .combine(published_sim_metadata)

    MAKE_FIGURE_TRUE_VS_ESTIMATE(fig_truevsest_input)

    // ── ESS ─────────────────────────────────────────────────────────────
    ess_inputs = combined_logs_ch
        .map { xmlname, base_name, variant_type, logs ->
            def logList = logs instanceof List ? logs : [logs]
            def mascot_log = logList.find { it.name.contains('mascot_logs') }
            mascot_log != null
                ? tuple(xmlname + ".mascot_logs", variant_type, mascot_log)
                : null
        }
        .filter { it != null }

    ess_results = CALCULATE_ESS(ess_inputs)

    ess_csvs_collected = ess_results.ess_csv
        .groupTuple(by: 0)
        .map { variant, csvs -> tuple("ess", variant, csvs) }

    ess_summary = AGGREGATE_ESS(ess_csvs_collected)

    ess_summary_original = ess_summary.ess_summary
        .filter { variant, csv -> variant == 'original' }
        .map { variant, csv -> tuple('all', csv) }

    ess_summary_other = ess_summary.ess_summary
        .filter { variant, csv -> variant != 'original' }
        .map { variant, csv -> tuple('all', variant, csv) }

    ess_heatmap_inputs = ess_summary_original
        .combine(ess_summary_other, by: 0)
        .map { key, orig_csv, variant, var_csv ->
            tuple(variant, orig_csv, var_csv)
        }

    PLOT_ESS_HEATMAP(ess_heatmap_inputs)
}
