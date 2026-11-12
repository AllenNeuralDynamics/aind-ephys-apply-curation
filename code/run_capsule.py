"""Capsule to apply curation to ephys data and generate a new asset"""
import os
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path
import shutil

import numpy as np
import json
import zarr

import spikeinterface as si
from spikeinterface.curation import apply_curation
from spikeinterface.curation.curation_model import Curation

from aind_data_schema.core.processing import DataProcess, ProcessStage
from aind_data_schema.components.identifiers import Code
from aind_data_schema_models.process_names import ProcessName

data_folder = Path("../data/")
scratch_folder = Path("../scratch/")
results_folder = Path("../results/")

PARAMS_FILE = Path("../params.json")

URL = os.getenv("CODE_REPO", "https://github.com/AllenNeuralDynamics/aind-ephys-apply-curation")
VERSION = os.getenv("CODE_VERSION", "")


def restore_recording_provenance(input_analyzer_folder: Path, output_analyzer_folder: Path) -> None:
    """
    Copies the recording provenance of the input analyzer to the curated one.

    save_as writes the recording paths relative to this capsule's results folder. Once published as
    postprocessed/{recording_name}.zarr, the curated analyzer sits where the input one was, so the input
    (relative or remote) recording paths are the valid ones.
    """
    if not (input_analyzer_folder / "recording").is_dir():
        return
    if (input_analyzer_folder / ".zgroup").is_file() != (output_analyzer_folder / ".zgroup").is_file():
        logging.info("\tInput and output analyzers have different zarr formats: recording provenance not restored")
        return
    shutil.rmtree(output_analyzer_folder / "recording", ignore_errors=True)
    shutil.copytree(input_analyzer_folder / "recording", output_analyzer_folder / "recording")
    zarr.consolidate_metadata(str(output_analyzer_folder))


NULL_VALUES = {"i": -1, "u": 0, "b": False, "f": np.nan, "U": "", "S": b""}


def complete_unit_properties(input_sorting: si.BaseSorting, analyzer_folder: Path) -> None:
    """
    Gives every curated stream the unit properties of its input, plus is_merged / is_split.
    aind-units-nwb writes all streams in one units table, and an int or bool property missing in some streams
    has no null value to fill their rows with.
    - merges drop the int properties whose values differ in the merged units (e.g. original_cluster_id): they are
      restored, with the input values for the kept units and a null value for the new ones
    - is_merged / is_split are only set when a stream has merges / splits: they are set to False otherwise
    """
    analyzer = si.load(analyzer_folder, load_extensions=False)
    unit_ids = analyzer.unit_ids
    # new units are flagged by curation: their ids can reuse the ids of removed units
    kept = np.ones(len(unit_ids), dtype=bool)
    for key in ("is_merged", "is_split"):
        if key in analyzer.sorting.get_property_keys():
            kept &= ~analyzer.sorting.get_property(key).astype(bool)
    for key in input_sorting.get_property_keys():
        input_values = input_sorting.get_property(key)
        if key in analyzer.sorting.get_property_keys() or input_values.dtype.kind not in NULL_VALUES:
            continue
        values = np.full((len(unit_ids),) + input_values.shape[1:], NULL_VALUES[input_values.dtype.kind], dtype=input_values.dtype)
        values[kept] = input_values[input_sorting.ids_to_indices(unit_ids[kept])]
        analyzer.set_sorting_property(key, values)
    for key in ("is_merged", "is_split"):
        if key not in analyzer.sorting.get_property_keys():
            analyzer.set_sorting_property(key, np.zeros(len(unit_ids), dtype=bool))


def run() -> None:
    """
    Applies noise removal and the curation.json merges to each postprocessed analyzer.

    Inputs: a sorted asset (or, in the pipeline, one stream of it) with postprocessed/{recording_name}.zarr
    and curated/{recording_name}/curation.json.
    Outputs: postprocessed_{recording_name}.zarr and the applied curation_{recording_name}.json,
    named like the outputs of the postprocessing and curation steps.
    """
    if not PARAMS_FILE.is_file():
        raise ValueError(f"Params file not found: {PARAMS_FILE}")
    with open(PARAMS_FILE, "r") as f:
        full_params = json.load(f)
    apply_curation_params = full_params["apply_curation"]
    LOGGING = full_params.get("logging", None)

    # setup logging before any other logging call
    if LOGGING is None:
        logging.basicConfig(level="INFO", stream=sys.stdout, format="%(message)s")
    elif LOGGING["package"] == "logging":
        logging.basicConfig(stream=sys.stdout, **LOGGING.get("logging_cfg", {}))
    elif LOGGING["package"] == "log-schema":
        import log_schema

        config = LOGGING.get("logging_cfg")
        if config is not None and len(config) == 0:
            config = None
        log_schema.setup_logging(
            config=config,
            model={
                "pipeline_name": LOGGING.get("pipeline_name", "AIND Ephys Pipeline"),
                "acquisition_name": LOGGING.get("acquisition_name", "unknown"),
                "process_name": "Apply curation",
            },
        )
    logging.info("Begin processing...", extra={"event_type": "stage_start"})
    # noise_removal_strategy: 'unitrefine', 'bombcell', or both combined with & (both noise) or | (either noise)
    noise_removal_strategy = apply_curation_params.get("noise_removal_strategy", "unitrefine&bombcell")
    censor_ms = float(apply_curation_params.get("censor_ms", 0.5))
    merging_mode = apply_curation_params.get("merging_mode", "soft")
    if merging_mode not in ("soft", "hard"):
        raise ValueError(f"Invalid merging mode: {merging_mode}")

    # assert that noise removal strategy is valid
    valid_strategies = ["unitrefine", "bombcell"]
    strategies = noise_removal_strategy.replace("&", "|").split("|")
    for strategy in strategies:
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid noise removal strategy: {strategy}")

    logging.info(f"Noise removal strategy: {noise_removal_strategy}")
    logging.info(f"Censor period (ms): {censor_ms}")
    logging.info(f"Merging mode: {merging_mode}")

    # Find sorted folder as the parent of the postprocessed folder
    ecephys_sorted_folders = [f.parent for f in data_folder.glob("**/postprocessed") if f.is_dir()]
    if len(ecephys_sorted_folders) == 0:
        raise FileNotFoundError("No ecephys sorted folders found")
    elif len(ecephys_sorted_folders) > 1:
        raise ValueError("Multiple ecephys sorted folders found")
    ecephys_sorted_folder = ecephys_sorted_folders[0]

    # Now loop through streams in the postprocessed folder
    postprocessed_folders = [
        p for p in (ecephys_sorted_folder / "postprocessed").iterdir()
        if p.is_dir() and p.name.endswith(".zarr")
    ]
    for postprocessed_folder in postprocessed_folders:
        recording_name = postprocessed_folder.name[:-5]  # Remove the ".zarr" suffix
        logging.info(recording_name)
        output_folder = results_folder / f"postprocessed_{recording_name}.zarr"

        # Look for curation file
        curation_file = ecephys_sorted_folder / "curated" / recording_name / "curation.json"
        if not curation_file.is_file():
            logging.info(f"No curation file found for {recording_name}: copying original analyzer")
            shutil.copytree(postprocessed_folder, output_folder)
            complete_unit_properties(si.load(output_folder, load_extensions=False).sorting, output_folder)
            continue
        logging.info(f"Found curation file for {recording_name}: {curation_file}")

        datetime_start = datetime.now(timezone.utc)
        analyzer = si.load(postprocessed_folder)
        if "unitrefine_label" not in analyzer.get_sorting_property_keys():
            if "decoder_label" in analyzer.get_sorting_property_keys():
                analyzer.set_unit_property("unitrefine_label", analyzer.get_unit_property("decoder_label"))

        label_masks = [
            analyzer.get_sorting_property(f"{strategy}_label") == "noise"
            for strategy in strategies
            if f"{strategy}_label" in analyzer.get_sorting_property_keys()
        ]
        if len(label_masks) < len(strategies):
            logging.info(f"\tMissing noise labels for {recording_name}: using {len(label_masks)}/{len(strategies)} labels")
        if len(label_masks) == 0:
            mask = np.zeros(analyzer.get_num_units(), dtype=bool)
        elif "|" in noise_removal_strategy:
            mask = np.any(label_masks, axis=0)
        else:
            mask = np.all(label_masks, axis=0)

        removed_unit_ids = analyzer.unit_ids[mask]

        # Now get curation
        with open(curation_file, "r") as f:
            curation_dict = json.load(f)
        curation = Curation(**curation_dict)
        curation.removed = removed_unit_ids.tolist()
        # Filter merges with removed units
        curation.merges = [
            merge for merge in (curation.merges or [])
            if not any(unit in removed_unit_ids for unit in merge.unit_ids)
        ]
        logging.info(f"\tCuration for {recording_name}: {len(curation.removed)} removed, {len(curation.merges)} merges")

        analyzer_curated = apply_curation(analyzer, curation, merging_mode=merging_mode, censor_ms=censor_ms, raise_error_if_overlap_fails=False)
        logging.info(f"\tCurated units for {recording_name}: {analyzer_curated.get_num_units()}")
        analyzer_curated.save_as(format="zarr", folder=output_folder)
        restore_recording_provenance(postprocessed_folder, output_folder)
        complete_unit_properties(analyzer.sorting, output_folder)

        # the curation that was actually applied, used by processing QC
        with open(results_folder / f"curation_{recording_name}.json", "w") as f:
            f.write(curation.model_dump_json(indent=4))

        # appended to processing.json by the pipeline
        apply_curation_process = DataProcess(
            process_type=ProcessName.EPHYS_CURATION,
            stage=ProcessStage.PROCESSING,
            name=f"Apply curation - {recording_name}",
            experimenters=["AIND Pipeline"],
            code=Code(url=URL, version=VERSION, parameters=apply_curation_params),
            start_date_time=datetime_start,
            end_date_time=datetime.now(timezone.utc),
            output_path=str(results_folder),
            output_parameters=dict(
                num_units_before=int(analyzer.get_num_units()),
                num_units_after=int(analyzer_curated.get_num_units()),
                num_removed=len(curation.removed),
                num_merges=len(curation.merges),
            ),
        )
        with open(results_folder / f"data_process_apply_curation_{recording_name}.json", "w") as f:
            f.write(apply_curation_process.model_dump_json(indent=3))

    logging.info("Pipeline stage completed", extra={"event_type": "stage_complete"})


if __name__ == "__main__":
    try:
        run()
    except Exception:
        logging.exception("Pipeline stage failed", extra={"event_type": "stage_error"})
        raise
