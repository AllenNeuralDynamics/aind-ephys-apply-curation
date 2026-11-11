"""Capsule to apply curation to ephys data and generate a new asset"""
from pathlib import Path
import shutil

import numpy as np
import json
import zarr

import spikeinterface as si
from spikeinterface.curation import apply_curation
from spikeinterface.curation.curation_model import Curation

data_folder = Path("../data/")
scratch_folder = Path("../scratch/")
results_folder = Path("../results/")

PARAMS_FILE = Path("../params.json")


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
        print("\tInput and output analyzers have different zarr formats: recording provenance not restored")
        return
    shutil.rmtree(output_analyzer_folder / "recording", ignore_errors=True)
    shutil.copytree(input_analyzer_folder / "recording", output_analyzer_folder / "recording")
    zarr.consolidate_metadata(str(output_analyzer_folder))


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
        apply_curation_params = json.load(f)["apply_curation"]
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

    print(f"Noise removal strategy: {noise_removal_strategy}")
    print(f"Censor period (ms): {censor_ms}")
    print(f"Merging mode: {merging_mode}")

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
        print(recording_name)
        output_folder = results_folder / f"postprocessed_{recording_name}.zarr"

        # Look for curation file
        curation_file = ecephys_sorted_folder / "curated" / recording_name / "curation.json"
        if not curation_file.is_file():
            print(f"No curation file found for {recording_name}: copying original analyzer")
            shutil.copytree(postprocessed_folder, output_folder)
            continue
        print(f"Found curation file for {recording_name}: {curation_file}")

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
            print(f"\tMissing noise labels for {recording_name}: using {len(label_masks)}/{len(strategies)} labels")
        if len(label_masks) == 0:
            mask = np.zeros(analyzer.get_num_units(), dtype=bool)
        elif "|" in noise_removal_strategy:
            mask = np.any(label_masks, axis=0)
        else:
            mask = np.all(label_masks, axis=0)

        removed_unit_ids = analyzer.unit_ids[mask]
        print(f"\tNoise units for {recording_name}: {len(removed_unit_ids)}/{analyzer.get_num_units()}")

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

        analyzer_curated = apply_curation(analyzer, curation, merging_mode=merging_mode, censor_ms=censor_ms)
        print(f"\tCurated units for {recording_name}: {analyzer_curated.get_num_units()}")
        analyzer_curated.save_as(format="zarr", folder=output_folder)
        restore_recording_provenance(postprocessed_folder, output_folder)

        # the curation that was actually applied, used by processing QC
        with open(results_folder / f"curation_{recording_name}.json", "w") as f:
            f.write(curation.model_dump_json(indent=4))


if __name__ == "__main__":
    run()
