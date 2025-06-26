"""
Copyright (c) 2025 MyoLab, Inc
All Rights Reserved.

This software and associated documentation files (the "Software") are the intellectual property of MyoLab, Inc. Unauthorized copying, modification, distribution, or use of this code, in whole or in part, without express written permission from the copyright owner is strictly prohibited.
"""

import os
import textwrap

import mujoco
from myo_api.mj.core import mj_api

from myo_model.myoskeleton.grouping import measures

curr_dir = os.path.dirname(os.path.realpath(__file__))


def get_model_xml_path(type=None):
    """
    Get the path to the model xml file
    Args:
        type (str): Type of model to retrieve.
                    If None, returns the default model path.
                    Other types: muscles, motors, etc.
    Returns:
        str: Path to the model xml file.
    """
    # Defaults to the basic model
    model_path = os.path.join(curr_dir, "../myoskeleton/myoskeleton.xml")

    # Check if the type is valid
    if type not in [None, "muscles", "motors", "skin"]:
        raise ValueError(
            f"Invalid type '{type}'. Valid types are: None, 'muscles', 'motors', 'skin'."
        )
    if type == "muscles":
        model_path = os.path.join(
            curr_dir, "../myoskeleton/myoskeleton_with_muscles.xml"
        )
    elif type == "motors":
        model_path = os.path.join(
            curr_dir, "../myoskeleton/myoskeleton_with_motors.xml"
        )
    elif type == "skin":
        model_path = os.path.join(curr_dir, "../myoskeleton/myoskeleton_with_skin.xml")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path '{model_path}' does not exist.")

    return model_path


def get_assets_path():
    """
    Get the path to the assets directory
    Returns:
        str: Path to the assets directory.
    """
    assets_path = os.path.join(curr_dir, "../myoskeleton/meshes")
    if not os.path.exists(assets_path):
        raise FileNotFoundError(f"Assets path '{assets_path}' does not exist.")
    return assets_path


def get_markerset_path(markerset_name="metrabs"):
    """
    Get the path to the markerset file
    Args:
        markerset_name (str): Name of the markerset to retrieve.
                              Default is "metrabs".
    Returns:
        str: Path to the markerset file.
    """
    markerset_dict = {
        "metrabs": "movi_metrabs_markerset.xml",
        "accad": "accad_markerset.xml",
        "cmu": "cmu_markerset.xml",
        "skel": "skel_markerset.xml",
        "smpl": "smpl_markerset.xml",
        "xsens": "xsens_markerset.xml",
        "mecka": "mecka_markerset.xml",
        "moyo": "moyo_markerset.xml",
        "body_origin": "body_origin_markerset.xml",
    }

    markerset_path = os.path.join(
        curr_dir, "../markerset/", markerset_dict[markerset_name]
    )
    if not os.path.exists(markerset_path):
        raise FileNotFoundError(f"Markerset path '{markerset_path}' does not exist.")
    return markerset_path


def measure_model(model=None, output_file=None):
    """
    Measure the model to get the size of different body parts.
    Args:
        model: The model to measure.
    Returns:
        dict: A dictionary with the size of body parts
        - `upper_arm_length`: Length of the upper arms.
        - `lower_arm_length`: Length of the lower arms.
        - `shoulder_width`: Width of the shoulders.
        - `wrist_to_wrist_distance`: Length from wrist to wrist.
        - `hip_width`: Width of the hips.
        - `neck_length`: Length of the neck.
        - `skull_height`: Height of the skull.
        - `ribcage_height`: Height of the ribcage.
        - `lumbar_length`: Length of the lumbar section.
        - `upper_leg_length`: Length of the upper legs.
        - `lower_leg_length`: Length of the lower legs.
        - `foot_length`: Length of the foot.
        - `total_height`: Total height of the body.
    """
    if model is None:
        model = get_model_xml_path()
    body_dimensions = {}

    # Load the model
    mjc_model, _ = mj_api.get_model(model)
    mjc_data = mujoco.MjData(mjc_model)

    # Run mj forward to populate data
    mujoco.mj_forward(mjc_model, mjc_data)

    # Get the measuremenet pairs
    measurement_pairs = measures.measurement_pairs

    # Measure the model
    for measure in measurement_pairs.keys():
        body_dimensions[measure] = mj_api.get_jnt_dist(
            mjc_data, measurement_pairs[measure][0], measurement_pairs[measure][1]
        )

    # Set the wrist to wrist distance
    body_dimensions["wrist_to_wrist_distance"] = (
        2 * body_dimensions["upper_arm_length"]
        + 2 * body_dimensions["lower_arm_length"]
        + body_dimensions["shoulder_width"]
    )

    # Set the total height
    body_dimensions["total_height"] = 1.0

    # Set the skull height
    body_dimensions["skull_height"] = 1.0

    # Write the measurements to a python file if file name is provided
    if output_file:
        HEADER = '''
        """Copyright (c) 2025 MyoLab, Inc
        All Rights Reserved.
        This software and associated documentation files (the "Software") are the intellectual property of MyoLab, Inc. Unauthorized copying, modification, distribution, or use of this code, in whole or in part, without express written permission from the copyright owner is strictly prohibited.
        """
        '''
        DOCSTRING = """
        # This file holds the measurements for the current myoskeleton model
        # All measurements are in meters
        # Measurements are generated using model_utils.measure_model(model_path, output_path)
        """

        with open(output_file, "w") as f:
            f.write(textwrap.dedent(HEADER))
            f.write(textwrap.dedent(DOCSTRING))
            f.write("body_dimensions = {\n")
            for key, value in body_dimensions.items():
                f.write(f"    '{key}': {value},\n")
            f.write("}\n")

    return body_dimensions
