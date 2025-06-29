"""
Copyright (c) 2025 MyoLab, Inc
All Rights Reserved.

This software and associated documentation files (the "Software") are the intellectual property of MyoLab, Inc. Unauthorized copying, modification, distribution, or use of this code, in whole or in part, without express written permission from the copyright owner is strictly prohibited.
"""

# This file holds the body grouping tuples for the myoskeleton model
body_groupings = {
    "root": ("myoskeleton_root",),
    "pelvis": ("pelvis",),
    "lumbar_spine": (
        "lumbar_spine_1",
        "lumbar_spine_2",
        "lumbar_spine_3",
        "lumbar_spine_4",
        "lumbar_spine_5",
    ),
    "torso": ("thoracic_spine",),
    "shoulders": ("scapula_l", "scapula_r", "clavicle_l", "clavicle_r"),
    "neck": (
        "cervical_spine_1",
        "cervical_spine_2",
        "cervical_spine_3",
        "cervical_spine_4",
        "cervical_spine_5",
        "cervical_spine_6",
        "cervical_spine_7",
    ),
    "head": ("skull",),
    "upper_arm_right": ("humerus_r",),
    "lower_arm_right": ("ulna_r", "radius_r"),
    "palm_right": ("lunate_r",),
    "fingers_right": (
        "metacarpal_1_r",
        "proximal_phalanx_1_r",
        "distal_phalanx_1_r",
        "proximal_phalanx_2_r",
        "intermediate_phalanx_2_r",
        "distal_phalanx_2_r",
        "proximal_phalanx_3_r",
        "intermediate_phalanx_3_r",
        "distal_phalanx_3_r",
        "proximal_phalanx_4_r",
        "intermediate_phalanx_4_r",
        "distal_phalanx_4_r",
        "proximal_phalanx_5_r",
        "intermediate_phalanx_5_r",
        "distal_phalanx_5_r",
    ),
    "upper_arm_left": ("humerus_l",),
    "lower_arm_left": ("ulna_l", "radius_l"),
    "palm_left": ("lunate_l",),
    "fingers_left": (
        "metacarpal_1_l",
        "proximal_phalanx_1_l",
        "distal_phalanx_1_l",
        "proximal_phalanx_2_l",
        "intermediate_phalanx_2_l",
        "distal_phalanx_2_l",
        "proximal_phalanx_3_l",
        "intermediate_phalanx_3_l",
        "distal_phalanx_3_l",
        "proximal_phalanx_4_l",
        "intermediate_phalanx_4_l",
        "distal_phalanx_4_l",
        "proximal_phalanx_5_l",
        "intermediate_phalanx_5_l",
        "distal_phalanx_5_l",
    ),
    "thigh_right": ("femur_r",),
    "shank_right": ("tibia_r",),
    "foot_right": ("talus_r", "calcaneus_r", "toes_r"),
    "thigh_left": ("femur_l",),
    "shank_left": ("tibia_l",),
    "foot_left": ("talus_l", "calcaneus_l", "toes_l"),
}
