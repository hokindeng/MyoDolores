"""
Copyright (c) 2025 MyoLab, Inc
All Rights Reserved.

This software and associated documentation files (the "Software") are the intellectual property of MyoLab, Inc. Unauthorized copying, modification, distribution, or use of this code, in whole or in part, without express written permission from the copyright owner is strictly prohibited.
"""

# This file holds the body measurement pairs for the myoskeleton model
# Calculate distances using:
# dist = mj_api.get_jnt_dist(mjc_data, measurement_pairs[dist_name][0], measurement_pairs[dist_name][1])
# wrist_to_wrist = (lower_arm_length + upper_arm_length
#                   + shoulder_width + lower_arm_length
#                   + upper_arm_length)

measurement_pairs = {
    "upper_arm_length": ("humerus_elev_r", "elbow_flex_r"),
    "lower_arm_length": ("elbow_flex_r", "wrist_flex_r"),
    "shoulder_width": ("humerus_elev_r", "humerus_elev_l"),
    "hip_width": ("hip_flex_r", "hip_flex_l"),
    "neck_length": ("occ_c1_flex", "c6_c7_flex"),
    "ribcage_height": ("c6_c7_flex", "t12_l1_flex"),
    "lumbar_length": ("t12_l1_flex", "l5_s1_flex"),
    "upper_leg": ("hip_flex_r", "knee_flex_r"),
    "lower_leg": ("knee_flex_r", "ankle_flex_r"),
    "foot_length": ("subtal_inve_r", "mtp_flex_r"),
}
