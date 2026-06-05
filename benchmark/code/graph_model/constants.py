"""Constants used throughout the code"""
import re
import uuid

CREMAD_DATASET = "CREMAD"
RAVDESS_DATASET = "RAVDESS"
CREMAD_AND_RAVDESS_DATASETS = "CREMAD_AND_RAVDESS"
_RAVDESS_RE = re.compile(r"^Actor_\d{2}$")   # e.g., Actor_01
_CREMA_RE   = re.compile(r"^C\d{4}$")        # e.g., C1022
_CREMAD_RAVDESS_RE = re.compile(r"^(Actor_\d{2}|C\d{4})$")

AVAILABLE_DATASETS = [CREMAD_DATASET, RAVDESS_DATASET, CREMAD_AND_RAVDESS_DATASETS]
"""List of dataset names available"""

VAL_IDS_CREMAD = [
    'C1001', 
    'C1010', 
    'C1011', 
    'C1016', 
    'C1021', 
    'C1024', 
    'C1030', 
    'C1032', 
    'C1037', 
    'C1040', 
    'C1050', 
    'C1053', 
    'C1055', 
    'C1057', 
    'C1063', 
    'C1069', 
    'C1075', 
    'C1086',
]
"""Validation IDs for the CREMA-D dataset"""

VAL_IDS_RAVDESS = [
    "Actor_03",
    "Actor_13",
    "Actor_19",
    "Actor_23",
]
"""Validation IDs for the RAVDESS dataset"""


ALL_VALIDATION_IDS = VAL_IDS_CREMAD + VAL_IDS_RAVDESS
"""Validation IDs for both datasets"""

LANDMARK_IDS = [
        46,53,52,65,55,285,295,282,283,276,70,63,105,66,107,
        336,296,334,293,300,33,161,160,159,158,157,
        133,154,153,145,144,468,362,384,385,
        386,387,388,263,373,374,380,381,473,
        205,50,425,64,294,
        280,9,168,5,4,19,185,40,39,37,
        0,267,269,270,409,191,80,82,13,312,310,415,95,
        88,178,87,14,317,402,318,324,146,91,181,84,17,314,
        405,321,375,61,291,10,297,284,389,454,
        361,397,379,400,152,
        176,150,172,132,234,162,54,
        67
    ]


FILENAME_SEPARATOR = "--"
NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")  # fixed namespace

ALLOWED_GENERATORS = ["GAGA", "LIVE", "HUNY", "ORIG"]