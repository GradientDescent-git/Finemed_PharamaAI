from __future__ import annotations

# Required ERP Tables

REQUIRED_FILES = [
    "INVOICE.DAT",
    "INVDET.DAT",
    "MEDIMAST.DAT",
    "PURCHASE.DAT",
    "COMPUR.DAT",
    "SUPMAST.DAT",
    "SFILE.DAT",
    "TFILE.DAT",
    "SFILE.DAT",
    "TFILE.DAT"
]

# Duplicate Validation Keys

DUPLICATE_KEYS = {

    "INVOICE.DAT": [
        "SOURCE_MONTH",
        "INVNO",
    ],

    "INVDET.DAT": [
        "SOURCE_MONTH",
        "INVNO",
        "MDCODE",
        "BATCH",
    ],

    "MEDIMAST.DAT": [
        "SOURCE_MONTH",
        "MDCODE",
    ],

    "PURCHASE.DAT": [
        "SOURCE_MONTH",
        "PINVNO",
        "MDCODE",
        "BAT",
    ],

    "COMPUR.DAT": [
        "SOURCE_MONTH",
        "PINVNO",
    ],

    "SUPMAST.DAT": [
        "SOURCE_MONTH",
        "SUPNO",
    ],

    "SFILE.DAT": [
        "SOURCE_MONTH",
        "SCODE",
    ],

    "TFILE.DAT": [
        "SOURCE_MONTH",
        "TCODE",
    ],
}

# Parent → Child Relationships

ORPHAN_CHECKS = [

    {
        "check_name": "INVDET invoice exists in INVOICE",
        "parent_table": "INVOICE.DAT",
        "child_table": "INVDET.DAT",
        "parent_key": "INVNO",
        "child_key": "INVNO",
    },

    {
        "check_name": "INVDET medicine exists in MEDIMAST",
        "parent_table": "MEDIMAST.DAT",
        "child_table": "INVDET.DAT",
        "parent_key": "MDCODE",
        "child_key": "MDCODE",
    },

    {
        "check_name": "PURCHASE medicine exists in MEDIMAST",
        "parent_table": "MEDIMAST.DAT",
        "child_table": "PURCHASE.DAT",
        "parent_key": "MDCODE",
        "child_key": "MDCODE",
    },

    {
        "check_name": "PURCHASE purchase invoice exists in COMPUR",
        "parent_table": "COMPUR.DAT",
        "child_table": "PURCHASE.DAT",
        "parent_key": "PINVNO",
        "child_key": "PINVNO",
    },

    {
        "check_name": "COMPUR supplier exists in SUPMAST",
        "parent_table": "SUPMAST.DAT",
        "child_table": "COMPUR.DAT",
        "parent_key": "SUPNO",
        "child_key": "SUPNO",
    },

    {
        "check_name": "INVOICE salesperson exists in SFILE",
        "parent_table": "SFILE.DAT",
        "child_table": "INVOICE.DAT",
        "parent_key": "SCODE",
        "child_key": "SCODE",
    },

    {
        "check_name": "INVDET tax code exists in TFILE",
        "parent_table": "TFILE.DAT",
        "child_table": "INVDET.DAT",
        "parent_key": "TCODE",
        "child_key": "TCODE",
    },
]