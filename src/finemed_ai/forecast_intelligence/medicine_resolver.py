from __future__ import annotations

import re

import pandas as pd

from finemed_ai.forecast_intelligence.schemas import MedicineResolution


class MedicineResolver:
    """
    Resolves employee-provided medicine references to canonical Medicine_IDs.

    Supported inputs:
    - Exact medicine code: "0001"
    - Exact medicine name
    - Exact product display name
    - Partial medicine name

    Ambiguous matches are never silently resolved.
    """

    CODE_COLUMN = "MDCODE"
    NAME_COLUMNS = ("MDNAME", "Product_Display_Name")

    def __init__(self, medicines: pd.DataFrame) -> None:
        if medicines is None or medicines.empty:
            self.medicines = pd.DataFrame(columns=[self.CODE_COLUMN] + list(self.NAME_COLUMNS))
            self._code_index = {}
            self._display_name_index = {}
            self._normalized_index = {}
            return


        required_columns = {self.CODE_COLUMN}
        missing = required_columns - set(medicines.columns)

        if missing:
            raise ValueError(
                f"Medicine master missing required columns: {sorted(missing)}"
            )

        self.medicines = medicines.copy()

        self.medicines[self.CODE_COLUMN] = (
            self.medicines[self.CODE_COLUMN]
            .astype(str)
            .str.strip()
            .str.zfill(4)
        )

        for column in self.NAME_COLUMNS:
            if column in self.medicines.columns:
                self.medicines[column] = (
                    self.medicines[column]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                )

    @staticmethod
    def _normalise(text: str) -> str:
        """
        Normalise employee input for case-insensitive matching.
        """

        text = str(text).strip().lower()
        text = re.sub(r"\s+", " ", text)

        return text

    def resolve(self, query: str) -> MedicineResolution:
        """
        Resolve a medicine query to a canonical medicine code.
        """

        if not isinstance(query, str) or not query.strip():
            return MedicineResolution(
                query=str(query),
                resolved=False,
                match_type="not_found",
            )

        raw_query = query.strip()
        normalised_query = self._normalise(raw_query)

        # ---------------------------------------------------------
        # 1. Exact medicine code
        # ---------------------------------------------------------

        if normalised_query.isdigit():
            medicine_code = normalised_query.zfill(4)

            matches = self.medicines[
                self.medicines[self.CODE_COLUMN] == medicine_code
            ]

            if len(matches) == 1:
                row = matches.iloc[0]

                return MedicineResolution(
                    query=raw_query,
                    resolved=True,
                    medicine_code=medicine_code,
                    medicine_name=self._get_display_name(row),
                    match_type="exact_code",
                )

        # ---------------------------------------------------------
        # 2. Exact name / display-name match
        # ---------------------------------------------------------

        exact_matches = self._find_name_matches(
            normalised_query,
            exact=True,
        )

        if len(exact_matches) == 1:
            row = exact_matches.iloc[0]

            return MedicineResolution(
                query=raw_query,
                resolved=True,
                medicine_code=str(row[self.CODE_COLUMN]),
                medicine_name=self._get_display_name(row),
                match_type="exact_name",
            )

        if len(exact_matches) > 1:
            return MedicineResolution(
                query=raw_query,
                resolved=False,
                match_type="ambiguous",
            )

        # ---------------------------------------------------------
        # 3. Partial name match
        # ---------------------------------------------------------

        partial_matches = self._find_name_matches(
            normalised_query,
            exact=False,
        )

        if len(partial_matches) == 1:
            row = partial_matches.iloc[0]

            return MedicineResolution(
                query=raw_query,
                resolved=True,
                medicine_code=str(row[self.CODE_COLUMN]),
                medicine_name=self._get_display_name(row),
                match_type="partial_name",
            )

        if len(partial_matches) > 1:
            return MedicineResolution(
                query=raw_query,
                resolved=False,
                match_type="ambiguous",
            )

        return MedicineResolution(
            query=raw_query,
            resolved=False,
            match_type="not_found",
        )

    def _find_name_matches(
        self,
        query: str,
        *,
        exact: bool,
    ) -> pd.DataFrame:
        """
        Search across available medicine-name columns.
        """

        masks: list[pd.Series] = []

        for column in self.NAME_COLUMNS:
            if column not in self.medicines.columns:
                continue

            normalised = (
                self.medicines[column]
                .fillna("")
                .astype(str)
                .map(self._normalise)
            )

            if exact:
                mask = normalised == query
            else:
                mask = normalised.str.contains(
                    query,
                    regex=False,
                    na=False,
                )

            masks.append(mask)

        if not masks:
            return self.medicines.iloc[0:0].copy()

        combined_mask = masks[0]

        for mask in masks[1:]:
            combined_mask = combined_mask | mask

        return (
            self.medicines[combined_mask]
            .drop_duplicates(subset=[self.CODE_COLUMN])
            .copy()
        )

    @staticmethod
    def _get_display_name(row: pd.Series) -> str | None:
        """
        Prefer Product_Display_Name, then MDNAME.
        """

        display_name = str(
            row.get("Product_Display_Name", "")
        ).strip()

        if display_name:
            return display_name

        medicine_name = str(
            row.get("MDNAME", "")
        ).strip()

        return medicine_name or None