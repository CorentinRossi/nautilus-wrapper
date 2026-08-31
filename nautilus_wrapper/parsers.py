"""Parsers for nautilus input files.

Each parser returns a list of dicts (rows) and a list of column headers.
The serialize function converts them back to the original text format.
"""

import re
from typing import Any


# ---------------------------------------------------------------------------
# parameters.in
# ---------------------------------------------------------------------------
def parse_parameters(text: str) -> tuple[list[dict], list[str]]:
    headers = ["Key", "Value", "Comment"]
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        comment = ""
        idx = stripped.find("!")
        if idx >= 0:
            comment = stripped[idx + 1:].strip()
            stripped = stripped[:idx].strip()
        if "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        rows.append({"Key": key.strip(), "Value": value.strip(), "Comment": comment})
    return rows, headers


def serialize_parameters(rows: list[dict]) -> str:
    lines = []
    for r in rows:
        part = f"{r['Key']} = {r['Value']}"
        if r.get("Comment"):
            part += f" ! {r['Comment']}"
        lines.append(part)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# element.in
# ---------------------------------------------------------------------------
def parse_element(text: str) -> tuple[list[dict], list[str]]:
    headers = ["Element", "Mass"]
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        parts = stripped.split()
        if len(parts) >= 2:
            rows.append({"Element": parts[0], "Mass": parts[1]})
    return rows, headers


def serialize_element(rows: list[dict]) -> str:
    lines = ["! Species name ; mass (AMU)"]
    for r in rows:
        lines.append(f"{r['Element']:12s}{r['Mass']}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# abundances.in
# ---------------------------------------------------------------------------
def _format_abundance(value: str) -> str:
    """Normalise an abundance to 3-decimal scientific notation, e.g. ``9.000E-02``.

    Accepts ``D``/``E``/plain-decimal input. Nautilus reads this column with
    ``(e12.6)``, and ``E`` is its native exponent letter. Text that is not a
    number is left as typed.
    """
    text = value.strip()
    if not text:
        return text
    try:
        number = float(text.replace("D", "e").replace("d", "e"))
    except ValueError:
        return text
    return f"{number:.3e}".replace("e", "E")


def parse_abundances(text: str) -> tuple[list[dict], list[str]]:
    headers = ["Species", "Abundance", "Comment"]
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        comment = ""
        idx = stripped.find("!")
        if idx >= 0:
            comment = stripped[idx + 1:].strip()
            stripped = stripped[:idx].strip()
        if "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        rows.append({
            "Species": key.strip(),
            "Abundance": _format_abundance(value.strip()),
            "Comment": comment,
        })
    return rows, headers


def serialize_abundances(rows: list[dict]) -> str:
    lines = ["! Abundances in relative abundances with respect to Hydrogen"]
    for r in rows:
        part = f"{r['Species']:12s}= {_format_abundance(r['Abundance'])}"
        if r.get("Comment"):
            part += f" ! {r['Comment']}"
        lines.append(part)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# gas_species.in / grain_species.in
# ---------------------------------------------------------------------------
ELEMENT_COLS = ["charge", "H", "He", "C", "N", "O", "Si", "S", "Fe", "Na", "Mg", "Cl", "P", "F"]


SPECIES_HEADERS = ["Species"] + ELEMENT_COLS + ["_comment"]


def parse_species(text: str) -> tuple[list[dict], list[str]]:
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        comment = ""
        idx = stripped.find("!")
        if idx >= 0:
            comment = stripped[idx + 1:].strip()
            stripped = stripped[:idx].strip()
        parts = stripped.split()
        if len(parts) < 2:
            continue
        name = parts[0]
        nums = parts[1:]
        row = {"Species": name, "_comment": comment}
        for i, col in enumerate(ELEMENT_COLS):
            row[col] = nums[i] if i < len(nums) else "0"
        rows.append(row)
    return rows, SPECIES_HEADERS


def serialize_species(rows: list[dict]) -> str:
    """Serialize species rows to the fixed-width layout Nautilus expects.

    The Fortran reader uses ``(A11, i3, 13(I3))`` (see
    ``read_species`` in ``src/nautilus_core/input_output.f90``): an 11-char
    label followed by 14 back-to-back 3-char integer fields with *no*
    separators. Emitting a space between fields (as the previous version did)
    shifts every column and makes the file unreadable.
    """
    lines = []
    for r in rows:
        line = f"{r['Species']:<11.11}" + "".join(
            f"{r.get(col, '0'):>3.3}" for col in ELEMENT_COLS
        )
        if r.get("_comment"):
            line += f"    ! {r['_comment']}"
        lines.append(line)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# gas_reactions.in / grain_reactions.in
# ---------------------------------------------------------------------------
REACTION_COLS = [
    "Reactant1", "Reactant2", "Reactant3",
    "Product1", "Product2", "Product3", "Product4", "Product5",
    "A", "B", "C",
    "ITYPE", "Tmin", "Tmax", "Formula", "ID",
]

# gas_reactions.in / grain_reactions.in use a strict fixed-width layout. The
# Fortran reader (``read_reactions`` in ``src/nautilus_core/input_output.f90``)
# builds the format ``(3A11, 1x, 5A11, 3D11.3, 23x, I3, 2I7, I3, I6)``.
#
# The 23-char block between C and ITYPE (branching exponents + a formula
# keyword like ``logn``/``NA``) and everything past the ID field (KIDA range
# flags, or a free-text reference on some networks) are *not* consumed by that
# read but are meaningful and vary line to line. We keep them verbatim in the
# hidden ``_mid`` / ``_tail`` fields and splice them back on serialization so a
# load/save round-trip is byte-for-byte identical.
_REACTION_FIELDS: dict[str, tuple[int, int]] = {
    "Reactant1": (0, 11), "Reactant2": (11, 22), "Reactant3": (22, 33),
    "Product1": (34, 45), "Product2": (45, 56), "Product3": (56, 67),
    "Product4": (67, 78), "Product5": (78, 89),
    "A": (89, 100), "B": (100, 111), "C": (111, 122),
    "ITYPE": (145, 148), "Tmin": (148, 155), "Tmax": (155, 162),
    "Formula": (162, 165), "ID": (165, 171),
}
_REACTION_MID = (122, 145)
_REACTION_TAIL_START = 171

REACTION_HEADERS = REACTION_COLS + ["_mid", "_tail"]

_REACTION_HEADER_LINE = (
    "!    Reactants                    Products                "
    "                                  A          B          C      "
    "xxxxxxxxxxxxxxxxxxxxx ITYPE Tmin   Tmax formula ID xxxxx"
)


def _parse_reaction_line(line: str) -> dict | None:
    """Parse a single reaction line using the fixed-width Fortran layout."""
    idx = line.find("!")
    if idx >= 0:
        line = line[:idx]
    if not line.strip():
        return None

    row = {name: line[a:b].strip() for name, (a, b) in _REACTION_FIELDS.items()}
    row["_mid"] = line[_REACTION_MID[0]:_REACTION_MID[1]]
    row["_tail"] = line[_REACTION_TAIL_START:]
    return row


def parse_reactions(text: str) -> tuple[list[dict], list[str]]:
    rows = []
    for line in text.splitlines():
        row = _parse_reaction_line(line)
        if row is not None:
            rows.append(row)
    return rows, REACTION_HEADERS


def serialize_reactions(rows: list[dict]) -> str:
    """Serialize reaction rows back to the exact fixed-width layout."""
    lines = [_REACTION_HEADER_LINE]
    for r in rows:
        mid = r.get("_mid", "")
        mid = mid[:23] if len(mid) >= 23 else mid.ljust(23)
        tail = r.get("_tail", "") or " 1  1"
        line = (
            f"{r.get('Reactant1', ''):<11.11}{r.get('Reactant2', ''):<11.11}"
            f"{r.get('Reactant3', ''):<11.11} "
            f"{r.get('Product1', ''):<11.11}{r.get('Product2', ''):<11.11}"
            f"{r.get('Product3', ''):<11.11}{r.get('Product4', ''):<11.11}"
            f"{r.get('Product5', ''):<11.11}"
            f"{r.get('A', ''):>11.11}{r.get('B', ''):>11.11}{r.get('C', ''):>11.11}"
            f"{mid}"
            f"{r.get('ITYPE', ''):>3.3}{r.get('Tmin', ''):>7.7}{r.get('Tmax', ''):>7.7}"
            f"{r.get('Formula', ''):>3.3}{r.get('ID', ''):>6.6}"
            f"{tail}"
        )
        lines.append(line)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# activation_energies_refs.in
# ---------------------------------------------------------------------------
def parse_activation(text: str) -> tuple[list[dict], list[str]]:
    headers = ["Reactant1", "Reactant2", "Product1", "Product2", "EA(K)", "Reference"]
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        # Try to parse: R1 R2 -> P1 P2 EA ref
        m = re.match(
            r"(\S+)\s+(\S+)\s*->\s*(\S+)\s*(\S*)\s+([\d.eE+\-]+)\s*(.*)",
            stripped,
        )
        if m:
            rows.append({
                "Reactant1": m.group(1),
                "Reactant2": m.group(2),
                "Product1": m.group(3),
                "Product2": m.group(4),
                "EA(K)": m.group(5),
                "Reference": m.group(6).strip(),
            })
        else:
            # Try with only one product (no second product)
            m2 = re.match(
                r"(\S+)\s+(\S+)\s*->\s*(\S+)\s+([\d.eE+\-]+)\s*(.*)",
                stripped,
            )
            if m2:
                rows.append({
                    "Reactant1": m2.group(1),
                    "Reactant2": m2.group(2),
                    "Product1": m2.group(3),
                    "Product2": "",
                    "EA(K)": m2.group(4),
                    "Reference": m2.group(5).strip(),
                })
    return rows, headers


def serialize_activation(rows: list[dict]) -> str:
    lines = ["!                                                         EA(K)"]
    for r in rows:
        p2 = f"  {r['Product2']}" if r.get("Product2") else ""
        ref = f"  {r['Reference']}" if r.get("Reference") else ""
        lines.append(
            f"{r['Reactant1']:<10s}{r['Reactant2']:<24s}-> {r['Product1']}{p2:<24s}"
            f"{r['EA(K)']:>12s}{ref}"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# surface_parameters.in
# ---------------------------------------------------------------------------
def parse_surface_params(text: str) -> tuple[list[dict], list[str]]:
    headers = ["Species", "m/mH", "ED(K)", "Eb(K)", "Ref for ED", "dHf(kcal/mol)"]
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        parts = stripped.split()
        if len(parts) < 4:
            continue
        row = {
            "Species": parts[0],
            "m/mH": parts[1],
            "ED(K)": parts[2],
            "Eb(K)": parts[3],
        }
        # Reference is everything between Eb and the * marker
        rest = " ".join(parts[4:])
        star_idx = rest.find("*")
        if star_idx >= 0:
            row["Ref for ED"] = rest[:star_idx].strip()
            row["dHf(kcal/mol)"] = rest[star_idx + 1:].strip()
        else:
            row["Ref for ED"] = rest
            row["dHf(kcal/mol)"] = ""
        rows.append(row)
    return rows, headers


def serialize_surface_params(rows: list[dict]) -> str:
    lines = ["! Spec        m/mH  ED(K) Eb(K)  Ref for ED               dHf(kcal/mol)"]
    for r in rows:
        ref = r.get("Ref for ED", "")
        dhf = r.get("dHf(kcal/mol)", "")
        ref_part = f"  {ref}" if ref else ""
        dhf_part = f"  *  {dhf}" if dhf else ""
        lines.append(
            f"{r['Species']:<12s}{r['m/mH']:>5s}  {r['ED(K)']:>6s}  {r['Eb(K)']:>5s}"
            f"{ref_part}{dhf_part}"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 1D_static.dat
# ---------------------------------------------------------------------------
STATIC_HEADERS = [
    "Distance [AU]", "n(H) [cm^-3]", "Tgas [K]", "Av [mag]",
    "Tdust [K]", "1/n_grain", "Av/NH", "a_grain [cm]", "zeta_H2",
]


def parse_1d_static(text: str) -> tuple[list[dict], list[str]]:
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        parts = stripped.split()
        if len(parts) < len(STATIC_HEADERS):
            continue
        rows.append({h: parts[i] for i, h in enumerate(STATIC_HEADERS)})
    return rows, STATIC_HEADERS


def serialize_1d_static(rows: list[dict]) -> str:
    lines = [
        "! Distance [AU] ; H Gas density [part/cm^3] ; Gas Temperature [K] ; "
        "Visual Extinction [mag] ; Dust Temperature [K] ; 1/abundance of grains ; "
        "AV/NH conversion factor ; radius of grains (cm); zeta_H2"
    ]
    for r in rows:
        line = " ".join(f"{r[h]:>14s}" for h in STATIC_HEADERS)
        lines.append(line)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# structure_evolution.dat
# ---------------------------------------------------------------------------
EVOLUTION_HEADERS = ["time [yr]", "log(Av) [mag]", "log(n) [cm^-3]", "log(T) [K]"]


def parse_structure_evolution(text: str) -> tuple[list[dict], list[str]]:
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        parts = stripped.split()
        if len(parts) < len(EVOLUTION_HEADERS):
            continue
        rows.append({h: parts[i] for i, h in enumerate(EVOLUTION_HEADERS)})
    return rows, EVOLUTION_HEADERS


def serialize_structure_evolution(rows: list[dict]) -> str:
    lines = [
        "! time    log(Av)    log(n)    log(T)",
        "! (yr)   (mag)      (cm-3)    (K)",
    ]
    for r in rows:
        line = " ".join(f"{r[h]:>14s}" for h in EVOLUTION_HEADERS)
        lines.append(line)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
PARSERS = {
    "parameters.in": parse_parameters,
    "element.in": parse_element,
    "abundances.in": parse_abundances,
    "gas_species.in": parse_species,
    "grain_species.in": parse_species,
    "gas_reactions.in": parse_reactions,
    "grain_reactions.in": parse_reactions,
    "activation_energies_refs.in": parse_activation,
    "surface_parameters.in": parse_surface_params,
    "1D_static.dat": parse_1d_static,
    "structure_evolution.dat": parse_structure_evolution,
}

SERIALIZERS = {
    "parameters.in": serialize_parameters,
    "element.in": serialize_element,
    "abundances.in": serialize_abundances,
    "gas_species.in": serialize_species,
    "grain_species.in": serialize_species,
    "gas_reactions.in": serialize_reactions,
    "grain_reactions.in": serialize_reactions,
    "activation_energies_refs.in": serialize_activation,
    "surface_parameters.in": serialize_surface_params,
    "1D_static.dat": serialize_1d_static,
    "structure_evolution.dat": serialize_structure_evolution,
}
