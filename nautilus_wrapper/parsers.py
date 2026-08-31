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
        rows.append({"Species": key.strip(), "Abundance": value.strip(), "Comment": comment})
    return rows, headers


def serialize_abundances(rows: list[dict]) -> str:
    lines = ["! Abundances in relative abundances with respect to Hydrogen"]
    for r in rows:
        part = f"{r['Species']:12s}= {r['Abundance']}"
        if r.get("Comment"):
            part += f" ! {r['Comment']}"
        lines.append(part)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# gas_species.in / grain_species.in
# ---------------------------------------------------------------------------
ELEMENT_COLS = ["charge", "H", "He", "C", "N", "O", "Si", "S", "Fe", "Na", "Mg", "Cl", "P", "F"]


def parse_species(text: str) -> tuple[list[dict], list[str]]:
    headers = ["Species"] + ELEMENT_COLS
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        name = parts[0]
        nums = parts[1:]
        row = {"Species": name}
        for i, col in enumerate(ELEMENT_COLS):
            row[col] = nums[i] if i < len(nums) else "0"
        rows.append(row)
    return rows, headers


def serialize_species(rows: list[dict]) -> str:
    lines = []
    for r in rows:
        parts = [r["Species"]]
        for col in ELEMENT_COLS:
            parts.append(r.get(col, "0"))
        lines.append(" ".join(f"{p:>3s}" if i > 0 else f"{p:<12s}" for i, p in enumerate(parts)))
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


def _parse_reaction_line(line: str) -> dict | None:
    """Parse a single reaction line using fixed-width fields."""
    stripped = line.strip()
    if not stripped or stripped.startswith("!"):
        return None

    row = {}
    # Reactants: positions 0-10, 11-21, 22-32
    row["Reactant1"] = stripped[0:11].strip()
    row["Reactant2"] = stripped[11:22].strip()
    row["Reactant3"] = stripped[22:33].strip()
    # Products: positions 33-43, 44-54, 55-65, 66-76, 77-87
    row["Product1"] = stripped[33:44].strip()
    row["Product2"] = stripped[44:55].strip()
    row["Product3"] = stripped[55:66].strip()
    row["Product4"] = stripped[66:77].strip()
    row["Product5"] = stripped[77:88].strip()
    # Rate coefficients A, B, C (11 chars each)
    row["A"] = stripped[88:99].strip()
    row["B"] = stripped[99:110].strip()
    row["C"] = stripped[110:121].strip()

    # The rest is variable - try to extract remaining fields
    rest = stripped[121:].strip()
    parts = rest.split()
    # fields: [possibly junk], ITYPE, Tmin, Tmax, Formula, ID, ...
    # We just store whatever we find after C
    row["ITYPE"] = parts[0] if len(parts) > 0 else ""
    row["Tmin"] = parts[1] if len(parts) > 1 else ""
    row["Tmax"] = parts[2] if len(parts) > 2 else ""
    row["Formula"] = parts[3] if len(parts) > 3 else ""
    row["ID"] = parts[4] if len(parts) > 4 else ""

    return row


def parse_reactions(text: str) -> tuple[list[dict], list[str]]:
    headers = REACTION_COLS
    rows = []
    for line in text.splitlines():
        row = _parse_reaction_line(line)
        if row is not None:
            rows.append(row)
    return rows, headers


def serialize_reactions(rows: list[dict]) -> str:
    """Serialize reaction rows back to fixed-width text.

    We reconstruct the original layout as closely as possible.
    """
    lines = ["!    Reactants                    Products                                                  A          B          C      xxxxxxxxxxxxxxxxxxxxx ITYPE Tmin   Tmax formula ID xxxxx"]
    for r in rows:
        # Build the fixed-width line
        reactants = f"{r['Reactant1']:<11s}{r['Reactant2']:<11s}{r['Reactant3']:<11s}"
        products = (
            f"{r['Product1']:<11s}{r['Product2']:<11s}{r['Product3']:<11s}"
            f"{r['Product4']:<11s}{r['Product5']:<11s}"
        )
        abc = f"{r['A']:>11s}{r['B']:>11s}{r['C']:>11s}"
        tail_parts = [r.get("ITYPE", ""), r.get("Tmin", ""), r.get("Tmax", ""),
                      r.get("Formula", ""), r.get("ID", "")]
        tail = " ".join(p for p in tail_parts if p)
        line = f"{reactants} {products}{abc} {tail}"
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
