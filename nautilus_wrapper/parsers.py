"""Parsers for nautilus input files.

Each parser returns a list of dicts (rows) and a list of column headers.
The serialize function converts them back to the original text format.
"""

import ast
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
# observations.txt
#
# An optional, per-input-set file of observed abundances, written in the R
# style astrochemists already use for this ("Species <- expression # ref"),
# one per line. A leading '#' deactivates a line (excludes it from the
# distance-of-disagreement calculation) without deleting it; everything else
# — banners, section headers, blank-separator comments — round-trips as a
# comment-only row.
# ---------------------------------------------------------------------------
OBSERVATION_HEADERS = ["Species", "Expression", "Comment"]

_OBS_ASSIGN_RE = re.compile(r"^(\S+)\s*(?:<-|=)\s*(.+)$")
# "### <name>" section headers group the species by key atom; a bare
# "####..." banner ends the current section.
_OBS_SECTION_RE = re.compile(r"^#{3,}\s*(.*?)\s*$")
_SECTION_ATOM = {"carbon": "C", "nitrogen": "N", "oxygen": "O",
                 "sulfur": "S", "sulphur": "S"}


def parse_observations(text: str) -> tuple[list[dict], list[str]]:
    rows = []
    atom: str | None = None  # key atom of the current "### <name>" section
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        sec = _OBS_SECTION_RE.match(stripped)
        if sec is not None:
            name = sec.group(1).strip("# ").strip()
            atom = _SECTION_ATOM.get(name.lower())

        active = not stripped.startswith("#")
        body = stripped if active else stripped[1:].strip()

        code, _, comment = body.partition("#")
        code = code.strip()
        comment = comment.strip()

        m = _OBS_ASSIGN_RE.match(code) if code else None
        if m:
            rows.append({
                "Species": m.group(1),
                "Expression": m.group(2).strip(),
                "Comment": comment,
                "_active": active,
                "_atom": atom,
            })
        else:
            # Not a "name <- expr" line even once uncommented (a banner, a
            # section header, a stray note): keep it verbatim as a
            # comment-only row so it round-trips.
            rows.append({
                "Species": "",
                "Expression": "",
                "Comment": body,
                "_active": False,
                "_atom": atom,
            })
    return rows, OBSERVATION_HEADERS


def serialize_observations(rows: list[dict]) -> str:
    lines = []
    for r in rows:
        species = r.get("Species", "")
        if not species:
            comment = r.get("Comment", "")
            lines.append(f"#{comment}" if comment else "#")
            continue
        part = f"{species} <- {r.get('Expression', '')}"
        if r.get("Comment"):
            part += f"  # {r['Comment']}"
        if not r.get("_active", True):
            part = f"#{part}"
        lines.append(part)
    return "\n".join(lines) + "\n"


# A trailing lowercase p/m marks a cation/anion in this notation (real
# species/element symbols are upper-case), and a leading c./l./t. marks a
# cyclic/linear/triplet isomer.
_OBS_ION_SUFFIX = {"p": "+", "m": "-"}
_OBS_ISOMER_PREFIX = {"c.": "c-", "l.": "l-", "t.": "t-"}


def obs_name_to_species(name: str) -> tuple[str, tuple[str, ...]]:
    """Map an observations.txt name to ``(label, ab_components)``.

    ``label`` is the key the observation is stored under (so duplicate rows
    for the same species/phase collapse, last active one winning).
    ``ab_components`` is the tuple of Nautilus ``ab/<species>.ab`` names
    whose predicted abundances are summed to compare against this
    observation.

    Rules, inferred from the TMC1 observation lists this format comes from:
    the token before the first ``_`` is the species (``H2O_TMC1`` -> ``H2O``,
    ``C9H8_TMC1_Gotham`` -> ``C9H8``); a leading ``c.``/``l.``/``t.`` becomes
    ``c-``/``l-``/``t-``; a trailing ``p``/``m`` becomes ``+``/``-``
    (``N2Hp`` -> ``N2H+``). An ``_Ice`` token anywhere after the species
    marks a grain value: label ``<species>_ice`` and components
    ``("J<species>", "K<species>")`` — the surface + mantle populations,
    which Nautilus tracks separately.
    """
    tokens = name.split("_")
    core = tokens[0]
    is_ice = any(t.lower() == "ice" for t in tokens[1:])

    prefix = ""
    for tag, repl in _OBS_ISOMER_PREFIX.items():
        if core.startswith(tag):
            prefix, core = repl, core[len(tag):]
            break

    suffix = ""
    if len(core) > 1 and core[-1] in _OBS_ION_SUFFIX:
        suffix, core = _OBS_ION_SUFFIX[core[-1]], core[:-1]

    species = f"{prefix}{core}{suffix}"
    if is_ice:
        return f"{species}_ice", (f"J{species}", f"K{species}")
    return species, (species,)


# Arithmetic-only expression evaluator for the Expression column (numbers,
# +-*/**, parentheses, and references to earlier names in the same file) —
# deliberately not a general `eval`, so a malformed line can't run code.
_SAFE_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
}


def _safe_eval_node(node: ast.AST, env: dict[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body, env)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ValueError(f"unknown reference '{node.id}'")
        return env[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BINOPS:
        return _SAFE_BINOPS[type(node.op)](
            _safe_eval_node(node.left, env), _safe_eval_node(node.right, env)
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _safe_eval_node(node.operand, env)
        return -value if isinstance(node.op, ast.USub) else value
    raise ValueError("unsupported expression")


def evaluate_observations(
    rows: list[dict],
) -> tuple[dict[str, tuple[float, tuple[str, ...], str | None]], dict[str, str]]:
    """Evaluate the *active* rows of a parsed observations.txt.

    Returns ``(by_species, errors)``. ``by_species`` maps a label (from
    :func:`obs_name_to_species`) to ``(observed_abundance, ab_components,
    atom)``, where ``ab_components`` is the tuple of ``ab/<species>.ab``
    names whose predicted abundances are summed to compare against it (one
    entry for a gas species, ``J`` + ``K`` for an ice one) and ``atom`` is
    the key atom (``"C"``/``"N"``/``"O"``/``"S"``) of the ``### <name>``
    section the row sits in, or ``None``. Rows are evaluated top-to-bottom
    with each name added to the environment as it's defined (so a later
    expression can reference an earlier one, e.g.
    ``SO_GEMs <- (SO_CPa+SO_CPb+SO_CPc)/3``), and — matching the R semantics
    this file is written in — a later active row for the same label
    overwrites an earlier one. ``errors`` maps a raw name to why its
    expression couldn't be evaluated (bad syntax, or a reference to a name
    never defined or only defined in a deactivated row).
    """
    env: dict[str, float] = {}
    by_species: dict[str, tuple[float, tuple[str, ...], str | None]] = {}
    errors: dict[str, str] = {}
    for r in rows:
        name = r.get("Species", "")
        if not name or not r.get("_active", True):
            continue
        expr = r.get("Expression", "").replace("^", "**")
        try:
            value = _safe_eval_node(ast.parse(expr, mode="eval"), env)
        except Exception as exc:
            errors[name] = str(exc)
            continue
        env[name] = value
        label, components = obs_name_to_species(name)
        by_species[label] = (value, components, r.get("_atom"))
    return by_species, errors


# ---------------------------------------------------------------------------
# control_plots.txt  (the 4x4 "Control plots" grid)
#
# One non-comment line per grid row (top to bottom); each line is split on
# '|' into up to 4 cells (left to right); each cell lists the species to
# overlay in that subplot, separated by whitespace and/or commas. '#' lines
# and blank lines are ignored. Cell (row 0, col 0) is always the distance of
# disagreement — whatever is written there is ignored.
# ---------------------------------------------------------------------------
CONTROL_GRID_SIZE = 4


def parse_control_grid(text: str) -> list[list[list[str]]]:
    """Parse control_plots.txt into a ``4x4`` grid of species-name lists.

    Always returns a full ``CONTROL_GRID_SIZE x CONTROL_GRID_SIZE`` structure;
    absent rows/cells come back as empty lists.
    """
    n = CONTROL_GRID_SIZE
    grid = [[[] for _ in range(n)] for _ in range(n)]
    r = 0
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if r >= n:
            break
        for c, cell in enumerate(line.split("|")[:n]):
            grid[r][c] = [tok for tok in re.split(r"[,\s]+", cell.strip()) if tok]
        r += 1
    return grid


# ---------------------------------------------------------------------------
# plot_settings.txt  (persisted abundance-plot configuration)
#
# An INI-ish file with four sections, rewritten every time "Plot" runs and
# safe to hand-edit while the app is closed. '#' comments and blank lines are
# ignored; section names and keys are case-insensitive.
#
#   [species]       one whitespace/comma-separated list of species to overlay
#   [layout]        cols / rows ints, plus "cell <n> = sp sp ..." per subplot
#                   (n counts left-to-right, top-to-bottom; "*" marks the
#                   distance-of-disagreement pseudo-series)
#   [style]         xlim = "<min> <max>" (years), observed = both|control|plots
#   [distribution]  "<output name> = <linestyle>" per checked output, in order
# ---------------------------------------------------------------------------
DISAGREEMENT_TOKEN = "*"
_OBSERVED_MODES = ("both", "control", "plots")


def parse_plot_settings(text: str) -> dict:
    """Parse plot_settings.txt into a dict with (optional) keys ``species``
    (list[str]), ``layout`` ({cols, rows, cells: list[list[str]]}), ``style``
    ({xlim: (lo, hi), observed: str}) and ``distribution`` (list[(name, ls)]).

    Missing sections/keys are simply absent from the result.
    """
    section: str | None = None
    species: list[str] = []
    lay: dict[str, Any] = {}
    lay_cells: dict[int, list[str]] = {}
    style: dict[str, Any] = {}
    distribution: list[tuple[str, str]] = []

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
            continue
        if section == "species":
            species.extend(t for t in re.split(r"[,\s]+", line) if t)
        elif section == "layout":
            if "=" not in line:
                continue
            key, val = (p.strip() for p in line.split("=", 1))
            kl = key.lower()
            if kl in ("cols", "rows"):
                try:
                    lay[kl] = max(1, min(6, int(val)))
                except ValueError:
                    pass
            elif kl.startswith("cell"):
                try:
                    idx = int(kl[4:].strip())
                except ValueError:
                    continue
                lay_cells[idx] = [t for t in re.split(r"[,\s]+", val) if t]
        elif section == "style":
            if "=" not in line:
                continue
            key, val = (p.strip() for p in line.split("=", 1))
            kl = key.lower()
            if kl == "xlim":
                nums = re.findall(r"[-+0-9.eE]+", val)
                if len(nums) >= 2:
                    try:
                        style["xlim"] = (float(nums[0]), float(nums[1]))
                    except ValueError:
                        pass
            elif kl == "observed" and val.lower() in _OBSERVED_MODES:
                style["observed"] = val.lower()
        elif section == "distribution":
            if "=" not in line:
                continue
            name, ls = (p.strip() for p in line.split("=", 1))
            if name:
                distribution.append((name, ls))

    out: dict[str, Any] = {}
    if species:
        out["species"] = species
    if lay or lay_cells:
        cols = lay.get("cols", 1)
        rows = lay.get("rows", 1)
        n = cols * rows
        cells = [lay_cells.get(i + 1, []) for i in range(n)]
        out["layout"] = {"cols": cols, "rows": rows, "cells": cells}
    if style:
        out["style"] = style
    if distribution:
        out["distribution"] = distribution
    return out


def serialize_plot_settings(settings: dict) -> str:
    """Inverse of :func:`parse_plot_settings`. ``settings`` uses the same keys;
    any that are missing get sensible defaults."""
    species = list(settings.get("species") or [])
    layout = settings.get("layout") or {}
    style = settings.get("style") or {}
    distribution = list(settings.get("distribution") or [])

    cols = max(1, min(6, int(layout.get("cols", 1))))
    rows = max(1, min(6, int(layout.get("rows", 1))))
    cells = list(layout.get("cells") or [])

    xlim = style.get("xlim", (1e3, 1e7))
    observed = style.get("observed", "both")

    def _fmt(x: float) -> str:
        return f"{x:g}"

    lines = [
        "# plot_settings.txt - abundance-plot configuration for the Nautilus",
        "# Wrapper. Rewritten every time \"Plot\" runs; safe to hand-edit while",
        "# the app is closed. '#' comments and blank lines are ignored.",
        "",
        "[species]",
        "# Species overlaid in the abundance plot (whitespace/comma separated).",
        "  ".join(species) if species else "",
        "",
        "[layout]",
        "# Subplot grid. One \"cell <n> = ...\" per subplot, numbered",
        f"# left-to-right, top-to-bottom. Put {DISAGREEMENT_TOKEN} in a cell to draw the",
        "# distance-of-disagreement curve there instead of species.",
        f"cols = {cols}",
        f"rows = {rows}",
    ]
    for i in range(cols * rows):
        cell = cells[i] if i < len(cells) else []
        lines.append(f"cell {i + 1} = " + "  ".join(cell))
    lines += [
        "",
        "[style]",
        "# xlim     abundance-plot time axis, \"<min> <max>\" in years",
        "# observed where observed-abundance bars are drawn: both | control | plots",
        f"xlim     = {_fmt(xlim[0])}  {_fmt(xlim[1])}",
        f"observed = {observed}",
        "",
        "[distribution]",
        "# Outputs checked in \"Outputs to plot\" and the line style each is drawn",
        "# with, in draw order (session runs are solid). Regenerated on every plot.",
    ]
    for name, ls in distribution:
        lines.append(f"{name} = {ls}")
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
    "observations.txt": parse_observations,
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
    "observations.txt": serialize_observations,
}
