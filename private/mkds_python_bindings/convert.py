#!/usr/bin/env python3
"""
Auto-generate ctypes bindings from MKDS headers using libclang.

Outputs (to ./out):
- One Python file per header.
- Each file contains:
    • typedef aliases (e.g., fx32 = c_int32)
    • structs/unions (typedef'd records become classes)
    • enums as integer constants (members -> top-level ints)
    • import statements for dependent types from other headers

Key fixes:
- Builds a GLOBAL dependency graph for all typedefs/records across all headers,
  performs a single topological sort, then emits per-file subsets in that order.
  -> No class ever references a later class in the same file.
- Enums are proper integer constants; fields with enum type use c_int.
- Stable, readable names for anonymous records (when no typedef names them):
  `anon_<headerStem>_<line>_<col>`
- Removes stray trailing underscores from identifiers.
"""

from pathlib import Path
import os
import sys
import re
from collections import defaultdict, deque
from clang.cindex import Config, Index, CursorKind, TypeKind

# -----------------------
# 1) Configurable paths
# -----------------------
LLVM_LIB = os.environ.get(
    "LIBCLANG_PATH",
    "/opt/homebrew/Cellar/llvm/20.1.8/lib/libclang.dylib"
)

DEVKITPRO_ROOT = "/opt/devkitpro"
DEVKITARM_SYSINC = f"{DEVKITPRO_ROOT}/devkitARM/arm-none-eabi/include"
MKDS_C_ROOT = "/Users/blakemoody/dev/mariokart_ml/mkds_c"
MKDS_INCLUDE = MKDS_C_ROOT
OUT_DIR = Path("./out")
OUT_DIR.mkdir(exist_ok=True)

STUB_DIR = Path(f"{MKDS_C_ROOT}/external_include")
STUB_DIR.mkdir(exist_ok=True)
MALLOC_STUB = STUB_DIR / "malloc.h"
if not MALLOC_STUB.exists():
    MALLOC_STUB.write_text(
        "#ifndef MALLOC_H\n#define MALLOC_H\n#include <stdlib.h>\n#endif\n"
    )

# -----------------------
# libclang setup
# -----------------------
try:
    Config.set_library_file(LLVM_LIB)
except Exception as e:
    sys.stderr.write(f"Failed to load libclang at {LLVM_LIB}\n{e}\n")
    sys.exit(1)

# -----------------------
# Helpers
# -----------------------
PYTHON_BINDINGS_IMPORT_PREFIX = "private.mkds_python_bindings.out"

def sanitize_name(name: str) -> str:
    """Make C identifiers valid in Python, remove trailing underscores."""
    if not name:
        return "_anon"
    ident = re.sub(r"\W", "_", name)
    ident = re.sub(r"_+$", "", ident)  # strip trailing underscores
    if not ident:
        ident = "_anon"
    if ident[0].isdigit():
        ident = "_" + ident
    return ident

def is_project_file(cursor) -> bool:
    loc = cursor.location
    return bool(loc.file) and str(loc.file.name).startswith(MKDS_INCLUDE)

def clang_args_for_parse():
    # DO NOT CHANGE (as requested)
    return [
        "-x", "c",
        "-fsyntax-only",
        "-DARM9", "-DNDS", "-D__NDS__", "-D__CALICO__", "-DGBA", "-D_REVOLUTION",
        f"-I{DEVKITPRO_ROOT}/libnds/include",
        f"-I{DEVKITPRO_ROOT}/libnds/include/nds",
        f"-I{DEVKITPRO_ROOT}/calico/include",
        f"-I{DEVKITPRO_ROOT}/calico/include/calico",
        f"-I{DEVKITPRO_ROOT}/devkitARM/arm-none-eabi/include",
        f"-I{DEVKITPRO_ROOT}/devkitARM/lib/gcc/arm-none-eabi/12.2.0/include",
        f"-I{MKDS_C_ROOT}/external_include",
        f"-I{MKDS_C_ROOT}",
        "-include nds.h",
        "-include nnsys.h",
        "-include nitro_sp.h",
        "-include dwc.h",
        "-include nitroWiFi.h",
        "-include ninet.h",
        "-include malloc.h"
    ]

def c_primitive_map(base: str) -> str | None:
    return {
        "char": "c_char",
        "signed": "c_int",
        "unsigned": "c_uint",
        "short": "c_short",
        "unsigned short": "c_ushort",
        "int": "c_int",
        "unsigned int": "c_uint",
        "long": "c_long",
        "unsigned long": "c_ulong",
        "float": "c_float",
        "double": "c_double",
        "s8": "c_int8", "u8": "c_uint8",
        "s16": "c_int16", "u16": "c_uint16",
        "s32": "c_int32", "u32": "c_uint32",
        "s64": "c_int64", "u64": "c_uint64",
    }.get(base)

def make_anon_record_name(node) -> str:
    """Deterministic fallback for anonymous records without typedef aliases."""
    loc = node.location
    header_stem = sanitize_name(Path(str(loc.file)).stem) if loc.file else "unknown"
    line = loc.line or 0
    col = loc.column or 0
    return f"anon_{header_stem}_{line}_{col}"

# -----------------------
# Globals
# -----------------------
# Collected declarations (phase 1)
seen_usrs: set[str] = set()
record_typedef_map: dict[str, str] = {}           # USR(record) -> typedef name
typedefs_by_header: dict[str, list[tuple[str,str]]] = defaultdict(list)  # header -> [(typedef, rhs_expr)]
records_by_header: dict[str, list[tuple[str,list[tuple[str,str]],bool]]] = defaultdict(list)  # header -> [(name, fields, is_union)]
enums_by_header: dict[str, list[tuple[str, list[tuple[str,int]]]]] = defaultdict(list)  # header -> [(enum_name, [(const, val)])]
known_typedef_names: set[str] = set()

# Name -> header
typedef_to_header: dict[str, str] = {}
record_to_header: dict[str, str] = {}

# Inter-file deps (for imports)
header_dependencies: dict[str, set[str]] = defaultdict(set)

# Global type dep-graph (typedef + records only; enums are constants)
# node -> set(dependencies)
global_type_deps: dict[str, set[str]] = defaultdict(set)
# Universe of globally known "type names" (typedefs + records)
global_type_nodes: set[str] = set()

# -----------------------
# Clang AST helpers
# -----------------------
def _unwrap_elaborated(t):
    if t.kind == TypeKind.ELABORATED:
        nt = t.get_named_type()
        if nt.kind != TypeKind.INVALID:
            return nt
    return t

def clang_type_to_ctypes(t) -> str:
    """C/C++ type -> ctypes expression (string). Uses typedef/record names when possible."""
    t = _unwrap_elaborated(t)
    kind = t.kind
    canonical = t.get_canonical()
    spelled = t.spelling.strip()

    # Arrays
    if kind == TypeKind.CONSTANTARRAY:
        elem = clang_type_to_ctypes(t.element_type)
        return f"({elem} * {t.element_count})"

    if kind == TypeKind.INCOMPLETEARRAY:
        elem = clang_type_to_ctypes(t.element_type)
        return f"POINTER32({elem})"

    # Pointers
    if kind == TypeKind.POINTER:
        ptee = _unwrap_elaborated(t.get_pointee())
        if ptee.kind in (TypeKind.VOID, TypeKind.INVALID, TypeKind.FUNCTIONPROTO, TypeKind.FUNCTIONNOPROTO):
            return "c_void_p32"
        return f"POINTER32({clang_type_to_ctypes(ptee)})"

    # Typedefs
    if kind == TypeKind.TYPEDEF:
        return sanitize_name(t.spelling)

    # If spelled matches a known typedef name
    if spelled in known_typedef_names:
        return sanitize_name(spelled)

    # Records (struct / union)
    if kind == TypeKind.RECORD:
        decl = t.get_declaration()
        tag = decl.spelling or record_typedef_map.get(decl.get_usr())
        if not tag:
            tag = make_anon_record_name(decl)
        return sanitize_name(tag)

    # Enums: represent as c_int in fields
    if kind == TypeKind.ENUM:
        return "c_int"

    # Primitives
    base = canonical.spelling.split()[-1]
    mapped = c_primitive_map(base)
    return mapped or "c_void_p32"

def resolve_typedef_rhs(t) -> str:
    """
    Resolve a typedef RHS to a python expression that doesn't self-reference.
    Prefer underlying canonical map; map enum -> c_int; struct/union -> typedef alias if any.
    """
    t = _unwrap_elaborated(t)
    canonical = t.get_canonical()

    # Primitive?
    base = canonical.spelling.split()[-1]
    mapped = c_primitive_map(base)
    if mapped:
        return mapped

    # Typedef to typedef chain
    if canonical.kind == TypeKind.TYPEDEF:
        return resolve_typedef_rhs(_unwrap_elaborated(canonical))

    # Enum becomes c_int
    if canonical.kind == TypeKind.ENUM:
        return "c_int"

    # Record -> prefer its typedef-alias, else stable anon/class tag
    decl = canonical.get_declaration()
    if decl.kind in (CursorKind.STRUCT_DECL, CursorKind.UNION_DECL):
        usr = decl.get_usr()
        if usr in record_typedef_map:
            return sanitize_name(record_typedef_map[usr])
        tag = decl.spelling or make_anon_record_name(decl)
        return sanitize_name(tag)

    # Fallback
    return "c_void_p32"

# -----------------------
# Collection (phase 1)
# -----------------------
def collect_typedefs(node):
    if not is_project_file(node):
        for c in node.get_children():
            collect_typedefs(c)
        return

    if node.kind == CursorKind.TYPEDEF_DECL:
        name = node.spelling
        if not name:
            return
        known_typedef_names.add(name)
        file_path = str(node.location.file)
        header_stem = Path(file_path).stem
        typedef_to_header[name] = header_stem

        underlying = _unwrap_elaborated(node.underlying_typedef_type)
        decl = underlying.get_declaration()

        if decl.kind in (CursorKind.STRUCT_DECL, CursorKind.UNION_DECL):
            # typedef struct {...} Foo;
            record_typedef_map[decl.get_usr()] = name
        else:
            rhs = resolve_typedef_rhs(underlying)
            if rhs != name:
                typedefs_by_header[header_stem].append((name, rhs))

        # Register node in global space
        global_type_nodes.add(name)

    for c in node.get_children():
        collect_typedefs(c)

def collect_enums(node):
    if node.kind == CursorKind.ENUM_DECL and is_project_file(node):
        # Capture enum members as constants
        name = node.spelling or "_anon_enum"
        file_path = str(node.location.file)
        header_stem = Path(file_path).stem
        members = []
        for child in node.get_children():
            if child.kind == CursorKind.ENUM_CONSTANT_DECL:
                members.append((child.spelling, child.enum_value))
        enums_by_header[header_stem].append((name, members))
    for c in node.get_children():
        collect_enums(c)

def collect_records(node):
    if node.kind in (CursorKind.STRUCT_DECL, CursorKind.UNION_DECL) and node.is_definition() and is_project_file(node):
        usr = node.get_usr()
        if usr in seen_usrs:
            pass
        else:
            # Prefer real tag or typedef alias
            name = node.spelling or record_typedef_map.get(usr)
            if not name:
                name = make_anon_record_name(node)
            clean = sanitize_name(name)

            seen_usrs.add(usr)
            is_union = node.kind == CursorKind.UNION_DECL
            fields = []
            file_path = str(node.location.file)
            header_stem = Path(file_path).stem
            record_to_header[clean] = header_stem

            # Register node
            global_type_nodes.add(clean)

            for f in node.get_children():
                if f.kind == CursorKind.FIELD_DECL:
                    fname = f.spelling or "_"
                    fexpr = clang_type_to_ctypes(f.type)
                    fields.append((fname, fexpr))
            records_by_header[header_stem].append((clean, fields, is_union))
    for c in node.get_children():
        collect_records(c)

def parse_header(path: str):
    index = Index.create()
    tu = index.parse(path, args=clang_args_for_parse())
    collect_typedefs(tu.cursor)
    collect_records(tu.cursor)
    collect_enums(tu.cursor)

# -----------------------
# Dependency graph (phase 2)
# -----------------------
def _record_dep(user: str, dep: str):
    """Add a dependency edge user -> dep if both are global type nodes."""
    if user and dep and user != dep and dep in global_type_nodes:
        global_type_deps[user].add(dep)

def build_dependency_graph():
    """
    Build a GLOBAL dependency graph across all headers:
      - typedef name -> rhs symbol (if rhs is a typedef/record)
      - record name  -> every symbol referenced in its field types
    """
    # Typedef dependencies
    for header_stem, typedefs in typedefs_by_header.items():
        for name, rhs in typedefs:
            # If rhs is a typedef/record symbol, set dependency
            if rhs in global_type_nodes:
                _record_dep(name, rhs)

            # Also track for inter-file imports
            if rhs in typedef_to_header and typedef_to_header[rhs] != header_stem:
                header_dependencies[header_stem].add(typedef_to_header[rhs])
            elif rhs in record_to_header and record_to_header[rhs] != header_stem:
                header_dependencies[header_stem].add(record_to_header[rhs])

    # Records dependencies (field types)
    for header_stem, records in records_by_header.items():
        for name, fields, _ in records:
            # Ensure node exists in graph even if no dependencies
            global_type_deps.setdefault(name, set())

            for _, fexpr in fields:
                # Tokenize field expression and add edges to any known types
                tokens = re.findall(r"[A-Za-z_]\w*", fexpr)
                for token in tokens:
                    if token in global_type_nodes:
                        _record_dep(name, token)

                    # Inter-file import tracking
                    if token in typedef_to_header and typedef_to_header[token] != header_stem:
                        header_dependencies[header_stem].add(typedef_to_header[token])
                    elif token in record_to_header and record_to_header[token] != header_stem:
                        header_dependencies[header_stem].add(record_to_header[token])

    # Ensure graph nodes present for all known types (even if isolated)
    for t in global_type_nodes:
        global_type_deps.setdefault(t, set())

def global_toposort_types() -> list[str]:
    """Topologically sort all typedef + record symbols by global dependency graph."""
    in_deg = {n: 0 for n in global_type_deps.keys()}
    for src, deps in global_type_deps.items():
        for d in deps:
            if d in in_deg:
                in_deg[d] += 1

    q = deque([n for n, deg in in_deg.items() if deg == 0])
    order: list[str] = []

    while q:
        cur = q.popleft()
        order.append(cur)
        for dep in global_type_deps.get(cur, []):
            if dep in in_deg:
                in_deg[dep] -= 1
                if in_deg[dep] == 0:
                    q.append(dep)

    # Append anything unsorted (cycles/unseen)
    for n in global_type_deps.keys():
        if n not in order:
            order.append(n)

    return order

# -----------------------
# Codegen
# -----------------------
def _typedefs_dict_for_header(header_stem):
    """Quick lookup dict typedef_name -> rhs for a header."""
    return {name: rhs for name, rhs in typedefs_by_header.get(header_stem, [])}

def _records_dict_for_header(header_stem):
    """Quick lookup dict record_name -> (fields, is_union) for a header."""
    d = {}
    for name, fields, is_union in records_by_header.get(header_stem, []):
        d[name] = (fields, is_union)
    return d

def to_ctypes_code(header_stem, sorted_all_types):
    """
    Emit one file:
      - imports
      - enums
      - typedefs (only those in header, in global order)
      - records  (only those in header, in global order)
    """
    lines = [
        "from ctypes import *",
        "from private.mkds_python_bindings.pointer import POINTER32, c_void_p32",
    ]

    # Imports for inter-file deps
    for dep in sorted(header_dependencies.get(header_stem, [])):
        lines.append(f"from {PYTHON_BINDINGS_IMPORT_PREFIX}.{dep} import *")
    if header_dependencies.get(header_stem):
        lines.append("")

    # Enums (emit first, always safe)
    for enum_name, members in enums_by_header.get(header_stem, []):
        for member, value in members:
            lines.append(f"{sanitize_name(member)} = {value}")
        if members:
            lines.append("")

    # Fast lookups
    typedefs_map = _typedefs_dict_for_header(header_stem)
    records_map = _records_dict_for_header(header_stem)

    # Emit everything in GLOBAL order but filter to this header
    for symbol in sorted_all_types:
        # Typedef in this header?
        if symbol in typedefs_map:
            rhs = typedefs_map[symbol]
            lines.append(f"{sanitize_name(symbol)} = {rhs}")
            continue

        # Record in this header?
        if symbol in records_map:
            fields, is_union = records_map[symbol]
            base = "Union" if is_union else "Structure"
            cname = sanitize_name(symbol)
            lines.append(f"\nclass {cname}({base}):")
            lines.append("    _fields_ = [")
            for fname, fexpr in fields:
                fn = sanitize_name(fname)
                lines.append(f"        ('{fn}', {fexpr}),")
            lines.append("    ]")

    if lines and not lines[-1].endswith("\n"):
        lines.append("")
    return "\n".join(lines)

# -----------------------
# Main
# -----------------------
def main():
    headers = list(Path(MKDS_INCLUDE).rglob("*.h"))
    if not headers:
        print(f"No headers found under {MKDS_INCLUDE}")
        return

    # Phase 1 — scan
    for header in headers:
        try:
            parse_header(str(header))
        except Exception as e:
            print(f"Parse failed for {header}: {e}")

    # Phase 2 — deps (inter-file + global type graph)
    build_dependency_graph()

    # Phase 3 — global topo sort
    sorted_all_types = global_toposort_types()

    # Phase 4 — write files (respect global order, filter per header)
    all_headers = sorted(
        set(records_by_header.keys())
        | set(typedefs_by_header.keys())
        | set(enums_by_header.keys())
    )
    for h in all_headers:
        code = to_ctypes_code(h, sorted_all_types)
        out_file = OUT_DIR / f"{h}.py"
        out_file.write_text(code)
        print(f"Generated {out_file}")

    (OUT_DIR / "__init__.py").write_text("")
    print(f"✅ All bindings generated in {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
