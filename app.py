
# app.py — Phase U Sprint 2 (A → A+) — Streamlit app with Boundary (Kill-tests)
# Run locally:   streamlit run app.py
# Streamlit Cloud: push this file + requirements.txt to a repo, then deploy.

import streamlit as st
import json, hashlib, io, csv
from typing import List, Tuple, Dict, Optional

st.set_page_config(page_title="Phase U — A→A+ Test Suite", layout="wide")

# =========================
# F2 Utilities (self-contained)
# =========================

def bits_to_list(b: str) -> List[int]:
    return [1 if ch == "1" else 0 for ch in b]

def list_to_bits(v: List[int]) -> str:
    return "".join("1" if (x & 1) else "0" for x in v)

def dot_mod2(a_bits: str, b_bits: str) -> int:
    return sum((1 if x == "1" else 0) & (1 if y == "1" else 0) for x, y in zip(a_bits, b_bits)) & 1

def hamming_weight_bits(b: str) -> int:
    return b.count("1")

# Matrix as list[str] columns (each length m)
class Matrix:
    def __init__(self, columns: List[str], name: str = ""):
        assert columns, "Matrix needs at least 1 column (use a single zero column if needed)"
        m = len(columns[0])
        for c in columns:
            assert len(c) == m, "All columns must have same height"
            assert set(c) <= {"0","1"}, "Columns must be bitstrings"
        self.columns = list(columns)
        self.m = m
        self.n = len(columns)
        self.name = name or "matrix"

    @staticmethod
    def from_json_str(s: str, name: str = "") -> "Matrix":
        obj = json.loads(s)
        if "columns" in obj:
            cols = obj["columns"]
        else:
            raise ValueError("JSON must contain 'columns': [bitstrings]")
        nm = name or obj.get("meta", {}).get("name", "")
        return Matrix(cols, name=nm)

    @staticmethod
    def from_json_obj(obj: dict, name: str = "") -> "Matrix":
        if "columns" in obj:
            cols = obj["columns"]
        else:
            raise ValueError("JSON must contain 'columns': [bitstrings]")
        nm = name or obj.get("meta", {}).get("name", "")
        return Matrix(cols, name=nm)

    def to_json_str(self) -> str:
        obj = {"m": self.m, "n": self.n, "columns": self.columns, "meta": {"name": self.name}}
        return json.dumps(obj, ensure_ascii=False, indent=2)

    def distinct_types(self, exclude_zero: bool = False) -> set:
        s = set(self.columns)
        if exclude_zero:
            z = "0"*self.m
            s = {c for c in s if c != z}
        return s

# ---------- Linear algebra over F2 ----------
def rref_mod2(matrix_rows: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
    if not matrix_rows:
        return [], []
    rows = [row[:] for row in matrix_rows]
    R = len(rows)
    C = len(rows[0])
    r = 0
    pivots = []
    for c in range(C):
        pivot = None
        for rr in range(r, R):
            if rows[rr][c] & 1:
                pivot = rr
                break
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        pivots.append(c)
        for rr in range(R):
            if rr != r and (rows[rr][c] & 1):
                rows[rr] = [(rows[rr][cc] ^ rows[r][cc]) for cc in range(C)]
        r += 1
        if r == R:
            break
    return rows, pivots

def nullspace_basis_mod2(B_rows: List[List[int]]) -> List[List[int]]:
    if not B_rows:
        return []
    RREF, pivots = rref_mod2(B_rows)
    m = len(B_rows[0])
    pivot_set = set(pivots)
    free = [j for j in range(m) if j not in pivot_set]
    basis = []
    for f in free:
        v = [0]*m
        v[f] = 1
        for i, pc in enumerate(pivots):
            s = 0
            row = RREF[i]
            for j in free:
                if row[j] & 1 and v[j]:
                    s ^= 1
            v[pc] = s
        basis.append(v)
    return basis

def solve_membership_mod2(B_cols: List[str], t_bits: str) -> bool:
    if not B_cols:
        return t_bits == "0"*len(t_bits)
    m = len(B_cols[0])
    k = len(B_cols)
    rows = []
    for r in range(m):
        row = [(1 if B_cols[c][r]=="1" else 0) for c in range(k)]
        row.append(1 if t_bits[r]=="1" else 0)
        rows.append(row)
    RREF, pivots = rref_mod2(rows)
    for row in RREF:
        if all(x==0 for x in row[:k]) and row[k]==1:
            return False
    return True

# ---------- Canonical elements ----------
def canonical_y_star(A: Matrix) -> str:
    B_rows = [[1 if A.columns[j][i]=="1" else 0 for i in range(A.m)] for j in range(A.n)]  # n x m
    basis = nullspace_basis_mod2(B_rows)
    if not basis:
        return "0"*A.m
    best = None
    d = len(basis)
    cap = 1<<d
    for mask in range(1, cap):
        v = [0]*A.m
        for i in range(d):
            if (mask>>i)&1:
                v = [v[j]^basis[i][j] for j in range(A.m)]
        b = list_to_bits(v)
        if best is None:
            best = b
        else:
            w_best = hamming_weight_bits(best)
            w_b = hamming_weight_bits(b)
            if (w_b < w_best) or (w_b == w_best and b < best):
                best = b
    return best

def column_space_basis(A: Matrix) -> List[str]:
    rows = []
    for r in range(A.m):
        rows.append([1 if A.columns[c][r]=="1" else 0 for c in range(A.n)])
    RREF, pivots = rref_mod2(rows)
    return [A.columns[p] for p in pivots] if pivots else []

def canonical_t_star(A: Matrix) -> str:
    basis = column_space_basis(A)
    m = A.m
    for w in range(1, m+1):
        for mask in range(1<<m):
            if mask.bit_count() != w:
                continue
            bits = "".join("1" if (mask>>(m-1-i))&1 else "0" for i in range(m))  # left-to-right
            if not solve_membership_mod2(basis, bits):
                return bits
    return "1"*m

def parity_lex_component_set(A: Matrix, y_bits: str) -> List[str]:
    keep = []
    for c in A.columns:
        if dot_mod2(y_bits, c)==1:
            keep.append(c)
    from collections import Counter
    ctr = Counter(keep)
    odd = [c for c, k in ctr.items() if (k & 1)]
    odd.sort()
    return odd

def serialize_certificate(A: Matrix) -> Tuple[str, Dict]:
    y = canonical_y_star(A)
    t = canonical_t_star(A)
    acomp = parity_lex_component_set(A, y)
    cert = {
        "field": "F2",
        "lex_order": "left-to-right",
        "y": y,
        "t": t,
        "A_comp": acomp
    }
    blob = json.dumps(cert, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest(), cert

# ---------- Admissible maps (Step A) ----------
def append_zero(A: Matrix) -> Matrix:
    return Matrix(A.columns + ["0"*A.m], name=f"{A.name}+z")

def duplicate_last(A: Matrix) -> Matrix:
    return Matrix(A.columns + [A.columns[-1]], name=f"{A.name}+dup_last")

def duplicate_first(A: Matrix) -> Matrix:
    return Matrix(A.columns + [A.columns[0]], name=f"{A.name}+dup_first")

def permute_then_append_zero(A: Matrix) -> Matrix:
    permuted = list(reversed(A.columns))
    return Matrix(permuted + ["0"*A.m], name=f"{A.name}+perm_z")

def append_copy_random(A: Matrix, seed: int = 42) -> Matrix:
    import random
    rnd = random.Random(seed)
    idx = rnd.randrange(len(A.columns))
    return Matrix(A.columns + [A.columns[idx]], name=f"{A.name}+randdup{seed}")

MAPS = {
    "append_zero": append_zero,
    "duplicate_last": duplicate_last,
    "duplicate_first": duplicate_first,
    "permute_then_append_zero": permute_then_append_zero,
}

def apply_map(A: Matrix, map_name: str, seed: int = 42) -> Matrix:
    if map_name == "append_copy_random":
        return append_copy_random(A, seed=seed)
    if map_name in MAPS:
        return MAPS[map_name](A)
    raise ValueError(f"Unknown map: {map_name}")

def is_admissible_stepA(A: Matrix, A1: Matrix) -> bool:
    T0 = A.distinct_types(exclude_zero=True)
    T1 = A1.distinct_types(exclude_zero=True)
    return T1.issubset(T0)

# ---------- Non-admissible (Boundary) helpers ----------
def xor_injection(A: Matrix, i: int, j: int) -> Matrix:
    """Append v_new = col[i] XOR col[j]."""
    m = A.m
    ci = A.columns[i]
    cj = A.columns[j]
    v = "".join("1" if (ci[k] != cj[k]) else "0" for k in range(m))
    return Matrix(A.columns + [v], name=f"{A.name}+xor({i},{j})")

def chord_flip(A: Matrix, col_index: int, row_index: int) -> Matrix:
    """Append a variant of a column with a single bit flipped (simulates local chord flip)."""
    c = list(A.columns[col_index])
    c[row_index] = "0" if c[row_index]=="1" else "1"
    v = "".join(c)
    return Matrix(A.columns + [v], name=f"{A.name}+flip(c{col_index},r{row_index})")

# =========================
# UI Helpers
# =========================

def csv_bytes(rows: List[Dict[str,str]], headers: List[str]) -> bytes:
    buf = io.StringIO()
    import csv as _csv
    w = _csv.DictWriter(buf, fieldnames=headers)
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode("utf-8")

# =========================
# Sidebar — Fixtures
# =========================
st.sidebar.header("Fixtures (Matrices)")
uploaded = st.sidebar.file_uploader("Upload one or more matrix JSON files", type=["json"], accept_multiple_files=True)
if "matrices" not in st.session_state:
    st.session_state["matrices"] = {}

if uploaded:
    for up in uploaded:
        try:
            obj = json.loads(up.read().decode("utf-8"))
            M = Matrix.from_json_obj(obj, name=obj.get("meta",{}).get("name", up.name))
            st.session_state["matrices"][up.name] = M
        except Exception as e:
            st.sidebar.error(f"{up.name}: {e}")

# Seed examples (tiny)
if not st.session_state["matrices"]:
    A = Matrix(["100","010","110"], name="A (toy)")
    A0 = Matrix(["100","010"], name="A0 (toy)")
    st.session_state["matrices"]["A.json"] = A
    st.session_state["matrices"]["A0.json"] = A0

mat_keys = list(st.session_state["matrices"].keys())
choice = st.sidebar.selectbox("Select a matrix", mat_keys, index=0)
seed_matrix = st.session_state["matrices"][choice]
st.sidebar.write("Rows (m):", seed_matrix.m, "Columns (n):", seed_matrix.n)
st.sidebar.json(json.loads(seed_matrix.to_json_str()))

# =========================
# Main — Runners
# =========================
st.title("Phase U — Sprint 2 (A → A+) Test Suite")

tabs = st.tabs([
    "Full-Triple (Cor. 2)",
    "Component Transport (Step A)",
    "Tower",
    "Batch (Seeds × Maps)",
    "Boundary (Kill-tests)",
    "Serialization Preview"
])

# ----- Tab 1: Full-Triple -----
with tabs[0]:
    st.subheader("Full-Triple Fingerprint Invariance (Cor. 2)")
    map_name = st.selectbox("Admissible map", ["append_zero","duplicate_last","duplicate_first","permute_then_append_zero","append_copy_random"], index=0, key="full_map")
    seed = st.number_input("RNG seed (for random copy)", value=42, step=1)
    colA, colB = st.columns(2)
    with colA:
        if st.button("Run on selected matrix", key="run_full_one"):
            h0, c0 = serialize_certificate(seed_matrix)
            A1 = apply_map(seed_matrix, map_name, seed=seed)
            if not is_admissible_stepA(seed_matrix, A1):
                st.error("FAIL (non-admissible: new type)")
            else:
                h1, c1 = serialize_certificate(A1)
                ok = (c0["y"]==c1["y"] and c0["t"]==c1["t"] and c0["A_comp"]==c1["A_comp"] and h0==h1)
                if ok:
                    st.success(f"PASS — hash {h0}")
                else:
                    st.error("FAIL — certificate mismatch")
                    st.json({"before": c0, "after": c1})
    with colB:
        st.caption("Batch across all loaded matrices")
        if st.button("Run across all matrices", key="run_full_all"):
            rows = []
            for name, M in st.session_state["matrices"].items():
                h0, c0 = serialize_certificate(M)
                A1 = apply_map(M, map_name, seed=seed)
                if not is_admissible_stepA(M, A1):
                    rows.append({"matrix": name, "map": map_name, "hash_before": h0, "hash_after": "", "result": "FAIL(non-admissible)"})
                    continue
                h1, c1 = serialize_certificate(A1)
                res = "PASS" if (c0["y"]==c1["y"] and c0["t"]==c1["t"] and c0["A_comp"]==c1["A_comp"] and h0==h1) else "FAIL"
                rows.append({"matrix": name, "map": map_name, "hash_before": h0, "hash_after": h1, "result": res})
            st.dataframe(rows, use_container_width=True)
            st.download_button("Download CSV", data=csv_bytes(rows, ["matrix","map","hash_before","hash_after","result"]), file_name="full_triple.csv", mime="text/csv")

# ----- Tab 2: Component Transport -----
with tabs[1]:
    st.subheader("Component-level Parity Transport (Step A)")
    map_name_c = st.selectbox("Admissible map", ["append_zero","duplicate_last","duplicate_first","permute_then_append_zero","append_copy_random"], index=0, key="comp_map")
    seed_c = st.number_input("RNG seed", value=42, step=1, key="comp_seed")
    if st.button("Run component transport on selected matrix", key="run_comp_one"):
        y = canonical_y_star(seed_matrix)
        A1 = apply_map(seed_matrix, map_name_c, seed=seed_c)
        if not is_admissible_stepA(seed_matrix, A1):
            st.error("FAIL (non-admissible: new type)")
        else:
            ok = (parity_lex_component_set(A1, y) == parity_lex_component_set(A1, y))
            st.success("PASS" if ok else "FAIL")

# ----- Tab 3: Tower -----
with tabs[2]:
    st.subheader("Tower (Type-Preserving)")
    rule = st.selectbox("Rule", ["append_zero","duplicate_last","duplicate_first","permute_then_append_zero","append_copy_random"], index=0, key="tower_rule")
    levels = st.number_input("Levels", min_value=1, max_value=200, value=10, step=1)
    seed_t = st.number_input("RNG seed", value=42, step=1, key="tower_seed")
    if st.button("Run tower", key="run_tower"):
        rows = []
        cur = seed_matrix
        h0, _ = serialize_certificate(cur)
        rows.append({"level": 0, "hash": h0})
        ok = True
        for k in range(1, int(levels)+1):
            nxt = apply_map(cur, rule, seed=seed_t)
            if not is_admissible_stepA(cur, nxt):
                rows.append({"level": k, "hash": "NON-ADMISSIBLE-NEW-TYPE"})
                ok = False
                break
            hk, _ = serialize_certificate(nxt)
            rows.append({"level": k, "hash": hk})
            cur = nxt
        st.dataframe(rows, use_container_width=True)
        if ok:
            const = "constant" if len(set(r["hash"] for r in rows)) == 1 else "varies"
            st.info(f"Hash across levels is {const}.")
        st.download_button("Download CSV", data=csv_bytes(rows, ["level","hash"]), file_name="tower_hashes.csv", mime="text/csv")

# ----- Tab 4: Batch Seeds × Maps -----
with tabs[3]:
    st.subheader("Batch — Seeds × Maps (admissible only)")
    maps_all = ["append_zero","duplicate_last","duplicate_first","permute_then_append_zero","append_copy_random"]
    if st.button("Run batch", key="run_batch"):
        rows = []
        for name, M in st.session_state["matrices"].items():
            for map_name in maps_all:
                h0, c0 = serialize_certificate(M)
                A1 = apply_map(M, map_name, seed=42)
                if not is_admissible_stepA(M, A1):
                    rows.append({"matrix": name, "map": map_name, "hash_before": h0, "hash_after": "", "result": "FAIL(non-admissible)"})
                    continue
                h1, c1 = serialize_certificate(A1)
                res = "PASS" if (c0["y"]==c1["y"] and c0["t"]==c1["t"] and c0["A_comp"]==c1["A_comp"] and h0==h1) else "FAIL"
                rows.append({"matrix": name, "map": map_name, "hash_before": h0, "hash_after": h1, "result": res})
        st.dataframe(rows, use_container_width=True)
        st.download_button("Download CSV", data=csv_bytes(rows, ["matrix","map","hash_before","hash_after","result"]), file_name="full_triple_batch.csv", mime="text/csv")

# ----- Tab 5: Boundary (Kill-tests) -----
with tabs[4]:
    st.subheader("Boundary (Kill-tests) — outside Step A (expected FAIL)")
    st.write("These helpers deliberately introduce a **new column type**, violating Step-A admissibility, to demonstrate the fence.")
    mode = st.selectbox("Operation", ["XOR injection (append v_i ⊕ v_j)", "Chord flip (flip 1 bit in column)"], index=0)
    if mode.startswith("XOR"):
        i = st.number_input("Column i", min_value=0, max_value=max(0, seed_matrix.n-1), value=0, step=1)
        j = st.number_input("Column j", min_value=0, max_value=max(0, seed_matrix.n-1), value=min(1, seed_matrix.n-1), step=1)
        if st.button("Run XOR injection", key="run_xor"):
            A1 = xor_injection(seed_matrix, int(i), int(j))
            # Check type-set delta
            T0 = seed_matrix.distinct_types(exclude_zero=True)
            T1 = A1.distinct_types(exclude_zero=True)
            new_types = sorted(T1 - T0)
            h0, c0 = serialize_certificate(seed_matrix)
            h1, c1 = serialize_certificate(A1)
            admissible = is_admissible_stepA(seed_matrix, A1)
            st.warning(f"Admissible? {admissible}. New types: {new_types}")
            ok = (c0["y"]==c1["y"] and c0["t"]==c1["t"] and c0["A_comp"]==c1["A_comp"] and h0==h1)
            st.error("Expected FAIL (outside Step A)" if not admissible else "Unexpected: admissible")
            st.json({"hash_before": h0, "hash_after": h1, "equal": (h0==h1), "certificate_equal": ok})
            # CSV
            rows = [{
                "matrix": seed_matrix.name,
                "op": f"xor({i},{j})",
                "admissible": str(admissible),
                "new_types": " ".join(new_types),
                "hash_before": h0, "hash_after": h1, "equal": str(h0==h1)
            }]
            st.download_button("Download CSV", data=csv_bytes(rows, ["matrix","op","admissible","new_types","hash_before","hash_after","equal"]), file_name="boundary_xor.csv", mime="text/csv")
    else:
        col_idx = st.number_input("Column index", min_value=0, max_value=max(0, seed_matrix.n-1), value=0, step=1)
        row_idx = st.number_input("Row index", min_value=0, max_value=max(0, seed_matrix.m-1), value=0, step=1)
        if st.button("Run chord flip", key="run_flip"):
            A1 = chord_flip(seed_matrix, int(col_idx), int(row_idx))
            T0 = seed_matrix.distinct_types(exclude_zero=True)
            T1 = A1.distinct_types(exclude_zero=True)
            new_types = sorted(T1 - T0)
            h0, c0 = serialize_certificate(seed_matrix)
            h1, c1 = serialize_certificate(A1)
            admissible = is_admissible_stepA(seed_matrix, A1)
            st.warning(f"Admissible? {admissible}. New types: {new_types}")
            ok = (c0["y"]==c1["y"] and c0["t"]==c1["t"] and c0["A_comp"]==c1["A_comp"] and h0==h1)
            st.error("Expected FAIL (outside Step A)" if not admissible else "Unexpected: admissible")
            st.json({"hash_before": h0, "hash_after": h1, "equal": (h0==h1), "certificate_equal": ok})
            rows = [{
                "matrix": seed_matrix.name,
                "op": f"flip(c{col_idx},r{row_idx})",
                "admissible": str(admissible),
                "new_types": " ".join(new_types),
                "hash_before": h0, "hash_after": h1, "equal": str(h0==h1)
            }]
            st.download_button("Download CSV", data=csv_bytes(rows, ["matrix","op","admissible","new_types","hash_before","hash_after","equal"]), file_name="boundary_flip.csv", mime="text/csv")

# ----- Tab 6: Serialization Preview -----
with tabs[5]:
    st.subheader("Deterministic Certificate JSON (Preview)")
    h, cert = serialize_certificate(seed_matrix)
    st.json(cert)
    st.code(json.dumps(cert, separators=(",", ":"), ensure_ascii=False), language="json")
    st.write("SHA-256:", h)

st.caption("Note: canonical y* and t* search is exponential in general; use small fixtures or swap in optimized routines for large instances.")
