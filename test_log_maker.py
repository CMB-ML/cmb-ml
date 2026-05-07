# test_log_maker.py
#   python test_log_maker.py

from pathlib import Path
import zipfile
import tempfile
import shutil

from cmbml.core.log_maker import LogMaker


# Fill in the following based on your local system
PYILC_SCRIPT    = "/home/jim/Code/cmb-ml-pyilc/main_pyilc_predict.py"
D2PS_DIRECTORY  = "/home/jim/Code/e-d2ps_a"
PYILC_DIRECTORY = "/home/jim/Code/cmb-ml-pyilc"

# ── 1. Test: _find_installed_cmbml_root ──────────────────────────────
print("=== 1. Finding installed cmbml root ===")
root = LogMaker._find_installed_cmbml_root()
print(f"  cmbml root: {root}")
assert root is not None and root.exists(), "FAIL: cmbml root not found"
print("  PASS")

# ── 2. Test: _script_imports_package ─────────────────────────────────
print("\n=== 2. Checking import detection ===")
# Use the pyilc main script — we know it imports from cmbml
pyilc_script = Path(PYILC_SCRIPT)
result = LogMaker._script_imports_package(pyilc_script, "cmbml")
print(f"  pyilc main imports cmbml: {result}")
assert result is True, "FAIL: should detect cmbml import"

result2 = LogMaker._script_imports_package(pyilc_script, "numpy")
print(f"  pyilc main imports numpy (expect False): {result2}")
print("  PASS")

# ── 3. Test: _find_local_package_roots ───────────────────────────────
print("\n=== 3. Finding local package roots ===")
d2ps_dir = Path(D2PS_DIRECTORY)
roots2 = LogMaker._find_local_package_roots(d2ps_dir)
print(f"  Found roots in e-d2ps: {[r.name for r in roots2]}")
assert any(r.name == "d2ps_nn" for r in roots2), "FAIL: d2ps_nn not found"
print("  PASS")

pyilc_dir = Path(PYILC_DIRECTORY)
roots = LogMaker._find_local_package_roots(pyilc_dir)
print(f"  Found roots: {[r.name for r in roots]}")
assert any(r.name == "pyilc_local" for r in roots), "FAIL: pyilc_local not found"
print("  PASS")

# ── 4. Test: _trace_imports on installed cmbml ───────────────────────
print("\n=== 4. Tracing imports from installed cmbml ===")
cmbml_root = LogMaker._find_installed_cmbml_root()
files_found = LogMaker._find_installed_imports(pyilc_script, cmbml_root)
print(f"  Files found in cmbml: {len(files_found)}")
assert len(files_found) > 0, "FAIL: no cmbml files traced"
for f in sorted(files_found)[:5]:
    print(f"    {f}")
print("  ...")
print("  PASS")

# ── 5. Test: _trace_imports on local pyilc_local ─────────────────────
print("\n=== 5. Tracing imports from pyilc_local ===")
pyilc_local_root = pyilc_dir / "pyilc_local"
files_local = LogMaker._find_local_imports(pyilc_script, pyilc_local_root)
print(f"  Files found in pyilc_local: {len(files_local)}")
assert len(files_local) > 0, "FAIL: no pyilc_local files traced"
for f in sorted(files_local)[:5]:
    print(f"    {f}")
print("  PASS")

# ── 6. Test: _archive_files produces a valid, browsable zip ──────────
print("\n=== 6. Archiving to zip ===")
with tempfile.TemporaryDirectory() as tmpdir:
    zip_path = Path(tmpdir) / "cmbml.zip"
    LogMaker._archive_files(files_found, cmbml_root.parent, zip_path)
    assert zip_path.exists(), "FAIL: zip not created"
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    print(f"  Zip contains {len(names)} files")
    assert len(names) > 0, "FAIL: zip is empty"
    # Paths inside zip should start with cmbml/
    bad = [n for n in names if not n.startswith("cmbml/")]
    assert not bad, f"FAIL: unexpected paths in zip: {bad[:3]}"
    print(f"  Sample entries:")
    for n in sorted(names)[:5]:
        print(f"    {n}")
    print("  PASS")

print("\n=== All tests passed ===")
