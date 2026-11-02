"""v1.3.1 runner: run_v13.mjs with the bridge at bridge-python-v131 (env/port_bridge_v131.py) and three scripted checks,
one for each request-contract error the board's agents met, each returned word for word in native_diagnostics. Their
text joins the forced failures that the visible-surface checks scan for hidden-world identity and host paths.
run_v13.mjs is unchanged; the board ran on it.
  python3 -I env/port_runner_v131.py"""
from pathlib import Path

ENV = Path(__file__).resolve().parent
src = (ENV / "run_v13.mjs").read_text()
dst = ENV / "run_v131.mjs"
assert not dst.exists(), "run_v131.mjs exists; it is built once"

EDITS = [
    ("process.env.NDFEB_PYTHON=path.join(v12,'bridge-python/bin/python');",
     "process.env.NDFEB_PYTHON=path.join(study,'bridge-python-v131/bin/python');   // v1.3.1: the legacy laboratory reports its error"),
    (" const trees={bridge_package:path.join(v12,'bridge-python/lib/python3.13/site-packages/ndfeb_sps'),",
     " const trees={bridge_package:path.join(study,'bridge-python-v131/lib/python3.13/site-packages/ndfeb_sps'),"),
    ("  fs.writeFileSync(path.join(ep,'SCRIPTED_FORCED_FAILURES.json'),JSON.stringify(forced,null,2));",
     """  // v1.3.1: the request-contract errors the board's agents met, each returned word for word (free, no reading)
  const legacy=await call('request_simulation',{request_id:'S-LEGACY',rationale:'a new preparation in the legacy laboratory (simulation omitted)',
   request:{...make,preparations:[{...prep,preparation_id:'S-L'}],specimens:[{...make.specimens[0],specimen_id:'S-L1',preparation_id:'S-L',cycle_id:'S-LC',
    conditions:{...make.specimens[0].conditions,measurement_context:{piece_id:'P1',density_method:'geometric',field_set_id:'S-ACQ-L'}}}]}});
  assert.equal(legacy.status,'execution_failed',JSON.stringify(legacy).slice(0,1500));
  assert.match(JSON.stringify(legacy.native_diagnostics),/this conditional family implements only the accepted preparation/);
  pass('legacy_laboratory_error_returned_word_for_word');
  const noAcq=await call('request_simulation',{request_id:'S-NOACQ',simulation:'magnetic',rationale:'a density reading without an acquisition ID',
   request:{...measure,specimens:[{...measure.specimens[0],conditions:{...measure.specimens[0].conditions,
    measurement_context:{piece_id:'P1',density_method:'archimedes_saturated'}}}]}});
  assert(!noAcq.delivery_id,JSON.stringify(noAcq).slice(0,1500));
  assert.match(JSON.stringify(noAcq.native_diagnostics),/requires measurement_context\\.field_set_id/);
  pass('missing_acquisition_id_error_returned_word_for_word');
  const undeclared=await call('request_simulation',{request_id:'S-UNDECLARED',simulation:'magnetic',rationale:'a body on REF-P without declaring it',
   request:{...make,preparations:[],specimens:[{...make.specimens[0],specimen_id:'S-R1',preparation_id:'REF-P',cycle_id:'S-RC',
    conditions:{...make.specimens[0].conditions,measurement_context:{piece_id:'P1',density_method:'geometric',field_set_id:'S-ACQ-R'}}}]}});
  assert(!undeclared.delivery_id,JSON.stringify(undeclared).slice(0,1500));
  assert.match(JSON.stringify(undeclared.native_diagnostics),/declare preparation 'REF-P' with its conditions in request\\.preparations/);
  pass('undeclared_preparation_error_returned_word_for_word');
  forced.push(JSON.stringify(legacy),JSON.stringify(noAcq),JSON.stringify(undeclared));
  fs.writeFileSync(path.join(ep,'SCRIPTED_FORCED_FAILURES.json'),JSON.stringify(forced,null,2));"""),
]
for a, b in EDITS:
    assert src.count(a) == 1, (a[:70], src.count(a))
    src = src.replace(a, b)
dst.write_text(src)
print("wrote", dst.name, "with", len(EDITS), "edits")
