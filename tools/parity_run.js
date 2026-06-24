// Reads a snapshot JSON on stdin, runs the Umeh JS, prints the result JSON.
// Used by tools/parity_check.py to compare against the Python reference.
const path = require('path');
const Umeh = require(path.join(__dirname, '..', 'docs', 'umeh.js'));

let buf = '';
process.stdin.on('data', d => (buf += d));
process.stdin.on('end', () => {
  const s = JSON.parse(buf);
  const res = Umeh.computeUmeh(s.closes, s.volumes, s.spot, s.order_flow_r, s.now_ms);
  res.slow = Umeh.nwachukwuSlow(s.closes, s.spot);
  process.stdout.write(JSON.stringify(res));
});
