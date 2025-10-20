// cars_loader.js (English)
// Parse UCI‑like cars CSV where missing values are '?'.
// Supports ',' and ';' delimiters. Target column is 'price'.

export async function parseCarsCSV(file) {
  const text = await file.text();
  const sep = text.includes(';') && !text.includes(',') ? ';' : ',';
  const lines = text.trim().split(/\r?\n/);
  const header = lines.shift().split(sep).map(s=>s.trim());
  const col = Object.fromEntries(header.map((h,i)=>[h,i]));
  if (!('price' in col)) throw new Error("Column 'price' not found");
  const rows = lines.map(line => {
    const cells = line.split(sep).map(s=>s.trim());
    const obj = {};
    header.forEach((h,i)=> obj[h] = cells[i] ?? '');
    return obj;
  });
  return rows;
}

function isNumericStrict(v) {
  if (v===null || v===undefined) return false;
  if (v==='' || v==='?') return false;
  const x = Number(v);
  return isFinite(x);
}

export function buildDataset(rows, targetName='price') {
  // Decide numeric vs categorical columns
  const headers = Object.keys(rows[0]);
  const numCols = [], catCols = [];
  for (const h of headers) {
    if (h===targetName) continue;
    // heuristic: if >=80% of values are numeric -> numeric
    let cnt=0, num=0;
    for (const r of rows) { cnt++; if (isNumericStrict(r[h])) num++; }
    if (num/cnt >= 0.8) numCols.push(h); else catCols.push(h);
  }

  // Category dictionaries
  const categories = {};
  for (const h of catCols) {
    const set = new Set();
    for (const r of rows) {
      const v = (r[h] && r[h] !== '?') ? r[h] : '__NA__';
      set.add(v);
    }
    categories[h] = Array.from(set.values()).sort();
  }

  // Min–max scalers for numeric columns
  const scalers = {};
  for (const h of numCols) {
    let mn=Infinity, mx=-Infinity;
    for (const r of rows) {
      if (!isNumericStrict(r[h])) continue;
      const x = Number(r[h]);
      if (x<mn) mn=x; if (x>mx) mx=x;
    }
    if (!isFinite(mn) || !isFinite(mx) || mx===mn) { mn=0; mx=1; }
    scalers[h] = {min: mn, max: mx};
  }

  // Build X, y
  const X = [];
  const y = [];
  const featureNames = [];
  featureNames.push(...numCols);
  for (const h of catCols) {
    for (const v of categories[h]) featureNames.push(`${h}==${v}`);
  }

  for (const r of rows) {
    if (!isNumericStrict(r[targetName])) continue;
    const row = [];
    // numeric
    for (const h of numCols) {
      const sc = scalers[h];
      const x = isNumericStrict(r[h]) ? Number(r[h]) : sc.min;
      row.push( (x - sc.min) / (sc.max - sc.min + 1e-8) );
    }
    // categorical → one‑hot
    for (const h of catCols) {
      const cats = categories[h];
      const v = (r[h] && r[h] !== '?') ? r[h] : '__NA__';
      for (const c of cats) row.push(c===v ? 1 : 0);
    }
    X.push(row);
    y.push(Number(r[targetName]));
  }

  return { X, y, featureNames, scalers, categories };
}

export function chunkToSequence(Xflat, seqLen=4, stepSizeAuto=null) {
  const F = Xflat[0].length;
  const stepSize = stepSizeAuto ?? Math.ceil(F/seqLen);
  const seqs = Xflat.map(vec => {
    const padded = vec.slice();
    while (padded.length < stepSize*seqLen) padded.push(0);
    const arr = [];
    for (let t=0; t<seqLen; t++) {
      arr.push(padded.slice(t*stepSize, (t+1)*stepSize));
    }
    return arr;
  });
  return { seqs, stepSize };
}
