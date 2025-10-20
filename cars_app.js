// cars_app.js (English UI)
import { parseCarsCSV, buildDataset, chunkToSequence } from './cars_loader.js';
import { buildRegModel, fitModel, evaluate } from './cars_model.js';

const st = {
  Xtrain: null, ytrain: null, Xtest: null, ytest: null,
  model: null,
  seqLen: 4,
  batch: 32,
  epochs: 80
};

const fileInput = document.getElementById('fileInput');
const fileName = document.getElementById('fileName');
const statusEl = document.getElementById('status');
const trainBtn = document.getElementById('trainBtn');
const evalBtn = document.getElementById('evalBtn');
const progressEl = document.getElementById('progress');
const epochText = document.getElementById('epochText');
const summary = document.getElementById('summary');
const saveBtn = document.getElementById('saveBtn');
const loadBtn = document.getElementById('loadBtn');

fileInput.addEventListener('change', async (e)=> {
  const f = e.target.files?.[0];
  if (!f) return;
  fileName.textContent = f.name;
  statusEl.textContent = 'Parsing CSV…';
  try {
    const rows = await parseCarsCSV(f);
    const { X, y, featureNames } = buildDataset(rows, 'price');

    // 80/20 split with a deterministic shuffle
    const idx = [...X.keys()];
    let seed = 42;
    function rnd(){ seed = (seed*1664525 + 1013904223) % 4294967296; return seed/4294967296; }
    idx.sort((a,b)=> rnd() - 0.5);
    const split = Math.floor(idx.length*0.8);
    const trainIdx = idx.slice(0,split), testIdx = idx.slice(split);

    const Xt = trainIdx.map(i=>X[i]), yt = trainIdx.map(i=>y[i]);
    const Xv = testIdx.map(i=>X[i]), yv = testIdx.map(i=>y[i]);

    // Make "pseudo-sequences" for the GRU
    const { seqs: XtSeq, stepSize } = chunkToSequence(Xt, st.seqLen);
    const { seqs: XvSeq } = chunkToSequence(Xv, st.seqLen, stepSize);

    st.Xtrain = tf.tensor3d(XtSeq);
    st.ytrain = tf.tensor2d(yt, [yt.length, 1]);
    st.Xtest  = tf.tensor3d(XvSeq);
    st.ytest  = tf.tensor2d(yv, [yv.length, 1]);

    st.featureDim = stepSize;
    statusEl.textContent = `Ready: ${rows.length} rows, ${featureNames.length} features (steps: ${st.seqLen}, step size: ${stepSize}).`;
    trainBtn.disabled = false;
  } catch (err) {
    console.error(err);
    statusEl.textContent = 'Error: ' + err.message;
  }
});

trainBtn.addEventListener('click', async ()=> {
  trainBtn.disabled = true;
  statusEl.textContent = 'Building model…';
  st.model = buildRegModel(st.seqLen, st.featureDim);
  statusEl.textContent = 'Training…';
  progressEl.value = 0; epochText.textContent = '';
  await fitModel(st.model, st.Xtrain, st.ytrain, {
    epochs: st.epochs, batchSize: st.batch,
    onEpoch: (ep, logs)=> {
      progressEl.value = (ep+1)/st.epochs;
      epochText.textContent = `Epoch ${ep+1}/${st.epochs} — loss ${logs.loss.toFixed(2)} — MAE ${logs.mae.toFixed(2)}`;
    }
  });
  statusEl.textContent = 'Trained.';
  evalBtn.disabled = false;
  saveBtn.disabled = false;
});

evalBtn.addEventListener('click', async ()=> {
  const mTrain = await evaluate(st.model, st.Xtrain, st.ytrain);
  const mTest  = await evaluate(st.model, st.Xtest,  st.ytest, true);
  appendMetrics('train', mTrain);
  appendMetrics('test', mTest);
  summary.textContent = `Test MAE: ${mTest.mae.toFixed(2)}, RMSE: ${mTest.rmse.toFixed(2)}, R²: ${mTest.r2.toFixed(3)} (n=${st.ytest.shape[0]}).`;
  drawScatter(mTest.yTrue, mTest.yPred);
});

saveBtn.addEventListener('click', async ()=> {
  if (!st.model) return;
  await st.model.save('downloads://tfjs_cars_gru_regression_en');
});
loadBtn.addEventListener('click', async ()=> {
  try {
    const model = await tf.loadLayersModel('indexeddb://tfjs_cars_gru_regression_en');
    st.model = model;
    evalBtn.disabled = false; saveBtn.disabled = false;
    statusEl.textContent = 'Weights loaded from IndexedDB.';
  } catch(e) {
    statusEl.textContent = 'No saved weights found. Train and click “Save Weights”.';
  }
});

function appendMetrics(split, m) {
  const tb = document.querySelector('#metricsTbl tbody');
  const tr = document.createElement('tr');
  function td(x){ const e=document.createElement('td'); e.textContent=String(x); return e; }
  tr.appendChild(td(split));
  tr.appendChild(td(m.mae.toFixed(2)));
  tr.appendChild(td(m.rmse.toFixed(2)));
  tr.appendChild(td(m.r2.toFixed(3)));
  tb.appendChild(tr);
}

function drawScatter(yTrue, yPred) {
  const canvas = document.getElementById('scatter');
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.clientWidth;
  const h = canvas.height = canvas.clientHeight;
  ctx.clearRect(0,0,w,h);
  // axes
  ctx.strokeStyle = '#2a3872'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(40,h-30); ctx.lineTo(w-10,h-30); ctx.lineTo(w-10,20); ctx.stroke();
  // scale
  const minY = Math.min(...yTrue, ...yPred);
  const maxY = Math.max(...yTrue, ...yPred);
  function xscale(v){ return 40 + ( (v - minY)/(maxY - minY + 1e-8) ) * (w-60); }
  function yscale(v){ return (h-30) - ( (v - minY)/(maxY - minY + 1e-8) ) * (h-60); }
  // y=x line
  ctx.strokeStyle = '#6aa3ff'; ctx.beginPath();
  ctx.moveTo(xscale(minY), yscale(minY)); ctx.lineTo(xscale(maxY), yscale(maxY)); ctx.stroke();
  // points
  ctx.fillStyle = '#e7ecff';
  for (let i=0;i<yTrue.length;i++) {
    const x = xscale(yTrue[i]);
    const y = yscale(yPred[i]);
    ctx.beginPath(); ctx.arc(x,y,2,0,Math.PI*2); ctx.fill();
  }
  // labels
  ctx.fillStyle = '#aab2d5'; ctx.font = '12px Inter, sans-serif';
  ctx.fillText('Actual', 44, h-10);
  ctx.save(); ctx.translate(w-6, 24); ctx.rotate(-Math.PI/2); ctx.fillText('Predicted', 0, 0); ctx.restore();
}
