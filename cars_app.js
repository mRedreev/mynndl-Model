// cars_app.js
import { parseCarsCSV, buildDataset, chunkToSequence } from './cars_loader.js';
import { buildRegModel, fitModel, evaluate } from './cars_model.js';

const st = {
  data: null,
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
  statusEl.textContent = 'Читаю CSV…';
  try {
    const rows = await parseCarsCSV(f);
    const { X, y, featureNames, scalers, categories } = buildDataset(rows, 'price');
    // Разбиваем на train/test (80/20), случайно но детерминированно
    const idx = [...X.keys()];
    // простая перетасовка фикс. зерном
    let seed = 42;
    function rnd(){ seed = (seed*1664525 + 1013904223) % 4294967296; return seed/4294967296; }
    idx.sort((a,b)=> rnd() - 0.5);
    const split = Math.floor(idx.length*0.8);
    const trainIdx = idx.slice(0,split), testIdx = idx.slice(split);

    const Xt = trainIdx.map(i=>X[i]), yt = trainIdx.map(i=>y[i]);
    const Xv = testIdx.map(i=>X[i]), yv = testIdx.map(i=>y[i]);

    // Превращаем плоские признаки в «последовательность» из seqLen шагов
    const seqLen = st.seqLen;
    const { seqs: XtSeq, stepSize } = chunkToSequence(Xt, seqLen);
    const { seqs: XvSeq } = chunkToSequence(Xv, seqLen, stepSize);

    st.Xtrain = tf.tensor3d(XtSeq);
    st.ytrain = tf.tensor2d(yt, [yt.length, 1]);
    st.Xtest  = tf.tensor3d(XvSeq);
    st.ytest  = tf.tensor2d(yv, [yv.length, 1]);

    st.featureDim = stepSize;
    statusEl.textContent = `Готово: ${rows.length} строк, признаков: ${featureNames.length} (шагов: ${seqLen}, размер шага: ${stepSize}).`;
    trainBtn.disabled = false;
  } catch (err) {
    console.error(err);
    statusEl.textContent = 'Ошибка: ' + err.message;
  }
});

trainBtn.addEventListener('click', async ()=> {
  trainBtn.disabled = true;
  statusEl.textContent = 'Строю модель…';
  st.model = buildRegModel(st.seqLen, st.featureDim);
  statusEl.textContent = 'Обучение…';
  progressEl.value = 0; epochText.textContent = '';
  await fitModel(st.model, st.Xtrain, st.ytrain, {
    epochs: st.epochs, batchSize: st.batch,
    onEpoch: (ep, logs)=> {
      progressEl.value = (ep+1)/st.epochs;
      epochText.textContent = `Epoch ${ep+1}/${st.epochs} — loss ${logs.loss.toFixed(2)} — MAE ${logs.mae.toFixed(2)}`;
    }
  });
  statusEl.textContent = 'Обучено.';
  evalBtn.disabled = false;
  saveBtn.disabled = false;
});

evalBtn.addEventListener('click', async ()=> {
  const mTrain = await evaluate(st.model, st.Xtrain, st.ytrain);
  const mTest  = await evaluate(st.model, st.Xtest,  st.ytest, true);
  fillMetrics('train', mTrain);
  fillMetrics('test', mTest);
  summary.textContent = `MAE тест: ${mTest.mae.toFixed(2)}, RMSE: ${mTest.rmse.toFixed(2)}, R²: ${mTest.r2.toFixed(3)} (n=${st.ytest.shape[0]}).`;
  drawScatter(mTest.yTrue, mTest.yPred);
});

saveBtn.addEventListener('click', async ()=> {
  if (!st.model) return;
  await st.model.save('downloads://tfjs_cars_gru_regression');
});
loadBtn.addEventListener('click', async ()=> {
  try {
    const model = await tf.loadLayersModel('indexeddb://tfjs_cars_gru_regression');
    st.model = model;
    evalBtn.disabled = false; saveBtn.disabled = false;
    statusEl.textContent = 'Веса загружены из IndexedDB.';
  } catch(e) {
    statusEl.textContent = 'Нет сохранённых весов. Обучите и нажмите «Сохранить веса».';
  }
});

function fillMetrics(tag, m) {
  const tb = document.querySelector('#metricsTbl tbody');
  const tr = document.createElement('tr');
  function td(x){ const e=document.createElement('td'); e.textContent=x; return e; }
  tr.appendChild(td(tag));
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
}
