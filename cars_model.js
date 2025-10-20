// cars_model.js (English)
export function buildRegModel(seqLen, featPerStep) {
  const inp = tf.input({shape:[seqLen, featPerStep]});
  const g1 = tf.layers.gru({units:64, returnSequences:true}).apply(inp);
  const g2 = tf.layers.gru({units:64}).apply(g1);
  const d1 = tf.layers.dense({units:64, activation:'relu'}).apply(g2);
  const out = tf.layers.dense({units:1, activation:'linear'}).apply(d1);
  const model = tf.model({inputs: inp, outputs: out});
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'meanSquaredError',
    metrics: ['mae']
  });
  return model;
}

export async function fitModel(model, X, y, {epochs=80, batchSize=32, onEpoch}={}) {
  return await model.fit(X, y, {
    epochs, batchSize, shuffle: true,
    callbacks: { onEpochEnd: async (ep, logs)=> onEpoch && onEpoch(ep, logs) }
  });
}

export async function evaluate(model, X, y, returnPred=false) {
  const pred = model.predict(X);
  const yPred = await pred.data();
  const yTrue = (await y.data());
  // MAE, RMSE, R²
  let ae=0, se=0, ssTot=0;
  const meanY = yTrue.reduce((a,b)=>a+b,0)/yTrue.length;
  for (let i=0;i<yTrue.length;i++) {
    const e = yPred[i] - yTrue[i];
    ae += Math.abs(e);
    se += e*e;
    const d = yTrue[i]-meanY;
    ssTot += d*d;
  }
  const mae = ae/yTrue.length;
  const rmse = Math.sqrt(se/yTrue.length);
  const r2 = 1 - (se/(ssTot+1e-8));
  if (returnPred) return {mae, rmse, r2, yTrue: Array.from(yTrue), yPred: Array.from(yPred)};
  return {mae, rmse, r2};
}
