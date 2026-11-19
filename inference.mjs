export function predict(model, inputs) {
  let values = model.features.map((key, index) =>
    (inputs[key] - model.mean[index]) / model.scale[index]);
  values.push(...model.days.map(day => Number(day === inputs.day_of_week)));
  for (const [index, { weights, bias }] of model.layers.entries()) {
    const hidden = index < model.layers.length - 1;
    values = bias.map((offset, unit) => {
      const sum = values.reduce((total, value, row) => total + value * weights[row][unit], offset);
      return hidden ? Math.max(0, sum) : sum;
    });
  }
  return Math.max(0, values[0] * model.targetScale + model.targetMean);
}

