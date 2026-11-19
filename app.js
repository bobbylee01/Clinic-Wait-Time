import { predict } from "./inference.mjs";

const byId = id => document.getElementById(id);
const minutes = value => new Intl.NumberFormat("en", {
  maximumFractionDigits: 1, minimumFractionDigits: 1,
}).format(value) + " min";
const weekdays = { Mon: "Monday", Tue: "Tuesday", Wed: "Wednesday", Thu: "Thursday", Fri: "Friday" };

function configureSimulator(model) {
  const day = byId("day_of_week");
  for (const value of model.days) day.add(new Option(weekdays[value], value));
  for (const key of model.features) {
    const input = byId(key);
    const [min, max] = model.bounds[key];
    input.min = min;
    input.max = max;
    input.value = Math.round((min + max) / 2);
  }
  function render() {
    const inputs = { day_of_week: day.value };
    for (const key of model.features) {
      inputs[key] = byId(key).valueAsNumber;
      byId(key + "-value").textContent = key === "hour" ? inputs[key] + ":00" : inputs[key];
    }
    const wait = predict(model, inputs);
    byId("prediction").textContent = minutes(wait);
    const staff = inputs.staff_on_shift;
    const [min, max] = model.bounds.staff_on_shift;
    const comparison = byId("comparison");
    comparison.textContent = "";
    if (min === max) return;
    const alternative = staff < max ? staff + 1 : staff - 1;
    const otherWait = predict(model, { ...inputs, staff_on_shift: alternative });
    comparison.textContent = "With " + alternative + " staff and the same hour, day, and queue, the predicted wait is " + minutes(otherWait) + ".";
  }
  byId("scenario").addEventListener("input", render);
  byId("scenario").addEventListener("submit", event => event.preventDefault());
  const { trainCount, testCount, baselineMedian, modelMae, baselineMae, mistake } = model.evaluation;
  byId("method").textContent = "The model trained on " + trainCount + " rows. Both predictions were evaluated on the same " + testCount + " held-out rows. The baseline always predicted the training median of " + minutes(baselineMedian) + ".";
  byId("network-mae").textContent = minutes(modelMae);
  byId("baseline-mae").textContent = minutes(baselineMae);
  const inputs = mistake.inputs;
  const scenario = weekdays[inputs.day_of_week] + " at " + inputs.hour + ":00 with " + inputs.staff_on_shift + " staff and " + inputs.patients_in_queue + " patients in queue";
  const mistakeText = scenario + ". The model predicted " + minutes(mistake.prediction) + " and the recorded wait was " + minutes(mistake.actual) + ", an absolute error of " + minutes(mistake.error) + ".";
  const matchingWaits = mistake.matchingWaits ?? [];
  const observedVariation = new Set(matchingWaits).size > 1
    ? " Visits with these exact inputs recorded waits of " + matchingWaits.map(minutes).join(", ") +
      ". The network gives the same estimate for identical inputs, so it cannot distinguish these visits."
    : "";
  const hypothesis = " In real clinic data, recording service duration or visit complexity might help explain differences in wait times. The exercise data does not identify the cause of this error.";
  byId("mistake").textContent = mistakeText + observedVariation + hypothesis;
  byId("try-mistake").addEventListener("click", () => {
    for (const [key, value] of Object.entries(inputs)) byId(key).value = value;
    render();
    day.focus();
  });
  byId("try-mistake").disabled = false;
  byId("controls").disabled = false;
  render();

  return [
    "The Clinic Wait-Time Simulator explores how staffing and queue size affect predicted waits.",
    "The data consists of " + (trainCount + testCount) + " simulated visits created by the Neural Network Explorer exercise authors.",
    "A scikit-learn neural network predicts wait in minutes from hour, weekday, staff on shift, and queue size.",
    "The network test MAE was " + minutes(modelMae) + " and the median baseline test MAE was " + minutes(baselineMae) + ". Lower MAE is better.",
    "Only this synthetic dataset was evaluated, so performance on any other dataset is unknown.",
    "Question: Where did your data come from? Answer: The Neural Network Explorer exercise authors generated all " + (trainCount + testCount) + " simulated visits.",
    "Question: How did your model compare with the baseline? Answer: On the same " + testCount + " held-out visits, the neural network MAE was " + minutes(modelMae) + " and the baseline MAE was " + minutes(baselineMae) + ".",
    "Question: How well did this model work on a dataset you never tested? Answer: The guide has no evidence about performance on an unevaluated dataset.",
  ].join("\n");
}

function configureChat(guide) {
  const enable = byId("enable-chat");
  const status = byId("chat-status");
  const form = byId("chat-form");
  const question = byId("question");
  const send = byId("send");
  const log = byId("messages");
  let engine;
  const messages = [{
    role: "system",
    content: "You are the Clinic Wait-Time Simulator exhibit assistant. Answer only from the supplied guide. If the guide lacks an answer, say that the guide has no evidence. Avoid clinical advice.",
  }];
  if (!navigator.gpu) {
    status.textContent = "WebGPU is unavailable here. You can still use the simulator.";
    return;
  }

  enable.disabled = false;
  enable.addEventListener("click", async () => {
    enable.disabled = true;
    status.textContent = "Checking WebGPU...";
    try {
      if (!await navigator.gpu.requestAdapter()) throw new Error("No WebGPU adapter available");
      status.textContent = "Loading WebLLM...";
      const { CreateMLCEngine } = await import("https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.79/lib/index.js");
      engine = await CreateMLCEngine("Qwen2.5-0.5B-Instruct-q4f32_1-MLC", {
        initProgressCallback: ({ text }) => { status.textContent = text; },
      });
      enable.hidden = true;
      status.textContent = "Chat ready.";
      form.hidden = false;
      question.focus();
    } catch (error) {
      status.textContent = "Chat could not load. You can still use the simulator. " + error.message;
      enable.disabled = false;
    }
  });

  function append(speaker, text) {
    const line = document.createElement("p");
    line.textContent = speaker + ": " + text;
    log.append(line);
    log.scrollTop = log.scrollHeight;
  }

  form.addEventListener("submit", async event => {
    event.preventDefault();
    const text = question.value.trim();
    if (!engine || !text || send.disabled) return;
    send.disabled = true;
    question.value = "";
    append("You", text);
    status.textContent = "Generating answer...";
    try {
      const reply = await engine.chat.completions.create({
        messages: [messages[0], {
          role: "user",
          content: "Exhibit guide:\n" + guide + "\n\nAnswer the question using only the guide above.\nQuestion: " + text,
        }],
        temperature: 0,
        max_tokens: 180,
      });
      const answer = reply.choices[0]?.message.content?.trim();
      if (!answer) throw new Error("Empty response");
      append("Assistant", answer);
      status.textContent = "Chat ready.";
    } catch (error) {
      status.textContent = "Chat could not answer: " + error.message;
    } finally {
      send.disabled = false;
      question.focus();
    }
  });
}

try {
  const response = await fetch("./model/clinic-model.json");
  if (!response.ok) throw new Error("Model file unavailable");
  configureChat(configureSimulator(await response.json()));
} catch (error) {
  byId("prediction").textContent = "Model unavailable";
  byId("comparison").textContent = "Run train_model.py and open this page through a local HTTP server.";
  console.error(error);
}



