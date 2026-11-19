# Clinic Wait-Time Simulator

Explore how staffing and queue size change a clinic wait estimate. This paired student activity uses the 2,000 simulated visits supplied by the [Neural Network Explorer exercise](https://snsie.github.io/neural-networks/exercise.html). The unchanged download is in [data/clinic-wait-times.csv](data/clinic-wait-times.csv).

## Run locally

Python 3.12 was used for training. The trained model is included, so you can preview the exhibit directly with a static server. From the project directory, run one of these commands and open http://localhost:8000/.

~~~powershell
python -m http.server 8000
# Or use the project environment if it is already set up:
.venv/Scripts/python.exe -m http.server 8000
~~~

Move the sliders or change the weekday. The simulator compares the current estimate with a different staff count while holding the other inputs constant. Use **Try this scenario** to replay the largest held-out error.

## Train and evaluate

~~~powershell
# Use Python 3.12 for this command.
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe train_model.py
~~~

If you use uv, run **uv venv --python 3.12** and **uv pip install -r requirements.txt** instead of the first two commands. Keep the exported [model/clinic-model.json](model/clinic-model.json) with the site.

[train_model.py](train_model.py) uses a seeded 80/20 split. It fits the numerical and target scalers and the scikit-learn MLP on 1,600 training visits. The network uses hour, weekday, staff on shift, and patients in queue to estimate wait in minutes. The export includes the fitted preprocessing and network weights. [inference.mjs](inference.mjs) runs the same calculations on new inputs and floors wait estimates at zero, just as evaluation does.

On the same 400 held-out visits, the network's mean absolute error was **4.95 minutes** and the median baseline's was **44.60 minutes**. The baseline always predicted **59.5 minutes**, the training median. The largest test error was **18.81 minutes**. On Wednesday at 9:00, with three staff and 19 patients in queue, the network predicted **115.91 minutes** and the CSV recorded **97.10 minutes**. Another visit with the same four inputs recorded **109.10 minutes**. The network gives both visits the same estimate because its inputs are identical. Service duration and visit complexity could help distinguish real clinic visits, but the exercise data does not identify the cause of this error.

## Deploy to GitHub Pages

Push the project to a GitHub repository. Keep **index.html**, **style.css**, **app.js**, **inference.mjs**, **.nojekyll**, **data/clinic-wait-times.csv**, and **model/clinic-model.json** at their current relative paths. Include **train_model.py** in the repository for submission.

In the repository, open **Settings > Pages**. Under **Build and deployment**, choose **Deploy from a branch**, select the branch containing the project and **/(root)**, then save. When Pages reports a successful deployment, use the HTTPS URL shown there. [GitHub Pages publishing instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)

## Chat and submission checklist

1. On the device you will use for the demo, open the site over localhost or HTTPS in a WebGPU-capable browser. Select **Enable chat** and wait for WebLLM to download and initialize the model. Internet access is required for the initial load. The simulator is available independently of chat.
2. Check the chatbot against the exhibit using these assignment questions:
   - **Where did your data come from?** The Neural Network Explorer exercise authors generated the simulated visits.
   - **How did your model compare with the baseline?** Compare its 4.95-minute test MAE with the baseline's 44.60-minute test MAE.
   - **How well did this model work on a dataset you never tested?** The guide has no evidence about performance on other datasets.
3. Review the highlighted mistake and chatbot answers with your partner. Revise the guide in [app.js](app.js) if the chatbot gives unsupported answers, then retest.
4. On the published HTTPS URL, move a slider, replay the largest error, and download the CSV. On the demo device, use a fresh browser profile to check chat loading progress and all three answers. Try a browser with WebGPU disabled or block the chat download and verify that the sliders still update the predicted wait.
5. Submit the direct exhibit URL and the source repository URL containing **train_model.py**, then answer the Canvas assignment questions.
