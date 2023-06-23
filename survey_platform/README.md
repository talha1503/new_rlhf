# Preference Survey Web App

This application provides a simple GUI for collecting human preferences within the RLHF pipeline.

## Installation

To install, run the following

```bash
conda create -n survey python=3.7.16
conda activate survey
pip install -r requirements.txt
```

## Running a survey

To run a survey, simply run

```bash
make begin_session
```

which will open an instance of the Streamlit app automatically in your web browser.

Read the instructions and and begin the survey. After completing the survey, the results
may be found in `./data/survey_results.csv`.
