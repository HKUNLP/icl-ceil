# To run use `streamlit run prediction_view.py`
import streamlit as st
import glob
import json
import os
import sys
import importlib

sys.path.append(".")


def get_task_meta(name):
    return importlib.import_module('src.dataset_readers.dataset_wrappers.{}'.format(name)).DatasetWrapper


st.set_page_config(layout="wide")

files = glob.glob("output/*/*/*/*/*/pred.json", recursive=True)
files.sort(key=os.path.getmtime)
files = files[::-1]

m_t_l_run = {}

for f in files:
    splits = f.split('/')[1:-1]
    if len(splits) == 5:
        method, task, lm_1, lm_2, run = splits
        lm = f"{lm_1}/{lm_2}"
    else:
        method, task, lm, run = splits

    m_t_l_run.setdefault(method, {})
    m_t_l_run[method].setdefault(task, {})
    m_t_l_run[method][task].setdefault(lm, [])
    m_t_l_run[method][task][lm].append(run)

col_method, col_task, col_lm, col_run = st.columns([1, 1, 1, 1])
with col_method:
    method = st.selectbox("Methods", options=m_t_l_run.keys())
with col_task:
    task = st.selectbox("Tasks", options=m_t_l_run[method].keys())
with col_lm:
    lm = st.selectbox("LMs", options=m_t_l_run[method][task].keys())
with col_run:
    run = st.selectbox("Runs", options=m_t_l_run[method][task][lm])


@st.experimental_singleton
def load_json(file):
    with open(file) as f:
        data = json.load(f)
    return data


filename = f'output/{method}/{task}/{lm}/{run}/pred.json'
data = load_json(filename)
task_meta = get_task_meta(task)

num_in = st.sidebar.number_input('Pick a question', 0, len(data), key="num_in")
curr_el = data[num_in]

st.write(f"## Question")
st.write(f"{curr_el[task_meta.question_field]}")


gold_col, pred_col = st.columns([1, 1])
gold = curr_el[task_meta.answer_field]
pred = curr_el['generated']
with gold_col:
    st.write(f"## Gold")
    st.write(f"{gold} ✅")
with pred_col:
    st.write(f"## Generated")
    st.write(
        f"{pred} {'✅' if gold == pred else '❌'}"
    )

prompt = curr_el['prompt']
st.write(f"## Prompt")
st.write(f"```{prompt}")