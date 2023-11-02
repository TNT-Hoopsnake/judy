from flask import Flask, render_template, request, abort
from gpt_eval.web_app.utils import (
    load_all_data,
    format_data,
    get_grouped_df,
    get_readable_timestamp,
)

all_data = load_all_data()
df = format_data(all_data)

app = Flask(__name__)
app.config.from_pyfile("config.py")
app.static_folder = "static"


@app.template_filter("timestamp")
def timestamp_filter(timestamp):
    return get_readable_timestamp(timestamp)


@app.route("/")
def runs_dashboard():
    run_data = all_data

    all_evaluations = sum(
        data["config"]["run"].num_evals for data in run_data.values()
    )
    combined_run = {
        "name": "All",
        "models": df["model"].nunique(),
        "tasks": df["task"].nunique(),
        "datasets": df["dataset"].nunique(),
        "total_eval_num": all_evaluations,
    }
    context = {"data": run_data, "combined": combined_run}
    return render_template("run_dashboard.html", context=context)


# not currently in use - might return it later idk?
# @app.route('/models/<model_name>', methods=['GET', 'POST'])
# def model_page(model_name):
#     groupby_options = {
#         'metric_group': 'Metric Group',
#         'run':'Run',
#         'task':'Task',
#         'dataset':'Dataset'
#     }

#     groupby = 'metric_group'
#     if request.method == 'POST':
#         groupby = request.form.get('groupby')

#     if model_name in df['model'].values:
#         filtered_df = df[df['model'] == model_name]
#     else:
#         abort(404)

#     data = get_grouped_df(filtered_df, groupby)

#     run_list = filtered_df['run'].unique()
#     tasks_list = filtered_df['task'].unique()
#     datasets_list = filtered_df['dataset'].unique()

#     context = {
#         'model':model_name,
#         'runs':run_list,
#         'datasets':datasets_list,
#         'tasks':tasks_list,
#         'table_data':data,
#         'groupby':groupby,
#         'groupby_options':groupby_options
#     }

#     return render_template('model_page.html', context=context)


@app.route("/runs/<run_name>", methods=["GET", "POST"])
def run_page(run_name):
    groupby_options = {
        "metric_group": "Metric Group",
        "model": "Model",
        "task": "Task",
        "dataset": "Dataset",
    }

    groupby = "metric_group"
    if request.method == "POST":
        groupby = request.form.get("groupby")

    if run_name in df["run"].values:
        filtered_df = df[df["run"] == run_name]
        run_data = all_data.get(run_name)
        run_config = run_data["config"]["run"]
        run_configs = {
            "Timestamp": get_readable_timestamp(run_data["metadata"]["timestamp"]),
            "Judge Model": run_config.judge.value,
            "Judge Temperature": run_config.judge_temperature,
            "Random Seed": run_config.random_seed,
            "Number of Evaluations": run_config.num_evals,
        }
    elif run_name == "all":
        filtered_df = df
        run_data = []
        run_config = []
        run_configs = {}
    else:
        abort(404)

    data = get_grouped_df(filtered_df, groupby)

    models_list = filtered_df["model"].unique()
    tasks_list = filtered_df["task"].unique()
    datasets_list = filtered_df["dataset"].unique()

    context = {
        "run": run_name,
        "models": models_list,
        "datasets": datasets_list,
        "tasks": tasks_list,
        "table_data": data,
        "groupby": groupby,
        "groupby_options": groupby_options,
        "run_configs": run_configs,
    }

    return render_template("run_page.html", context=context)


@app.route("/tasks")
def tasks():
    item_filter = request.args.get("item")
    seen_tasks = set()
    task_data = []
    for run_data in all_data.values():
        for task in run_data["tasks_used"]:
            if task.id not in seen_tasks:
                if not item_filter or (item_filter and task.id == item_filter):
                    seen_tasks.add(task.id)
                    task_data.append(
                        {
                            "title": task.name,
                            "desc": task.desc,
                            "links": {"datasets": task.datasets, "tags": task.tags},
                        }
                    )

    return render_template(
        "card_page.html",
        cards_data=task_data,
        page_title="Task Info",
        page_subtitle="List of Tasks used in your Judy Runs",
    )


@app.route("/datasets")
def datasets():
    item_filter = request.args.get("item")

    seen_datasets = set()
    dataset_data = []
    for run_data in all_data.values():
        for dataset in run_data["datasets_used"]:
            if dataset.id not in seen_datasets:
                if not item_filter or (item_filter and dataset.id == item_filter):
                    seen_datasets.add(dataset.id)
                    dataset_data.append(
                        {
                            "title": dataset.id,
                            "desc": None,
                            "data": {"Source": dataset.source},
                            "links": {"tags": dataset.tags, "tasks": dataset.tasks},
                        }
                    )

    return render_template(
        "card_page.html",
        cards_data=dataset_data,
        page_title="Dataset Info",
        page_subtitle="List of Datasets used in your Judy Runs",
    )


@app.route("/models")
def models():
    seen_models = set()
    model_data = []
    item_filter = request.args.get("item")

    for run_data in all_data.values():
        for model in run_data["models_used"]:
            if model.id not in seen_models:
                if not item_filter or (item_filter and model.id == item_filter):
                    seen_models.add(model.id)
                    model_data.append(
                        {
                            "title": model.id,
                            "desc": None,
                            "data": {
                                "API Type": model.api_type,
                                "Api Base": model.api_base,
                            },
                            "links": {},
                        }
                    )

    return render_template(
        "card_page.html",
        cards_data=model_data,
        page_title="Model Info",
        page_subtitle="List of Models used in your Judy Runs",
    )


@app.route("/raw")
@app.route("/raw/<run_name>")
def raw_results(run_name=None):
    model_filter = request.args.get("model")
    dataset_filter = request.args.get("dataset")

    run_list = df["run"].unique()
    if not run_name:
        run_name = run_list[0]

    run_data = all_data.get(run_name).get("data")
    if not model_filter:
        model_filter = df[df["run"] == run_name]["model"].iloc[0]

    if not dataset_filter:
        dataset_filter = df[df["run"] == run_name]["dataset"].iloc[0]

    filtered_data = run_data
    if model_filter in filtered_data:
        filtered_data = filtered_data[model_filter]

    if dataset_filter in filtered_data:
        filtered_data = filtered_data[dataset_filter]

    return render_template(
        "raw_results.html",
        runs=run_list,
        run_name=run_name,
        model_name=model_filter,
        dataset_name=dataset_filter,
        raw_data=run_data,
        filtered_data=filtered_data,
    )


if __name__ == "__main__":
    app.run(debug=True)
