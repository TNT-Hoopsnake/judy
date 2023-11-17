from flask import render_template, request, abort, g, Blueprint, jsonify
from judy.web_app.utils import (
    get_heatmap_data,
    get_readable_timestamp,
)

app_bp = Blueprint("judy", __name__)


@app_bp.route("/api/heatmap_data/<run_name>/<groupby>", methods=["GET"])
def fetch_heatmap_data(run_name, groupby):
    df = g.runs_df

    if run_name in df["run"].values:
        filtered_df = df[df["run"] == run_name]
    elif run_name == "all":
        filtered_df = df
    else:
        abort(404)

    data = get_heatmap_data(filtered_df, groupby)

    return jsonify(data)


@app_bp.route("/")
def runs_dashboard():
    all_data = g.all_data
    df = g.runs_df

    combined_run = {
        "name": "All",
        "models": df["model"].nunique(),
        "tasks": df["task"].nunique(),
        "datasets": df["dataset"].nunique(),
        "scenarios": df["scenario"].nunique(),
        "total_eval_num": all_data["total_evaluations"],
    }
    context = {
        "data": all_data["runs"],
        "combined": combined_run,
        "dataframe": df.to_json(orient="records"),
        "scenarios": all_data["data_index"]["scenarios"],
    }
    return render_template("run_dashboard.html", context=context)


@app_bp.route("/about")
def about():
    return render_template("about_page.html", context={})


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
def get_all_items_dict(run_data, data_index):
    item_keys = ["models", "scenarios", "tasks", "datasets"]
    items_dict = {}

    for key in item_keys:
        items_used = f"{key}_used"
        for item_id in run_data[items_used]:
            item_dict = items_dict.get(key, {})
            item_dict[item_id] = data_index[key].get(item_id)
            items_dict[key] = item_dict

    return items_dict


@app_bp.route("/runs/<run_name>", methods=["GET", "POST"])
def run_page(run_name):
    all_data = g.all_data
    df = g.runs_df

    groupby_options = {
        "scenario": "Scenario",
        "task": "Task",
        "dataset": "Dataset",
    }

    groupby = "scenario"
    if request.method == "POST":
        groupby = request.form.get("groupby")

    if run_name in df["run"].values:
        filtered_df = df[df["run"] == run_name]
        run_data = all_data["runs"].get(run_name)

        run_config = run_data["config"]["run"]
        run_configs = {
            "Timestamp": get_readable_timestamp(run_data["metadata"]["timestamp"]),
            "Judge Model": run_config.judge.value,
            "Judge Temperature": run_config.judge_temperature,
            "Random Seed": run_config.random_seed,
            "Total Evaluations": run_data["total_evaluations"],
        }
    elif run_name == "all":
        run_data = all_data
        filtered_df = df
        run_configs = {}
    else:
        abort(404)

    items_dict = get_all_items_dict(run_data, all_data["data_index"])
    data = get_heatmap_data(filtered_df, groupby)
    # we dont actually use the data here (loaded via api)
    # but need to use the keys from the data so we can build out the necessary structure for our heatmaps
    context = {
        "run": run_name,
        "table_keys": data.keys(),
        "groupby": groupby,
        "groupby_options": groupby_options,
        "run_configs": run_configs,
        **items_dict,
    }

    return render_template("run_page.html", context=context)


@app_bp.route("/scenarios")
def scenarios():
    all_data = g.all_data

    item_filter = request.args.get("item")
    seen_scenarios = set()
    scenario_data = []
    for scenario in all_data["data_index"]["scenarios"].values():
        if scenario.id not in seen_scenarios:
            if not item_filter or (item_filter and scenario.id == item_filter):
                seen_scenarios.add(scenario.id)
                scenario_data.append(
                    {
                        "title": scenario.name,
                        "desc": scenario.desc,
                        "links": {"datasets": scenario.datasets},
                    }
                )

    return render_template(
        "card_page.html",
        cards_data=scenario_data,
        page_title="Scenario Info",
        page_subtitle="List of Scenarios used in your Judy Runs",
        page_name="judy.scenarios",
        is_filtered=bool(item_filter),
    )


@app_bp.route("/tasks")
def tasks():
    all_data = g.all_data

    item_filter = request.args.get("item")
    seen_tasks = set()
    task_data = []
    for task in all_data["data_index"]["tasks"].values():
        if task.id not in seen_tasks:
            if not item_filter or (item_filter and task.id == item_filter):
                seen_tasks.add(task.id)
                task_data.append(
                    {
                        "title": task.name,
                        "desc": task.desc,
                        "links": {"tags": task.tags},
                    }
                )

    return render_template(
        "card_page.html",
        cards_data=task_data,
        page_title="Task Info",
        page_subtitle="List of Tasks used in your Judy Runs",
        page_name="judy.tasks",
        is_filtered=bool(item_filter),
    )


@app_bp.route("/datasets")
def datasets():
    all_data = g.all_data

    item_filter = request.args.get("item")

    seen_datasets = set()
    dataset_data = []
    for dataset in all_data["data_index"]["datasets"].values():
        if dataset.id not in seen_datasets:
            if not item_filter or (item_filter and dataset.id == item_filter):
                seen_datasets.add(dataset.id)
                dataset_data.append(
                    {
                        "title": dataset.id,
                        "desc": None,
                        "data": {
                            "Source": dataset.source,
                            "Tags": ", ".join(dataset.tags),
                        },
                        "links": {"tasks": [t.id.value for t in dataset.tasks]},
                    }
                )

    return render_template(
        "card_page.html",
        cards_data=dataset_data,
        page_title="Dataset Info",
        page_subtitle="List of Datasets used in your Judy Runs",
        page_name="judy.datasets",
        is_filtered=bool(item_filter),
    )


@app_bp.route("/models")
def models():
    all_data = g.all_data

    seen_models = set()
    model_data = []
    item_filter = request.args.get("item")
    for model in all_data["data_index"]["models"].values():
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
        page_name="judy.models",
        is_filtered=bool(item_filter),
    )


@app_bp.route("/raw")
@app_bp.route("/raw/<run_name>")
def raw_results(run_name=None):
    all_data = g.all_data
    df = g.runs_df

    model_filter = request.args.get("model")
    dataset_filter = request.args.get("dataset")
    task_filter = request.args.get("task")
    scenario_filter = request.args.get("scenario")

    run_list = df["run"].unique()
    if not run_name:
        run_name = run_list[0]

    run_data = all_data["runs"].get(run_name)
    run_df = df[df["run"] == run_name]

    if not model_filter:
        model_filter = run_df["model"].iloc[0]

    if not scenario_filter:
        scenario_filter = run_df["scenario"].iloc[0]

    if not dataset_filter:
        dataset_filter = run_df[run_df["scenario"] == scenario_filter]["dataset"].iloc[
            0
        ]

    if not task_filter:
        task_filter = run_df[run_df["dataset"] == dataset_filter]["task"].iloc[0]

    results_data = run_data.get("results")
    model_data = results_data.get(model_filter, {})
    if not model_data:
        abort(404)

    scenarios_used = model_data["data"].keys()
    scenario_data = model_data["data"].get(scenario_filter, {})
    datasets_used = scenario_data.keys()

    dataset_data = scenario_data.get(dataset_filter, {})
    dataset = run_data["data_index"]["datasets"].get(dataset_filter, None)
    if not dataset:
        abort(404)

    tasks_used = [ds_task.id.value for ds_task in dataset.tasks]
    filtered_data = dataset_data.get(task_filter, {})
    if not filtered_data:
        abort(404)

    return render_template(
        "raw_results.html",
        runs=run_list,
        run_name=run_name,
        model_name=model_filter,
        dataset_name=dataset_filter,
        scenario_name=scenario_filter,
        task_name=task_filter,
        raw_data=run_data,
        tasks_used=tasks_used,
        datasets_used=datasets_used,
        scenarios_used=scenarios_used,
        filtered_data=filtered_data,
    )
