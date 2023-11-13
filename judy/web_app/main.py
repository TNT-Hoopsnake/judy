from flask import Flask, g
from judy.web_app.utils import load_all_data, get_readable_timestamp
from judy.web_app.views import app_bp

app = Flask(__name__)
app.config.from_pyfile("config.py")
app.static_folder = "static"


@app.before_request
def before_request():
    g.all_data = app.config.get("all_data")
    g.used_data = app.config.get("used_data")


@app.template_filter("timestamp")
def timestamp_filter(timestamp):
    return get_readable_timestamp(timestamp)


def run_webapp(host, port, data_directory):
    with app.app_context():
        app.register_blueprint(app_bp)
        all_data, used_data = load_all_data(data_directory)
        app.config["all_data"] = all_data
        app.config["used_data"] = used_data
        app.run(host, port, debug=True)


if __name__ == "__main__":
    run_webapp("localhost", 5000, "./results")
