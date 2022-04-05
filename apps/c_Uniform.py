from apps.dist_dashboard_base import dashboard_template
from distributions.c_Uniform import Uniform_distribution


def app():
    dist_cls = Uniform_distribution
    dist_name = "Uniform"
    text_file = "c_Uniform.txt"

    dashboard_template(dist_cls, dist_name, text_file)
