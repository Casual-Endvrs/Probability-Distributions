from apps.dist_dashboard_base import dashboard_template
from distributions.c_Gaussian import Gaussian_distribution


def app():
    dist_cls = Gaussian_distribution
    dist_name = "Gaussian"
    text_file = "c_Gaussian.txt"

    dashboard_template(dist_cls, dist_name, text_file)
