from apps.dist_dashboard_base import dashboard_template
from distributions.c_Exponential import Exponential_distribution


def app():
    dist_cls = Exponential_distribution
    dist_name = "Exponential"
    text_file = "c_Exponential.txt"

    dashboard_template(dist_cls, dist_name, text_file)
