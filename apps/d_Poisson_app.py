from apps.dist_dashboard_base import dashboard_template
from distributions.d_Poisson import Poisson_distribution


def app():
    dist_cls = Poisson_distribution
    dist_name = "Poisson"
    text_file = "d_Poisson.txt"

    dashboard_template(dist_cls, dist_name, text_file)
