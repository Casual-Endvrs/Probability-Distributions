from apps.dist_dashboard_base import dashboard_template
from distributions.d_Binomial import Binomial_distribution


def app():
    dist_cls = Binomial_distribution
    dist_name = "Binomial"
    text_file = "d_Binomial.txt"

    dashboard_template(dist_cls, dist_name, text_file)
