from apps.dist_dashboard_base import dashboard_template
from distributions.d_Multinomial import Multinomial_distribution


def app():
    dist_cls = Multinomial_distribution
    dist_name = "Multinomial"
    text_file = "d_Multinomial.txt"

    dashboard_template(dist_cls, dist_name, text_file)
