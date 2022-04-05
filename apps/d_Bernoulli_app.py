from apps.dist_dashboard_base import dashboard_template
from distributions.d_Bernoulli import Bernoulli_distribution


def app():
    dist_cls = Bernoulli_distribution
    dist_name = "Bernoulli"
    text_file = "d_Bernoulli.txt"

    dashboard_template(dist_cls, dist_name, text_file)
