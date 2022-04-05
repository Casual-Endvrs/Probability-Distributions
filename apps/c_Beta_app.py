from apps.dist_dashboard_base import dashboard_template
from distributions.c_Beta import Beta_distribution


def app():
    dist_cls = Beta_distribution
    dist_name = "Beta"
    text_file = "c_Beta.txt"

    dashboard_template(dist_cls, dist_name, text_file)
