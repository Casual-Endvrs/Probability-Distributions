import streamlit as st
from helpers.multiapp import MultiApp
from apps import introduction_app
from apps import d_Bernoulli_app
from apps import d_Binomial_app
from apps import d_Multinomial_app
from apps import d_Poisson_app
from apps import c_Uniform
from apps import c_Gaussian_app
from apps import c_Exponential_app
from apps import c_Beta_app

# Use the full width of the display for the dashboard
st.set_page_config(layout="wide")

# This is a diagnostic. Set to True for development, set to False when pushing for general users.
showWarningOnDirectExecution = False

app = MultiApp()

# app.add_app("Dashboard Introduction", introduction_app.app)
app.add_app("Bernoulli Distribution", d_Bernoulli_app.app)
app.add_app("Binomial Distribution", d_Binomial_app.app)
app.add_app("Multinomial Distribution", d_Multinomial_app.app)
app.add_app("Poisson Distribution", d_Poisson_app.app)
app.add_app("Uniform Distribution", c_Uniform.app)
app.add_app("Gaussian Distribution", c_Gaussian_app.app)
app.add_app("Exponential Distribution", c_Exponential_app.app)
app.add_app("Beta Distribution", c_Beta_app.app)

app.run()
