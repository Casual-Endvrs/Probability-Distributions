import streamlit as st
from helpers.multiapp import MultiApp
from apps import Gaussian_app

app = MultiApp()

app.add_app("Gaussian Distribution", Gaussian_app.app)

app.run()
