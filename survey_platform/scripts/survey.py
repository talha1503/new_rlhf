import streamlit as st
import os
import pandas as pd
import random
from csv import DictWriter
import tempfile
from pathlib import Path
from source.generate_plots import generate_plots
from source.hashing import make_hashes, check_hashes
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from source.pymongo_functions import (test_connection,
                                      get_collection,
                                      add_doc,
                                      match_doc
                                      )

st.set_page_config(layout="wide")

# Paths
trajectory_dir = "../data/trajectories/"
survey_result_path = "../data/survey_results.csv"

########################################
# MongoDB
########################################

db_username = st.secrets["db_username"]
db_pswd = st.secrets["db_pswd"]
cluster_name = st.secrets["cluster_name"]

uri = f"mongodb+srv://{db_username}:{db_pswd}@{cluster_name}.gqraedl.mongodb.net/test"
client = MongoClient(uri, server_api=ServerApi('1'))
login_collection = get_collection(client, "user_login")
preferences_collection = get_collection(client, "preferences")

########################################
# Frontend rendering
########################################

trajectories = os.listdir(trajectory_dir)
trajectory_A, trajectory_B = random.sample(trajectories, 2)

_, col0, _ = st.columns([3, 10, 3])
_, col1, _, col2, _ = st.columns([4, 6, 1, 6, 2])
_, col3, _ = st.columns([3, 5, 3])


if 'LOGGED_IN' not in st.session_state:
    st.session_state.LOGGED_IN =  False

def main():
    menu = ["Home","Login","SignUp"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        with st.container():
            with col0:
                st.title("Welcome!")
                st.subheader("Background information")
                st.markdown("- This survey aims to learn about your notions of fairness. In particular, we are interested in your evaluation of whether a series of decisions made over time by an agent may be deemed fair.")
                st.markdown("- - In this simulated scenario, you will be presented with information relating to the decisions of an artificial agent tasked with either accepting or rejecting loan applications. You will see how these statistics differ between two groups, and have to judge whether the agent has acted fairly.")
                st.markdown("- If you are ready, please click **Begin** at the top of the page to start the survey.")

    elif choice == "Login":
        if not st.session_state.LOGGED_IN:
            username = st.sidebar.text_input("User Name")
            password = st.sidebar.text_input("Password",type='password')
            if st.sidebar.button("Login"):
                if not st.session_state.LOGGED_IN:
                    hashed_pswd = make_hashes(password)
                    result = match_doc(login_collection, username, check_hashes(password,hashed_pswd))
                    if result: 
                        st.success("Logged In as {}".format(username))
                        st.session_state.LOGGED_IN = True
                    else:
                        st.warning("Incorrect Username/Password")

        if st.session_state.LOGGED_IN:
            with tempfile.TemporaryDirectory() as tmpdir:
                generate_plots(
                    filepath=os.path.join(trajectory_dir, trajectory_A),
                    saved_plot_dir=os.path.join(Path(tmpdir), "A"))
                generate_plots(
                    filepath=os.path.join(trajectory_dir, trajectory_B),
                    saved_plot_dir=os.path.join(Path(tmpdir), "B"))

                with st.container():
                    with col1:
                        st.header("Trajectory A")
                        st.image(f"{os.path.join(Path(tmpdir), 'A')}/acceptance rate.png", width=400)
                        st.image(f"{os.path.join(Path(tmpdir), 'A')}/average credit score.png", width=400)
                        st.image(f"{os.path.join(Path(tmpdir), 'A')}/default rate.png", width=400)
                    with col2:
                        st.header("Trajectory B")
                        st.image(f"{os.path.join(Path(tmpdir), 'B')}/acceptance rate.png", width=400)
                        st.image(f"{os.path.join(Path(tmpdir), 'B')}/average credit score.png", width=400)
                        st.image(f"{os.path.join(Path(tmpdir), 'B')}/default rate.png", width=400)

                with st.container():
                    with col3:
                        st.subheader("")
                        st.subheader("")
                        st.subheader("Please review the graphs and share your preference:")
                        user_id = 1 # Get user id through ip address, to be changed
                        form = st.form(key='my_form')
                        option = form.radio("", ["Trajectory A", "Trajectory B"])
                        strength = form.slider("0 - Weak preference, 5 - Strong preference", 0, 5, value=3)
                        justification = form.text_area("Optionally, please provide your justification", height=10)
                        if form.form_submit_button(label="Submit"):
                            save_data = {
                                "user_id": user_id,
                                "option_a": trajectory_A,
                                "option_b": trajectory_B,
                                "decision": option,
                                "strength": strength,
                                "justification": justification,
                            }
                            preferences_collection.insert_one(save_data)
                            st.success("Decision has been recorded.")

                        
    elif choice == "SignUp" and not st.session_state.LOGGED_IN:
        with col0:
            st.subheader("Create New Account")
            new_user = st.text_input("Username")
            new_password = st.text_input("Password",type='password')
            if st.button("Signup"):
                add_doc(login_collection, new_user, make_hashes(new_password))
                st.success("You have successfully created a valid Account")
                st.info("Go to Login Menu to login")


if __name__ == '__main__':
	main()
