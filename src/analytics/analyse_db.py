from attr import define
from tinydb import TinyDB, Query
from pandas import DataFrame 
import streamlit as st
import matplotlib.pyplot as plt

def _database_selection():
    db = TinyDB("db.json")

    # Récup les caractéristiques de la db que l’on peut sélectionner
    selection = db.all()
    criterias = {}
    not_criterias = ["description", "results"]
    for select in selection:
        get_criterias(select, criterias, not_criterias)

    # Afficher les choix possibles
    choices = {}
    with st.sidebar.expander("Select benchmark criterias :"):
        select_criterias(criterias, choices)

    # Faire la sélection
    run = Query()
    run = test_criterias(choices, run)
    selection = db.search(run)
    
    # Print les caractéristiques de la selection
    # Moyenner les valeurs de la sélection
    # Moyenner les courbes de la sélection
    # Afficher les valeurs
    # Afficher les courbes

    st.title("Benchmark Report")

    st.header("Benchmark selection")
    # Headers
    columns = st.columns(len(selection))
    for i in range(len(selection)):
        with columns[i]:
            st.subheader(f"Benchmark {i+1}")
            bench = selection[i]
            for part, elements in bench.items():
                if part != "results":
                    st.markdown(f"- **{part} :**")
                    for key, value in elements.items():
                        if key != "stable":
                            st.markdown(f"*{key}* : {value}")

    st.header("Results")

    # results
    for result in selection[0]["results"]:
        if type(selection[0]["results"][result]) != dict:
            st.markdown(f"**{result} :**")
            for i in range(len(selection)):
                value = selection[i]["results"][result]
                st.markdown(f"**- benchmark {i+1}** : {value}")

    # 1st figure: number of jobs vs time
    fig = plt.figure()
    for i in range(len(selection)):
        profile = DataFrame.from_dict(selection[i]["results"]["profile"])
        plt.step(profile.timestamp, profile.n_jobs_running, where="post", label=f"benchmark {i+1}")
    plt.legend()
    plt.title("profile")
    st.pyplot(fig)
    plt.close()

    # 2nd figure: objective vs iter
    def to_max(l):
        r = [l[0]]
        for e in l[1:]:
            r.append(max(r[-1], e))
        return r

    fig = plt.figure()
    for i in range(len(selection)):
        plt.plot(to_max(DataFrame.from_dict(selection[i]["results"]["search"]).objective), label=f"benchmark {i+1}")
    plt.legend()
    plt.title("search")
    st.pyplot(fig)
    plt.close()

def get_criterias(data, criterias, not_criterias):
    for key, val in data.items():
        if key not in not_criterias:
            if type(val) == list:
                val = tuple(val)
            if key not in criterias:
                if type(val) == dict:
                    criterias[key] = {}
                else:
                    criterias[key] = {val}
            if type(val) == dict :
                get_criterias(data[key], criterias[key], not_criterias)
            else:
                criterias[key].add(val)

def select_criterias(criterias, choices):
    for key, val in criterias.items():
        if type(val) == dict :
            st.markdown("------")
            st.write(f"{key} :")
            choices[key] = {}
            select_criterias(criterias[key], choices[key])
        else:
            choice = st.multiselect(
                label=key,
                options=val,
                default=list(val)
            )
            if len(choice) != 0 and type(choice[0]) == tuple:
                choice = list(map(lambda x: list(x), choice))
            choices[key] = choice

def test_criterias(choices, query):
    sentence = None
    for key, val in choices.items():
        if type(val) == dict :
            test = (~ (query[key].exists())) | (test_criterias(choices[key], query[key]))
        else:
            test = (~ (query[key].exists())) | (query[key].one_of(val))
        sentence = (test) & (sentence)
    return sentence

_database_selection()