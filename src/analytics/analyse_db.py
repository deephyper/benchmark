from tinydb import TinyDB, Query
from pandas import DataFrame 
import matplotlib.pyplot as plt

db = TinyDB('db.json', sort_keys=True, indent=4, separators=(',', ': '))
run = Query()

def plot_report(report):

    # 1st figure: number of jobs vs time
    plt.figure()
    for i in range(len(report)):
        profile = DataFrame.from_dict(report[i]["results"]["profile"])
        t0 = profile.iloc[0].timestamp
        profile.timestamp -= t0
        plt.step(profile.timestamp, profile.n_jobs_running, where="post")
    plt.savefig("profile.png")
    plt.close()

    # 2nd figure: objective vs iter
    def to_max(l):
        r = [l[0]]
        for e in l[1:]:
            r.append(max(r[-1], e))
        return r

    plt.figure()
    for i in range(len(report)):
        plt.plot(to_max(DataFrame.from_dict(report[i]["results"]["search"]).objective))
    plt.savefig("search.png")
    plt.close()

# visualize the last run

# compare the test runs
report = db.all()
plot_report(report)