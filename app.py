from flask import Flask, render_template, request
import numpy as np

import hdd_classification_VS as hdd


app = Flask(__name__)

@app.route("/", methods = ["GET", "POST"])
def failure(failStatus=0,failFlag='healthy'):
    if request.method == 'POST':
        smart_attr = request.form["smart_attributes"]
        smart_attr = smart_attr.split(',')
        smart_attr = [float(x) for x in smart_attr]
        # smart_attr = smart_attr.split()
        smart_attr = np.reshape(smart_attr,(1,-1))
        # smart_attr.astype('float')
        failure_pred = hdd.HDD_prediction(smart_attr)
        failStatus = failure_pred
        print(type(failStatus))
        if failStatus[0] == 0:
            failFlag = 'healthy'
        else:
            failFlag = 'unhealthy'
        

    return render_template("index.html", fail_status = failFlag)


# @app.route("/sub", methods = ['POST'])
# def submit():
#     #HTML -> .py
#     if request.method == "POST":
#         name = request.form["username"]

#     #.py -> HTML
#     return render_template("sub.html", n = name)

if __name__ == "__main__":
    app.run(debug=True)