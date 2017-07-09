from flask import Flask
app = Flask(__name__)
from flask import request

@app.route("/test", methods=['GET', 'POST'] )
def test():
    print request.args
    inp = request.args.get('text')
    portmanteau_inputs = inp.split(',')
    #shakepeare_inputs = inp.strip()
    print portmanteau_inputs
    # TO DO: insert logic for portmanteau prediction / shakeeare style prediction
    return inp
