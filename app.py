from flask import Flask

app = Flask(__name__)


@app.route("/api/human-to-statue")
def hello():
    # take image input in base64?

    # pre-process / scaler to 256 x 256

    # use gen-model to cook

    # return cooked image as downloadable?

    return "Hello World!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

