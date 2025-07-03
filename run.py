from flask import Flask, send_from_directory
from url_folder.app import url_app
app = Flask(__name__)

# Register the Blueprints
app.register_blueprint(url_app)
# Route for landing page (home.html inside /home)
@app.route('/')
def home():
    return send_from_directory('home', 'home.html')

# Serve static files from /home (like CSS, JS)
@app.route('/home/<path:filename>')
def serve_home_assets(filename):
    return send_from_directory('home', filename)

if __name__ == '__main__':
    app.run(debug=True)



