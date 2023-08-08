from gevent import monkey

monkey.patch_all()

from pathlib import Path

from gevent.pywsgi import WSGIServer

from app import app
from config import *
from httpServer import run_http_server
from utils import clear_media_folder, init_logger

Path(MEDIA_PATH).mkdir(parents=True, exist_ok=True)
Path(LOG_FILE_PATH).parent.mkdir(parents=True, exist_ok=True)

logger = init_logger()

if __name__ == "app":
    logger.info("========================= Program Started =========================")
    run_http_server(MEDIA_PATH, MEDIA_SERVER_PORT)

if __name__ == "__main__":
    logger.info("========================= Program Started =========================")
    try:
        run_http_server(MEDIA_PATH, MEDIA_SERVER_PORT)
        logger.info(f"Starting WSGI Server on port {WEB_SERVER_PORT}...")
        web_server = WSGIServer(('0.0.0.0', WEB_SERVER_PORT), app)
        web_server.serve_forever()
        # app.run(host='0.0.0.0', port=WEB_SERVER_PORT)
    except KeyboardInterrupt:
        logger.info("Stopping WSGI Server...")
        # clear_media_folder()
        web_server.stop()
        logger.info("WSGI Server stopped.")
        logger.info("========================= Program Stopped =========================")
