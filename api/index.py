from http.server import BaseHTTPRequestHandler
import json


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            html = b"""<!DOCTYPE html>
<html><head><title>YOLOv8 Object Detection</title></head>
<body><h1>YOLOv8 Object Detection</h1><p>Test page - function is working!</p></body></html>"""
            self.wfile.write(html)

    def do_POST(self):
        if self.path == "/api/detect":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"message": "Detection endpoint working - ML deps not yet installed"}).encode())
        else:
            self.send_response(404)
            self.end_headers()
