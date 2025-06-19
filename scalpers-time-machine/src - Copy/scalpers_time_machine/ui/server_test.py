from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT = 8000
print(f"Running on http://localhost:{PORT}")
server = HTTPServer(('localhost', PORT), SimpleHTTPRequestHandler)
server.serve_forever()