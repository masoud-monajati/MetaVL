import socketserver

with socketserver.TCPServer(("localhost", 0), None) as s:
    free_port = s.server_address[1]
    print(free_port)