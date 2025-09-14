
import socket
import json

# --- Configuration ---
UDP_IP = "127.0.0.1"  # Standard loopback interface address (localhost)
UDP_PORT = 12345      # Port to listen on (non-privileged port)
# -------------------

def main():
    """
    Receives EEG bandpower data streaming over UDP from the OpenBCI GUI,
    parses the JSON data, and prints it to the console.
    """
    sock = None  # Initialize sock to None
    try:
        # Create a UDP socket
        sock = socket.socket(socket.AF_INET,  # Internet
                             socket.SOCK_DGRAM)  # UDP

        # Bind the socket to the IP address and port
        sock.bind((UDP_IP, UDP_PORT))
        print(f"Listening for UDP data on {UDP_IP}:{UDP_PORT}")

        while True:
            data, addr = sock.recvfrom(1024)  # buffer size is 1024 bytes
            try:
                # Decode the received bytes to a string
                json_string = data.decode('utf-8')

                # Parse the JSON string
                parsed_json = json.loads(json_string)

                # Extract the bandpower data
                if "data" in parsed_json:
                    bandpower_data = parsed_json["data"]
                    print(f"Received bandpower data: {bandpower_data}")
                else:
                    print(f"Received data does not contain 'data' key: {parsed_json}")

            except json.JSONDecodeError:
                print(f"Error decoding JSON from: {data}")
            except Exception as e:
                print(f"An error occurred: {e}")

    except socket.error as e:
        print(f"Socket error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if sock:
            sock.close()
            print("Socket closed.")

if __name__ == "__main__":
    main()
