This should start a server that runs only the base qwen model so that you can compare its results with that of the pipelined model. Open a new cmd terminal to run the demo app server on:

cd Path/To/Base Model Server
python server.py

And now you should have the model running as a server. To call it from the gui use the ip address of the device running the server on port 8001/predict so it should look like "https://XX.XX.XX.XX:8001/predict"
