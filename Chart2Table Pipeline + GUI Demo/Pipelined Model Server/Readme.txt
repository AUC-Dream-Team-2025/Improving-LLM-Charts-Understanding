you need to run two servers, the paddle paddle server for the chart2text and the demo app server.
Let's start with the paddle paddle server. Open a new cmd terminal and run the following commands:

cd Path/To/PaddlePaddleServer
python server.py


This should start that server, next open a new cmd terminal to run the demo app server on:

cd Path/To/Model Server
python server.py

And now you should have the model running as a server. To call it from the gui use the ip address of the device running the server on port 8001/predict so it should look like "https://XX.XX.XX.XX:8001/predict"
