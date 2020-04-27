# Hand Movement Recognizer

In this project, we'll make a glove that can recognize some basic hand movements, using a ​MicroBit, and a few sensors. We'll be using the Bluetooth capabilities on the MicroBit, in conjunction with an Android App and an Web Server to train a machine learning model to identify hand movements.

## Getting Started

A majority of the effort involved in this project is on the software side, and all the code needed to run this project is available in this repository. The code base involves 3 components, the code to generate a HEX file for the MicroBit, the Android App codebase​ which is heavily based on the MicroBit Foundation's MicroBit Blue app, with modifications made for this specific use case, and a web server with code for training a Tensorflow based model to identify hand movements.

Instructions for building the hardware component is available on instructable.