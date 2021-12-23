# AgeGuesser

![Testing](https://github.com/se4ai2122-cs-uniba/AgeGuesser/actions/workflows/testing.yaml/badge.svg)
![Deployment](https://github.com/se4ai2122-cs-uniba/AgeGuesser/actions/workflows/build_and_deploy.yaml/badge.svg)

AgeGuesser is an end-to-end, deep-learning based, Age Estimation system. 

It is made of a face detection (Yolov5) and age regression model (EfficientNet).

The service is available at [ageguesser.com](https://ageguesser.com)

## Deployment 

The system is currently made of 2 nodes in the eu-west-2 (London) aws region. Requests (https) are managed and served by the Traefik web server.

<a href="https://ibb.co/yYPvWZF"><img src="https://i.ibb.co/MGSwMtk/aws-high-level-Page-2-4.jpg" alt="aws-high-level-Page-2-4"></a>
