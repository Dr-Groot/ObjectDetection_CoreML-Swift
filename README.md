# Object Detection using CoreML
![image](https://user-images.githubusercontent.com/63160825/215057345-34a5a6a5-b3ea-451d-bbbd-7ddc859dc9fc.png)

Core ML applies a machine learning algorithm to a set of training data to create a model. You use a model to make predictions based on new input data. Models can accomplish a wide variety of tasks that would be difficult or impractical to write in code. For example, you can train a model to categorize photos, or detect specific objects within a photo directly from its pixels.
                                                                          After you create the model, integrate it in your app and deploy it on the user’s device. Your app uses Core ML APIs and user data to make predictions and to train or fine-tune the model.

![image](https://user-images.githubusercontent.com/63160825/215057785-94f6d5fa-08b1-4c9f-a8c8-8c3eb1c97420.png)

You can build and train a model with the Create ML app bundled with Xcode. Models trained using are in the Core ML model format and are ready to use in your app. Alternatively, you can use a wide variety of other machine learning libraries and then use Core ML Tools to convert the model into the Core ML format. Once a model is on a user’s device, you can use Core ML to retrain or fine-tune it on-device, with that user’s data.  
                                                                                                                            Core ML optimizes on-device performance by leveraging the CPU, GPU, and Neural Engine while minimizing its memory footprint and power consumption. Running a model strictly on the user’s device removes any need for a network connection, which helps keep the user’s data private and your app responsive.

Core ML is the foundation for domain-specific frameworks and functionality. Core ML supports Vision for analyzing images, Natural Language for processing text, Speech for converting audio to text, and Sound Analysis for identifying sounds in audio. Core ML itself builds on top of low-level primitives like Accelerate and BNNS, as well as Metal Performance Shaders.

![image](https://user-images.githubusercontent.com/63160825/215057612-e69a36f6-137f-41f4-8a01-9e033f47787a.png)

> Vision framework performs face and face landmark detection, text detection, barcode recognition, image registration, and general feature tracking. Vision also allows the use of custom Core ML models for tasks like classification or object detection.

## Project Work
Real time camera object detection with Machine Learning. Basic introduction to Core ML, Vision and ARKit.

In our project we have two important functions, which we need to understand:
+ loadCameraAndPreview
+ captureOutput

**loadCameraAndPreview**

```swift
//  SETTING UP THE CAMERA FOR RECOGNITION USING AVCaptureSession
    private func loadCameraAndPreview() {
        let captureSession = AVCaptureSession() // Creating Capture Session
        captureSession.sessionPreset = .photo // Capture Present Style
        guard let captureDevice = AVCaptureDevice.default(for: .video) else { return } // Capture Device location is given to back camera
        guard let input = try? AVCaptureDeviceInput(device: captureDevice) else { return } // Setting up the Capture device input from the device
        captureSession.addInput(input) // Adding input to Capture Session
        captureSession.startRunning() // Starting Capture Session

        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession) // Addedd the Capture Session to preview layer
        view.layer.addSublayer(previewLayer) // Added previewLayer to View for displaying on the screen + Frame
        previewLayer.frame = view.frame

//      Capturing the data from the video frame and adding delegate.
        let dataOutput = AVCaptureVideoDataOutput()
        dataOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        captureSession.addOutput(dataOutput)
    }
```

**captureOutput**

```swift
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {

        guard let pixelBuffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        guard let model = try? VNCoreMLModel(for: SqueezeNet().model) else { return }

        let request = VNCoreMLRequest(model: model) { finishRequest, error in
            guard let results = finishRequest.results as? [VNClassificationObservation] else { return }
            guard let observation = results.first else { return }
            DispatchQueue.main.async {
                self.identifierLabel.text = "\(observation.identifier) \(observation.confidence * 100)"
            }
        }

        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform([request])
    }
}
```

captureOutput is a delegate method which is called everytime when camera is capturing a frame and in this function we will be setting up our model and request handler for object detection.

So first we will talk about **pixelBuffer**, pixelBuffer will be storing the image which it will be getting from *CMSampleBufferGetImageBuffer*. *CMSampleBufferGetImageBuffer* returns an image buffer that contains the media data and here we are getting media data from **sampleBuffer** which is a type of *CMSampleBuffer* (An object that contains zero or more media samples of a uniform media type.). **sampleBuffer** is a capture output from *setSampleBufferDelegate(_:queue:)* (Sets the delegate that will accept captured buffers and the dispatch queue on which the delegate will be called and this function is called in **loadCameraAndPreview** function).
                                                              
Then we have our **model** of type *VNCoreMLModel* which is a container for the model to use with the Vison Request. We have used **Resnet50** which is a Image Classification model and i have also added **SqueezeNet** which is also a Image Classification model but lighter one. We will talk about later in this project why i have added two models.😁
                                                              
Now most important part comes that is our **request** which is a *VNCoreMLRequest*. *VNCoreMLRequest* in simple word is a Vision Request which is a image analysis request that uses a CoreML model to process images. We have a completion handler that is providing us with an **finishRequest** and **error**. We take **results** from the finishRequest as a VNClassificationObservation which is an object that represents classification information that an image analysis request produces and at last we have our **observation**, which is an identifier which is known as classification label identifing the type of observation, which we are getting from the **results**.

This request is handle by *VNImageRequestHandler*, an object that processes one or more image analysis requests pertaining to a single image. Here we are requesting a **pixelBuffer** which is a type of *cvPixelBuffer* and asking to perform **request** which is a type of Vison Request.
