import './App.css';
import * as faceapi from 'face-api.js';
import React from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import Container from 'react-bootstrap/Container';
import Button from 'react-bootstrap/Button';
import Webcam from 'react-webcam';
import ReactDOM from 'react-dom';

const WebcamCapture = () => {
  const webcamRef = React.useRef(null);
  const [imgSrc, setImgSrc] = React.useState(null);
  const [modelsLoaded, setModelsLoaded] = React.useState(false);
  const [webCamEnabled, setWebCamEnabled] = React.useState(false);
  const [faceDetected, setFaceDetected] = React.useState(false);
  
  const [capturedImage, setCapturedImage] = React.useState(false);
  const [greeting, setGreeting] = React.useState();
  const [prediction, setPrediction] = React.useState();
  const [newImgPathBase64, setNewImgPathBase64] = React.useState('')

  const videoRef = React.useRef();
  const canvasRef = React.useRef();

  React.useEffect(() => {

    const loadModels = async () => {
      const MODEL_URL = '/models';
      await fetch("https://app1.ouicodedata.com:8000/api")
        .then(response => response.json())
        .then(data=> setGreeting(data));
      Promise.all([
        faceapi.nets.tinyFaceDetector.load(MODEL_URL),
        faceapi.nets.faceLandmark68Net.load(MODEL_URL),
        faceapi.nets.faceRecognitionNet.load(MODEL_URL),
        //faceapi.nets.faceExpressionNet.load(MODEL_URL),
      ]).then(setModelsLoaded(true));
      }
      loadModels();
  }, []);

  const startCamera = async () =>{
    setWebCamEnabled(true);
    
    setInterval(async ()=> { 

      if (canvasRef && canvasRef.current) {
        canvasRef.current.innerHTML = faceapi.createCanvasFromMedia(videoRef.current);
         const displaySize = { width: webcamRef.width, height: webcamRef.height}
      }
      if (!!webcamRef) {

          const canvas = faceapi.createCanvasFromMedia(webcamRef.current);
          // resize the overlay canvas to the input dimensions

          const displaySize = { width: webcamRef.width, height: webcamRef.height }
          faceapi.matchDimensions(canvas, displaySize)

          /* Display detected face bounding boxes */
          const detection = await faceapi.detectSingleFace(videoRef.current, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks()

          // resize the detected boxes in case your displayed image has a different size than the original
          const resizedDetections = faceapi.resizeResults(detection, displaySize)
          // draw detections into the canvas
          faceapi.draw.drawDetections(canvas, resizedDetections)
      } else {
        setFaceDetected(false)
      }
    
    },2000)
  }

  const stopCamera = () =>{
    setWebCamEnabled(false);
  }

  
  const capture = React.useCallback( async () => {
    const imageSrc = webcamRef.current.getScreenshot();
    setImgSrc(imageSrc);

  

  }, [webcamRef, setImgSrc]);

  return (
    <Container fluid>
      <Container style={{ textAlign: 'center', padding: '10px' }}>
          {greeting ?
            <Container> 
              <p>API in online </p>
              <Button onClick={startCamera} style={{ cursor: 'pointer', backgroundColor: 'green', color: 'white', padding: '15px', fontSize: '25px', border: 'none', borderRadius: '10px' }}>
                  Open Webcam
              </Button>
            </Container>
            : 
            <Container>API in offline </Container>
          } 
        {
          webCamEnabled && modelsLoaded ?
          <Container fluid>
            <Webcam 
              audio={false}
              ref={webcamRef}
              id="videoCam"
              screenshotFormat="image/jpeg"
              style={{ padding: '10px',justifyContent: 'center' }}
            />
            <Container>
              <Button onClick={capture} 
                      //disabled={!faceDetected} 
                      style={{ cursor: 'pointer', backgroundColor: 'blue', color: 'white', padding: '15px', fontSize: '25px', border: 'none', borderRadius: '10px'}}>
                      Dectect Gender
              </Button>
              {imgSrc && (  <img src={imgSrc} id="imageCam" alt=""/> )}
              </Container>
          </Container>
          : 
          <></> 
        }
      </Container>
    </Container>
  );
};

ReactDOM.render(<WebcamCapture />, document.getElementById("root"));

export default WebcamCapture;
