import './App.css';
import * as faceapi from 'face-api.js';
import React from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import Container from 'react-bootstrap/Container';
import Button from 'react-bootstrap/Button';

function App() {
  
  const [modelsLoaded, setModelsLoaded] = React.useState(false);
  const [captureVideo, setCaptureVideo] = React.useState(false);
  const [faceDetected, setFaceDetected] = React.useState(false);
  
  const [capturedImage, setCapturedImage] = React.useState(false);
  const [greeting, setGreeting] = React.useState();
  const [prediction, setPrediction] = React.useState();
  const [newImgPathBase64, setNewImgPathBase64] = React.useState('')

  const videoRef = React.useRef();
  const videoHeight = 480;
  const videoWidth = 640;
  const canvasRef = React.useRef();
  const count = 0;
  const urlBase = "http://127.0.0.1:8000"

  React.useEffect(() => {

    const loadModels = async () => {
      const MODEL_URL = '/models';
      const api = async () => {
        fetch(urlBase + "/api")
        .then(response => response.json())
        .then(data=> {
            setGreeting(data);
        });
      };
      Promise.all([
        faceapi.nets.tinyFaceDetector.load(MODEL_URL),
        faceapi.nets.faceLandmark68Net.load(MODEL_URL),
        faceapi.nets.faceRecognitionNet.load(MODEL_URL),
        //faceapi.nets.faceExpressionNet.load(MODEL_URL),
        api(),
      ]).then(setModelsLoaded(true));
    }
    loadModels();
  }, [count]);

  const startVideo = () => {
    setCaptureVideo(true);
    navigator.mediaDevices
      .getUserMedia({ video: { width: 300 } })
      .then(stream => {
        let video = videoRef.current;
        video.srcObject = stream;
        video.play();
      })
      .catch(err => {
        console.error("error:", err);
      });
  }

  const handleVideoOnPlay = () => {
    setInterval(async () => {
      if (canvasRef && canvasRef.current) {
        canvasRef.current.innerHTML = faceapi.createCanvasFromMedia(videoRef.current);
        const displaySize = {
          width: videoWidth,
          height: videoHeight
        }

        faceapi.matchDimensions(canvasRef.current, displaySize);

        const detections = await faceapi.detectAllFaces(videoRef.current, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks();
        
        detections.length > 0 ? setFaceDetected(true) : setFaceDetected(false)
        
        const resizedDetections = faceapi.resizeResults(detections, displaySize);

        canvasRef && canvasRef.current && canvasRef.current.getContext('2d').clearRect(0, 0, videoWidth, videoHeight);
        canvasRef && canvasRef.current && faceapi.draw.drawDetections(canvasRef.current, resizedDetections);
        canvasRef && canvasRef.current && faceapi.draw.drawFaceLandmarks(canvasRef.current, resizedDetections);
        
        //canvasRef && canvasRef.current && faceapi.draw.drawFaceExpressions(canvasRef.current, resizedDetections);
      }
    }, 100)
  }

  const screenShot = async () => {

    const detection = await faceapi.detectSingleFace(videoRef.current, new faceapi.TinyFaceDetectorOptions())
    if (detection) {
      setFaceDetected(true)
      const track = videoRef.current.srcObject.getVideoTracks()[0]
      const imageCapture = new ImageCapture(track)
      
      //Create a new image
      var img = new Image()
      //Crea a blob
      const blob = await imageCapture.takePhoto();
      var objectURL = URL.createObjectURL(blob);
      // Assign blob to image
      img.src = objectURL;
      
      
      //const displaySize = {
      //  width: videoRef.current.width,
      //  height: videoRef.current.height
      //};
      
      //const resizeDetection = faceapi.resizeResults(detection, displaySize);
      // faceapi.draw.drawFaceLandmarks(canvasRef.current, resizeDetections);
      //canvasRef.current
      //  .getContext("2d")
      //  .clearRect(0, 0, displaySize.width, displaySize.height);
      //faceapi.draw.drawDetections(canvasRef.current, resizeDetection);
  
      console.log(
        `Width ${detection.box._width} and Height ${detection.box._height}`
      );
      
      extractFaceFromBox(img, detection.box);
    

      //extractFace(videoRef, x, y, width, height);
    } else {
      setFaceDetected(false)
    }

  }



  async function extractFaceFromBox(imageRef, box) {
    const regionsToExtract = [
      new faceapi.Rect(box.x, box.y, box.width, box.height)
    ];
    let faceImages = await faceapi.extractFaces(imageRef, regionsToExtract);
    
    if (faceImages.length === 0) {
      console.log("No face found");
    } else {
      setCapturedImage(true)
      const outputImage = new Image();
      console.log("Face canvas : ",faceImages[0],"imageref :", faceImages)
      outputImage.src = faceImages[0].toDataURL();
      //faceImages.forEach((cnv) => {
      //  console.log("Face : ", cnv)
      //  outputImage.src = cnv.toDataURL();
      //  setCropped(cnv.toDataURL());
      //});
      setNewImgPathBase64(outputImage.src);

      // Create a blob from cropped image and send to
      let blob = await fetch(outputImage.src).then(r => r.blob());


      const preds = await handleSubmission(blob)
      if (preds) setPrediction(preds)

    }
  }

  const handleSubmission = async (file) => {
    const formData = new FormData()
    formData.append('file', file);
    const postData = {
      method: 'POST',
      body: formData,
      headers: {
        Accept: 'application/json',
      },
    }
    fetch(urlBase + "/api/image", postData)
    .then(response => response.json())
    .then(object => {const json = {"gender":Object.keys(object)[0], "p":Object.values(object)[0]}
                    setPrediction(json)})
    .catch((error) => { console.error(error); });
  };

  const closeWebcam = () => {
    videoRef.current.pause();
    videoRef.current.srcObject.getTracks()[0].stop();
    setCaptureVideo(false);
  }

  return (
    <Container>
      <Container style={{ textAlign: 'center', padding: '10px' }}>
        <Container>
          {greeting ? <p>API in online </p>: <p>API in offline </p> }
        </Container>
        {
          captureVideo && modelsLoaded ?
            <Button onClick={closeWebcam} style={{ cursor: 'pointer', backgroundColor: 'green', color: 'white', padding: '15px', fontSize: '25px', border: 'none', borderRadius: '10px' }}>
              Close Webcam
            </Button>
            :
            <Container>
            {greeting ?
            <Button onClick={startVideo} style={{ cursor: 'pointer', backgroundColor: 'green', color: 'white', padding: '15px', fontSize: '25px', border: 'none', borderRadius: '10px' }}>
              Open Webcam
            </Button>: <></>
            }
            </Container>
        }
        
      </Container>
      {
        captureVideo ?
          modelsLoaded ?
            <Container>
              <Container style={{ display: 'flex', justifyContent: 'center', padding: '10px' }}>
                <video ref={videoRef} height={videoHeight} width={videoWidth} onPlay={handleVideoOnPlay} 
                        style={{ borderRadius: '10px' }} />
                <canvas ref={canvasRef} style={{ position: 'absolute' }} />
              </Container>
              
            </Container>        
            :
            <Container>Loading...</Container>
          :
          <>
          </>
      }
      {captureVideo ?
        <Container style={{ display: 'flex', justifyContent: 'center', padding: '10px' }}>
          <Button onClick={screenShot} disabled={!faceDetected} style={{ cursor: 'pointer', backgroundColor: 'blue', color: 'white', padding: '15px', fontSize: '25px', border: 'none', borderRadius: '10px' }}>
            Detect gender
          </Button>
        </Container>
      : <></>
      }

      <Container> 
        { capturedImage ? 
            <Container style={{ display: 'flex', justifyContent: 'center', padding: '10px' }}>
              <img src={newImgPathBase64} width="200" height="230" alt=''/>
              {prediction ? 
                <Container>Predicted as {prediction.gender} with a probability of {prediction.p} %  </Container> : <>Fetching predictions...</>
              }
            </Container> : 
            <></>
        }
      </Container>

    </Container>
  );
}

export default App;