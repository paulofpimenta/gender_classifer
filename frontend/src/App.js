import './App.css';
import * as faceapi from 'face-api.js';
import React from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import Container from 'react-bootstrap/Container';
import Button from 'react-bootstrap/Button';

function App() {
  
  const [modelsLoaded, setModelsLoaded] = React.useState(false);
  const [captureVideoStarted, setCaptureVideoStarted] = React.useState(false);
  const [faceDetected, setFaceDetected] = React.useState(false);
  
  const [capturedImage, setCapturedImage] = React.useState(false);
  const [greeting, setGreeting] = React.useState();
  const [prediction, setPrediction] = React.useState();
  const [newImgPathBase64, setNewImgPathBase64] = React.useState('')

  const videoRef = React.useRef();
  const videoHeight = 480;
  const videoWidth = 640;
  const canvasRef = React.useRef();

  React.useEffect(() => {

    const loadModels = async () => {
      const MODEL_URL = '/models';
      await fetch("https://app1.ouicodedata.com:8000/api")
        .then(response => response.json())
        .then(data=> {
            setGreeting(data);
      });
      Promise.all([
        faceapi.nets.tinyFaceDetector.load(MODEL_URL),
        faceapi.nets.faceLandmark68Net.load(MODEL_URL),
        faceapi.nets.faceRecognitionNet.load(MODEL_URL),
        //faceapi.nets.faceExpressionNet.load(MODEL_URL),
      ]).then(setModelsLoaded(true));
    }
    loadModels();
  }, []);

  const startVideo = () => {
    setCaptureVideoStarted(true);
    navigator.mediaDevices
      .getUserMedia({ video: {} })
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
          width: videoRef.current.width,
          height: videoRef.current.height
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
      
      //Create a new image with the same size of the video cam
      var img = new Image(videoWidth,videoHeight)
      //Crea a blob
      const blob = await imageCapture.takePhoto();
      var objectURL = URL.createObjectURL(blob);
      // Assign blob to image
      img.src = objectURL;
      
  
      console.log(
        `Width ${detection.imageWidth} and Height ${detection.imageHeight}`
      );
      console.log("detection : ",img)
      extractFaceFromBox(img, detection.box);
    
    } else {
      setFaceDetected(false)
    }

  }

  async function extractFaceFromBox(imageRef, box) {
    const regionsToExtract = [
      new faceapi.Rect(box.x + 13 , box.y - 18 , box.width -13 , box.height + 18)
    ];
    console.log("Box : ", box)
    let faceImages = await faceapi.extractFaces(imageRef, regionsToExtract);
    
    if (faceImages.length === 0) {
      console.log("No face found");
    } else {
      setCapturedImage(true)
      const outputImage = new Image();
      //console.log("Face canvas : ",faceImages[0],"imageref :", faceImages)
      outputImage.src = faceImages[0].toDataURL();
      //faceImages.forEach((cnv) => {
      //  console.log("Face : ", cnv)
      //  outputImage.src = cnv.toDataURL();
      //  setCropped(cnv.toDataURL());
      //});
      setNewImgPathBase64(outputImage.src);

      // Create a blob from cropped image and send to
      let blob_image = await fetch(outputImage.src).then(r => r.blob());
      
      //The canvas of the extracted image
      //document.body.appendChild(faceImages[0]);

      const preds = await handleSubmission(blob_image)
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
    fetch("https://app1.ouicodedata.com:8000/api/image", postData)
    .then(response => response.json())
    .then(object => {const json = {"gender":Object.keys(object)[0], "p":Object.values(object)[0]}
                    setPrediction(json)})
    .catch((error) => { console.error(error); });
  };

  const closeWebcam = () => {
    videoRef.current.pause();
    videoRef.current.srcObject.getTracks()[0].stop();
    setCaptureVideoStarted(false);
  }

  return (
    <Container >
      <Container style={{ textAlign: 'center', padding: '10px' }}>
        <Container>
          {greeting ? <p>API in online </p>: <p>API in offline </p> }
        </Container>
        {
          captureVideoStarted && modelsLoaded ?
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
        captureVideoStarted ?
          modelsLoaded ?
              <Container style={{ display: 'flex', justifyContent: 'center', padding: '10px' }}>
                <video ref={videoRef} height={videoHeight} width={videoWidth} onPlay={handleVideoOnPlay} 
                        style={{ borderRadius: '10px' }} />
                <canvas ref={canvasRef} style={{ position: 'absolute' }} />
              </Container>
              : <Container >Loading...</Container>
          :<></>
      }
      {captureVideoStarted ?
        <Container style={{ display: 'flex', justifyContent: 'center', padding: '10px' }}>
          <Button onClick={screenShot} disabled={!faceDetected} style={{ cursor: 'pointer', backgroundColor: 'blue', color: 'white', padding: '15px', fontSize: '25px', border: 'none', borderRadius: '10px' }}>
            Detect gender
          </Button>
        </Container>
        : <></>
      }

      { capturedImage ? 
        <Container fluid>
          <img src={newImgPathBase64} alt='' style={{float:"left", paddingRight: "5px"}}/>
          {prediction ? 
            <span>Predicted as <strong>{prediction.gender}</strong> with a probability of <strong>{Number(prediction.p).toFixed(2)} %</strong>  </span> 
            : <> <span>Fetching predictions...</span></>
          }
        </Container> 
        :<></>
      }

    </Container>
  );
}

export default App;