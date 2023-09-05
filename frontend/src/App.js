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

  const videoRef = React.useRef();
  const videoHeight = 480;
  const videoWidth = 640;
  const canvasRef = React.useRef();
  const [newImgPathBase64, setNewImgPathBase64] = React.useState('')


  React.useEffect(() => {

    const loadModels = async () => {
      const MODEL_URL = '/models';
      console.log("MODL URL : ", MODEL_URL)

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
      const displaySize = {
        width: videoRef.current.width,
        height: videoRef.current.height
      };
      
      const resizeDetection = faceapi.resizeResults(detection, displaySize);
      // faceapi.draw.drawFaceLandmarks(canvasRef.current, resizeDetections);
      canvasRef.current
        .getContext("2d")
        .clearRect(0, 0, displaySize.width, displaySize.height);
      faceapi.draw.drawDetections(canvasRef.current, resizeDetection);
  
      console.log(
        `Width ${detection.box._width} and Height ${detection.box._height}`
      );
      
      console.log(detection)
      extractFaceFromBox(img, detection.box);
      
      // Get prediction from API
      const formData = new FormData();

      formData.append('File', img);

      //extractFace(videoRef, x, y, width, height);
    } else {
      setFaceDetected(false)
    }

  }

};


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

      // setPic(faceImages.toDataUrl);
      console.log("face found ");
      //document.body.appendChild(outputImage);

    }
  }

  const handleSubmission = (image) => {
    const formData = new FormData();

    formData.append('File', image);

    fetch(
        'https://localhost:8000/image',
        {
            method: 'POST',
            body: formData,
        }
    )
        .then((response) => response.json())
        .then((result) => {
            console.log('Success:', result);
        })
        .catch((error) => {
            console.error('Error:', error);
        });
  };


  const closeWebcam = () => {
    videoRef.current.pause();
    videoRef.current.srcObject.getTracks()[0].stop();
    setCaptureVideo(false);
  }

  return (
    <Container>
      <Container style={{ textAlign: 'center', padding: '10px' }}>
        {
          captureVideo && modelsLoaded ?
            <Button onClick={closeWebcam} style={{ cursor: 'pointer', backgroundColor: 'green', color: 'white', padding: '15px', fontSize: '25px', border: 'none', borderRadius: '10px' }}>
              Close Webcam
            </Button>
            :
            <Button onClick={startVideo} style={{ cursor: 'pointer', backgroundColor: 'green', color: 'white', padding: '15px', fontSize: '25px', border: 'none', borderRadius: '10px' }}>
              Open Webcam
            </Button>
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
      { faceDetected ?
        <Container style={{ display: 'flex', justifyContent: 'center', padding: '10px' }}>
          <Button onClick={screenShot} onChange={uploadPicture} style={{ cursor: 'pointer', backgroundColor: 'blue', color: 'white', padding: '15px', fontSize: '25px', border: 'none', borderRadius: '10px' }}>
            Detect gender
          </Button>
        </Container> : 
        <></>
      } 


      <Container> 
        { capturedImage ? 
            <Container style={{ display: 'flex', justifyContent: 'center', padding: '10px' }}>
              <img src={newImgPathBase64} alt=''/>
            </Container> : 
            <></>
        }
      </Container>

    </Container>
  );
}

export default App;


