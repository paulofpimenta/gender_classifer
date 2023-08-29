import { useNavigate } from 'react-router-dom';
  
function Main() {
  const navigate = useNavigate();
  
  const goToSecondsComp = () => {
  
    // This will navigate to second component
    navigate('/webcam'); 
  };
 
  
  return (
    <div className="App">
      <header className="App-header">
        <p>Main components</p>
        <button onClick={goToSecondsComp}>go to 2nd </button>
      </header>
    </div>
  );
}
  
export default Main;