import './App.css';
import CanvasDisplayer from './CanvasDisplayer'

function App() {
  
  /*async function Executor() {
    const response = await fetch('http://localhost:8080');
    const json = await response.json();
    const rowWithText = [json];
    const newBodyData = [];
    newBodyData.push(rowWithText);
    for (let i = 0; i < bodyData.length; i++) {
      newBodyData.push(bodyData[i]);
    }
    setBodyData(newBodyData);
  }*/

  /*const [bodyData, setBodyData] = useState<string[][]>([]);

  function CaretakerOfIndicatorThatButtonHasBeenClicked() {
    const onClick = async () => {
      await Executor();
    };
    return onClick
  }*/

  return (
    <CanvasDisplayer aspectRatio = {16/9} backgroundColor = '#ff0000' widthPercentage = {100}/>
  );
}

export default App;