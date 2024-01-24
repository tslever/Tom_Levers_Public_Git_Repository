import './App.css';
import BaseBoardDisplayer from './BaseBoardDisplayer'

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
    <BaseBoardDisplayer/>
  );
}

export default App;