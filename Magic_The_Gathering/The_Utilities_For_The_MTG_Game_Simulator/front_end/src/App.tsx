import { useState } from 'react';
import './App.css';
import ButtonDisplayer from "./ButtonDisplayer";
import Displayer from "./Displayer";
import TableDisplayer from "./TableDisplayer";

function App() {
  
  function Caretaker() {
    const [bodyData, setBodyData] = useState<string[][]>([]);
    const onClick = async () => {
      try {
        const response = await fetch('http://localhost:8080');
        const text = await response.text();
        const rowWithText = [text];
        const newBodyData = [];
        newBodyData.push(rowWithText);
        for (let i = 0; i < bodyData.length; i++) {
          newBodyData.push(bodyData[i]);
        }
        setBodyData(newBodyData);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };
    return {
      "bodyData": bodyData,
      "onClick": onClick
    }
  }
  const jsonObjectWithBodyDataAndOnClick = Caretaker();
  
  return (
    <Displayer>
      <ButtonDisplayer
        label = "Next"
        onClick = { jsonObjectWithBodyDataAndOnClick.onClick }
      />
      <TableDisplayer
        headerData = { ["Action"] }
        bodyData = { jsonObjectWithBodyDataAndOnClick.bodyData }
      />
    </Displayer>
  );
}

export default App;