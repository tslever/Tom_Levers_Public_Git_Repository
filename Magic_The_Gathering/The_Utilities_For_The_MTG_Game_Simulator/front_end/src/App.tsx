import { useState } from 'react';
import './App.css';
import ButtonDisplayer from "./ButtonDisplayer";
import Displayer from "./Displayer";
import TableDisplayer from "./TableDisplayer";

function App() {
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

  return (
    <Displayer>
      <ButtonDisplayer
        label = "Next"
        onClick = { onClick }
      />
      <TableDisplayer
        headerData = { ["Action"] }
        bodyData = { bodyData }
      />
    </Displayer>
  );
}

export default App;