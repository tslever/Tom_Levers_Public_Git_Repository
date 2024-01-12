import { useEffect } from 'react';
import './App.css';
import ButtonDisplayer from "./ButtonDisplayer";
import Displayer from "./Displayer";
import TableDisplayer from "./TableDisplayer";

function App() {
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('http://localhost:8080/echo');
        const data = await response.text();

        // Handle the data or do something with it
        console.log(data);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
  }, []); // Empty dependency array ensures the effect runs once when the component mounts

  return (
    <Displayer>
      <ButtonDisplayer label = "Next"/>
      <TableDisplayer
        headerData = { ["Action"] }
      />
    </Displayer>
  );
}

export default App;