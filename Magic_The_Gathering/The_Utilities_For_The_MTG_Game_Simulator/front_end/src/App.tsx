import { useEffect } from 'react';
import './App.css';
import ButtonDisplayer from "./ButtonDisplayer";
import Displayer from "./Displayer";
import TableDisplayer from "./TableDisplayer";

function App() {
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