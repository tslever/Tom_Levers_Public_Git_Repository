import './App.css';
import ButtonDisplayer from "./ButtonDisplayer";
import Displayer from "./Displayer";
import TableDisplayer from "./TableDisplayer";

function App() {
  const bodyData : String[][] = [
    ["Line 1"],
    ["Line 2"]
  ]
  return (
    <Displayer>
      <ButtonDisplayer label = "Next"/>
      <TableDisplayer
        headerData = { ["Action"] }
        bodyData = { bodyData }
      />
    </Displayer>
  );
}

export default App;