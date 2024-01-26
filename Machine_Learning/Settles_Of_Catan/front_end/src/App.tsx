import './App.css';
import BaseBoardDisplayer from './BaseBoardDisplayer'
import TableDisplayer from './TableDisplayer';

function App() {
  const bodyData = [[<BaseBoardDisplayer/>], ['Table Describing Cards'], ['Table Describing Messages']];
  const colgroup = <colgroup>
    <col style = { { width: '100%' } }/>
  </colgroup>
  return (
    <TableDisplayer bodyData = { bodyData } colgroup = { colgroup } widthPercentage = { 100 }/>
  );
}

export default App;