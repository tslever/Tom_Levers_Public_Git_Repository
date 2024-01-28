import './App.css';
import BaseBoardDisplayer from './BaseBoardDisplayer'
import TableDisplayer from './TableDisplayer';

function App() {
  const column_group_for_table_describing_bank_cards = <colgroup>
    <col style = { { width: '12.5%' } }/>
    <col style = { { width: '12.5%' } }/>
    <col style = { { width: '12.5%' } }/>
    <col style = { { width: '12.5%' } }/>
    <col style = { { width: '12.5%' } }/>
    <col style = { { width: '12.5%' } }/>
    <col style = { { width: '12.5%' } }/>
    <col style = { { width: '12.5%' } }/>
  </colgroup>
  const header_data_for_table_describing_bank_cards = ['P(Knight)', 'P(Progress)', 'P(Victory Point)', 'Brick', 'Grain', 'Lumber', 'Ore', 'Wool']
  const body_data_for_table_describing_bank_cards = [[0.56, 0.24, 0.20, 19, 19, 19, 19, 19]]
  const table_displayer_for_table_describing_bank_cards = <TableDisplayer
    headerData = { header_data_for_table_describing_bank_cards }
    bodyData = { body_data_for_table_describing_bank_cards }
    colgroup = { column_group_for_table_describing_bank_cards }
    title = <div>
      <h2>
        Table Describing Bank Cards
      </h2>
      P(E) represents probability that the next development card to be drawn is an E card.
    </div>
    widthPercentage = { 100 }
  />
  const body_data_for_primary_table_displayer = [
    [<BaseBoardDisplayer/>],
    [table_displayer_for_table_describing_bank_cards],
    ['Table Describing Development Hand Cards'],
    ['Table Describing Messages']
  ];
  const column_group_for_primary_table = <colgroup>
    <col style = { { width: '100%' } }/>
  </colgroup>
  const primary_table_displayer = <TableDisplayer
    bodyData = { body_data_for_primary_table_displayer }
    colgroup = { column_group_for_primary_table }
    widthPercentage = { 100 }
  />
  return (
    primary_table_displayer
  );
}

export default App;