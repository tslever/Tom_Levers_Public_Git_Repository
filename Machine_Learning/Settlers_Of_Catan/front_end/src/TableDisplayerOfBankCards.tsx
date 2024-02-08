import TableDisplayer from "./TableDisplayer"

function TableDisplayerOfBankCards() {
  const body_data_for_table_describing_bank_cards = [[0.56, 0.08, 0.08, 0.08, 0.20, 19, 19, 19, 19, 19]]
  const column_group_for_table_describing_bank_cards = <colgroup>
    <col style = { { width: '10%' } }/>
    <col style = { { width: '10%' } }/>
    <col style = { { width: '10%' } }/>
    <col style = { { width: '10%' } }/>
    <col style = { { width: '10%' } }/>
    <col style = { { width: '10%' } }/>
    <col style = { { width: '10%' } }/>
    <col style = { { width: '10%' } }/>
  </colgroup>
  const header_data_for_table_describing_bank_cards = ['P(Knight)', 'P(Road Building)', 'P(Year Of Plenty)', 'P(Monopoly)', 'P(Victory Point)', 'Brick', 'Grain', 'Lumber', 'Ore', 'Wool']
  const title_for_table_describing_bank_cards = <div>
    <h3>
      Table Describing Bank Cards
    </h3>
    P(E) represents probability that the next development card to be drawn is an E card.
  </div>
  return <TableDisplayer
    bodyData = { body_data_for_table_describing_bank_cards }
    colgroup = { column_group_for_table_describing_bank_cards }
    headerData = { header_data_for_table_describing_bank_cards }
    title = { title_for_table_describing_bank_cards }
    widthPercentage = { 100 }
  />
}

export default TableDisplayerOfBankCards;