import TableDisplayer from "./TableDisplayer"

function TableDisplayerOfNonactivePlayersHandCards() {
  const body_data_for_table_describing_nonactive_players_hand_cards = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
  ]
  const column_group_for_table_describing_nonactive_players_hand_cards = <colgroup>
    <col style = { { width: '16.667%' } }/>
    <col style = { { width: '16.667%' } }/>
    <col style = { { width: '16.667%' } }/>
    <col style = { { width: '16.667%' } }/>
    <col style = { { width: '16.666%' } }/>
    <col style = { { width: '16.666%' } }/>
  </colgroup>
  const header_data_for_table_describing_nonactive_players_hand_cards = ['Development Cards', 'Brick', 'Grain', 'Lumber', 'Ore', 'Wool']
  const title_for_table_describing_nonactive_players_hand_cards = <div>
    <h3>
      Table Describing Nonactive Players' Hand Cards
    </h3>
  </div>
  return <TableDisplayer
    bodyData = { body_data_for_table_describing_nonactive_players_hand_cards }
    colgroup = { column_group_for_table_describing_nonactive_players_hand_cards }
    headerData = { header_data_for_table_describing_nonactive_players_hand_cards }
    title = { title_for_table_describing_nonactive_players_hand_cards }
    widthPercentage = { 100 }
  />
}

export default TableDisplayerOfNonactivePlayersHandCards;