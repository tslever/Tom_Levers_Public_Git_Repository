import TableDisplayer from './TableDisplayer';

function TableDisplayerOfActivePlayersHandCards() {
    const body_data_for_table_describing_active_players_hand_cards = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    const column_group_for_table_describing_active_players_hand_cards = <colgroup>
      <col style = { { width: '10%' } }/>
      <col style = { { width: '10%' } }/>
      <col style = { { width: '10%' } }/>
      <col style = { { width: '10%' } }/>
      <col style = { { width: '10%' } }/>
      <col style = { { width: '10%' } }/>
      <col style = { { width: '10%' } }/>
      <col style = { { width: '10%' } }/>
    </colgroup>
    const header_data_for_table_describing_active_players_hand_cards = ['Knight', 'Road Building', 'Year Of Plenty', 'Monopoly', 'Victory Point', 'Brick', 'Grain', 'Lumber', 'Ore', 'Wool']
    const title_for_table_describing_active_players_hand_cards = <div>
      <h3>
        Table Describing Active Player's Hand Cards
      </h3>
    </div>
    return <TableDisplayer
      bodyData = { body_data_for_table_describing_active_players_hand_cards }
      colgroup = { column_group_for_table_describing_active_players_hand_cards }
      headerData = { header_data_for_table_describing_active_players_hand_cards }
      title = { title_for_table_describing_active_players_hand_cards }
      widthPercentage = { 100 }
    />
}

export default TableDisplayerOfActivePlayersHandCards;