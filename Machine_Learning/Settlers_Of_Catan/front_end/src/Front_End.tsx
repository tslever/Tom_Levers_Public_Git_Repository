import ActionDisplayer from './ActionDisplayer';
import BaseBoardDisplayer from './BaseBoardDisplayer'
import Displayer from './Displayer';
import { useState } from 'react';
import TableDisplayer from './TableDisplayer';

function Front_End() {

    // table displayer for describing active player's hand cards
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
    const table_displayer_for_table_describing_active_players_hand_cards = <TableDisplayer
      bodyData = { body_data_for_table_describing_active_players_hand_cards }
      colgroup = { column_group_for_table_describing_active_players_hand_cards }
      headerData = { header_data_for_table_describing_active_players_hand_cards }
      title = { title_for_table_describing_active_players_hand_cards }
      widthPercentage = { 100 }
    />

  // table displayer for describing bank cards
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
  const table_displayer_for_table_describing_bank_cards = <TableDisplayer
    bodyData = { body_data_for_table_describing_bank_cards }
    colgroup = { column_group_for_table_describing_bank_cards }
    headerData = { header_data_for_table_describing_bank_cards }
    title = { title_for_table_describing_bank_cards }
    widthPercentage = { 100 }
  />

  // table describing nonactive players' hand cards
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
  const table_displayer_for_table_describing_nonactive_players_hand_cards = <TableDisplayer
    bodyData = { body_data_for_table_describing_nonactive_players_hand_cards }
    colgroup = { column_group_for_table_describing_nonactive_players_hand_cards }
    headerData = { header_data_for_table_describing_nonactive_players_hand_cards }
    title = { title_for_table_describing_nonactive_players_hand_cards }
    widthPercentage = { 100 }
  />

  // table of messages
  const body_data_for_table_of_messages = [
    []
  ]
  const column_group_for_table_of_messages = <colgroup>
    <col style = { { width: '100%' } }/>
  </colgroup>
  const header_data_for_table_of_messages = ['Messages']
  const title_for_table_of_messages = <div>
    <h3>
      Table Of Messages
    </h3>
  </div>
  const table_displayer_for_table_of_messages = <TableDisplayer
    bodyData = { body_data_for_table_of_messages }
    colgroup = { column_group_for_table_of_messages }
    headerData = { header_data_for_table_of_messages }
    title = { title_for_table_of_messages }
    widthPercentage = { 100 }
  />

  async function act(action: string) {
    const url = 'http://localhost:5000';
    const JSON_object = { action: action };
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(JSON_object)
    });  
    const json = await response.json();
    console.log(json);
    const displayer_of_action = <ActionDisplayer
      backgroundColor='#ffffff'
      act = { act }
    >
      { json.next_action }
    </ActionDisplayer>
    setListOfActions([displayer_of_action]);
  }

  const displayer_of_action_click_me_to_get_started = <ActionDisplayer
    backgroundColor = '#ffffff'
    act = { act }
  >
    Click me to get started.
  </ActionDisplayer>
  const [listOfActions, setListOfActions] = useState([displayer_of_action_click_me_to_get_started]);

  const column_group_for_menu_of_actions = <colgroup>
    <col style = { { width: '100%' } }/>
  </colgroup>
  const header_data_for_menu_of_actions = ['Menu Of Actions']
  const body_data_for_menu_of_actions = [
    listOfActions
  ]
  const table_displayer_for_menu_of_actions = <TableDisplayer
    headerData = { header_data_for_menu_of_actions }
    bodyData = { body_data_for_menu_of_actions }
    colgroup = { column_group_for_menu_of_actions }
    title = <div>
      <h3>
        Settlers Of Catan, Monte Carlo Tree Search, And Neural Networks
      </h3>
    </div>
    widthPercentage = { 100 }
  />

  const column_group_for_table_for_base_board_displayer_and_menu_of_actions = <colgroup>
    <col style = { { width: '50%' } }/>
    <col style = { { width: '50%' } }/>
  </colgroup>
  const table_displayer_for_base_board_displayer_and_menu_of_actions = <TableDisplayer
    bodyData = { [[<BaseBoardDisplayer/>, table_displayer_for_menu_of_actions]] }
    colgroup = { column_group_for_table_for_base_board_displayer_and_menu_of_actions }
    widthPercentage = { 100 }
  />

  const body_data_for_primary_table_displayer = [
    [table_displayer_for_base_board_displayer_and_menu_of_actions],
    [table_displayer_for_table_describing_bank_cards],
    [table_displayer_for_table_describing_active_players_hand_cards],
    [table_displayer_for_table_describing_nonactive_players_hand_cards],
    [table_displayer_for_table_of_messages]
  ];
  const column_group_for_primary_table = <colgroup>
    <col style = { { width: '100%' } }/>
  </colgroup>
  const row_styles_for_primary_table = [
    { 'backgroundColor': 'rgb(255, 248, 195)' },
    { 'backgroundColor': 'rgb(255, 243, 137)' },
    { 'backgroundColor': 'rgb(255, 248, 195)' },
    { 'backgroundColor': 'rgb(255, 243, 137)' }
  ]
  const primary_table_displayer = <TableDisplayer
    bodyData = { body_data_for_primary_table_displayer }
    colgroup = { column_group_for_primary_table }
    rowStyles = { row_styles_for_primary_table }
    widthPercentage = { 100 }
  />
  return (
    <Displayer backgroundColor = 'rgb(255, 248, 195)'>
      { primary_table_displayer }
    </Displayer>
  );
}

export default Front_End;