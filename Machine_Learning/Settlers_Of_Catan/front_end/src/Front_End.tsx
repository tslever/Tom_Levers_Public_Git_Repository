import ActionDisplayer from './ActionDisplayer';
import BaseBoardDisplayer from './BaseBoardDisplayer'
import Displayer from './Displayer';
import { useState } from 'react';
import TableDisplayer from './TableDisplayer';
import TableDisplayerOfActivePlayersHandCards from './TableDisplayerOfActivePlayersHandCards';
import TableDisplayerOfBankCards from './TableDisplayerOfBankCards';
import TableDisplayerOfNonactivePlayersHandCards from './TableDisplayerOfNonactivePlayersHandCards';
import TableDisplayerOfMessages from './TableDisplayerOfMessages';
import TableDisplayerOfMenuOfActions from './TableDisplayerOfMenuOfActions';

function Front_End() {

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
  const [listOfActionDisplayers, setListOfActions] = useState([displayer_of_action_click_me_to_get_started]);


  const column_group_for_table_for_base_board_displayer_and_menu_of_actions = <colgroup>
    <col style = { { width: '50%' } }/>
    <col style = { { width: '50%' } }/>
  </colgroup>
  const table_displayer_for_base_board_displayer_and_menu_of_actions = <TableDisplayer
    bodyData = { [[<BaseBoardDisplayer/>, <TableDisplayerOfMenuOfActions listOfActionDisplayers = {listOfActionDisplayers}/>]] }
    colgroup = { column_group_for_table_for_base_board_displayer_and_menu_of_actions }
    widthPercentage = { 100 }
  />

  const body_data_for_primary_table_displayer = [
    [table_displayer_for_base_board_displayer_and_menu_of_actions],
    [<TableDisplayerOfBankCards/>],
    [<TableDisplayerOfActivePlayersHandCards/>],
    [<TableDisplayerOfNonactivePlayersHandCards/>],
    [<TableDisplayerOfMessages/>]
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