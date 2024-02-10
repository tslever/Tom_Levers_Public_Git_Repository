import ActionDisplayer from './ActionDisplayer';
import Displayer from './Displayer';
import { useState } from 'react';
import PrimaryTableDisplayer from './PrimaryTableDisplayer';

function Front_End() {

  const [listOfMessages, setListOfMessages] = useState([''])
  const [listOfPossibleActions, setListOfPossibleActions] = useState(['Click me to get started.'])

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
    setListOfMessages([...listOfMessages, json.action_completed])
    setListOfPossibleActions(json.list_of_possible_actions);
  }

  function mapToActionDisplayer(possibleAction: string) {
    return <ActionDisplayer
      act = { act }
      backgroundColor = 'white'
    >
      { possibleAction }
    </ActionDisplayer>
  }

  const listOfActionDisplayers = listOfPossibleActions.map(mapToActionDisplayer);

  return (
    <Displayer backgroundColor = 'rgb(255, 248, 195)'>
      <PrimaryTableDisplayer
        listOfActionDisplayers = { listOfActionDisplayers }
        listOfMessages = { listOfMessages }
      />
    </Displayer>
  );
}

export default Front_End;